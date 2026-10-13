"""
Model Training and Optimization Orchestrator.
Manages epoch iterations, loss tracking, learning rate scheduling, early stopping,
checkpoint saving, and metric logging.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation.confusion_matrix import plot_and_save_confusion_matrix
from src.training.early_stopping import EarlyStopping
from src.training.losses import get_loss_function
from src.training.validate import validate_epoch
from src.utils.checkpoint import (
    save_checkpoint,
    save_training_history_and_plots,
    save_metrics_report
)
from src.utils.logger import setup_logger

logger = setup_logger("training_engine")


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int
) -> Tuple[float, float]:
    """
    Execute one complete training epoch.
    """
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs} [Train]", leave=False)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        _, preds = torch.max(outputs, 1)
        correct_predictions += torch.sum(preds == targets).item()
        total_samples += batch_size

        current_loss = running_loss / total_samples
        current_acc = correct_predictions / total_samples
        pbar.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.4f}")

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return float(epoch_loss), float(epoch_acc)


def train_model(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_names: List[str],
    config: Dict[str, Any],
    experiment_dir: str,
    device: Optional[torch.device] = None
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Full model training lifecycle:
    - Optimization (AdamW)
    - Learning rate scheduling (CosineAnnealingLR)
    - Early stopping on validation loss/F1
    - History recording & curves plotting
    - Saving best model checkpoint
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    exp_path = Path(experiment_dir)
    exp_path.mkdir(parents=True, exist_ok=True)

    # Hyperparameters
    train_cfg = config.get("training", {})
    epochs = train_cfg.get("epochs", 30)
    lr = float(train_cfg.get("learning_rate", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    
    criterion = get_loss_function(train_cfg.get("loss", "CrossEntropyLoss"))
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    es_cfg = train_cfg.get("early_stopping", {})
    early_stopping = EarlyStopping(
        patience=es_cfg.get("patience", 7),
        min_delta=es_cfg.get("min_delta", 0.001),
        monitor=es_cfg.get("monitor", "val_loss"),
        mode=es_cfg.get("mode", "min")
    )

    logger.info(f"--- Initiating Training for [{model_name}] on Device: {device} ---")
    logger.info(f"Total Epochs: {epochs}, Base LR: {lr}, Weight Decay: {weight_decay}")

    history: List[Dict[str, Any]] = []
    best_val_loss = float("inf")
    best_val_f1 = 0.0
    best_val_acc = 0.0
    best_epoch = 0
    best_metrics: Dict[str, Any] = {}
    best_model_path = str(exp_path / "best_model.pth")

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Train Epoch
        train_loss, train_acc = train_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_epochs=epochs
        )

        # Validation Epoch
        val_metrics, y_true, y_pred, _ = validate_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            class_names=class_names
        )

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        val_loss = val_metrics["loss"]
        val_acc = val_metrics["accuracy"]
        val_f1_macro = val_metrics["f1_macro"]
        epoch_duration = time.time() - epoch_start

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "val_precision_macro": round(val_metrics["precision_macro"], 4),
            "val_recall_macro": round(val_metrics["recall_macro"], 4),
            "val_f1_macro": round(val_f1_macro, 4),
            "learning_rate": current_lr,
            "duration_seconds": round(epoch_duration, 2)
        }
        history.append(epoch_record)

        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f} - Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} - Acc: {val_acc:.4f} - Macro F1: {val_f1_macro:.4f} | "
            f"Time: {epoch_duration:.1f}s"
        )

        # Check if this is the best checkpoint (based on validation loss)
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_val_f1 = val_f1_macro
            best_val_acc = val_acc
            best_epoch = epoch
            best_metrics = val_metrics

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
                filepath=best_model_path,
                class_names=class_names,
                scheduler=scheduler
            )

            # Save confusion matrix for best validation epoch
            plot_and_save_confusion_matrix(
                y_true=y_true,
                y_pred=y_pred,
                class_names=class_names,
                output_path=str(exp_path / "val_confusion_matrix.png"),
                title=f"{model_name} — Best Validation Confusion Matrix (Epoch {epoch})"
            )

        # Early stopping check
        if early_stopping(val_loss):
            logger.info(f"Early stopping triggered at epoch {epoch} (No improvement for {early_stopping.patience} epochs).")
            break

    total_training_time = time.time() - start_time
    logger.info(f"Training completed for [{model_name}] in {total_training_time:.2f}s ({total_training_time/60:.2f} min).")
    logger.info(f"Best Validation Epoch: {best_epoch} | Best Val Loss: {best_val_loss:.4f} | Best Val Acc: {best_val_acc:.4f} | Best Val F1: {best_val_f1:.4f}")

    # Save training curves & history
    save_training_history_and_plots(history, str(exp_path), model_name)

    # Save summary metadata
    summary = {
        "model_name": model_name,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_acc,
        "best_val_f1_macro": best_val_f1,
        "total_training_time_seconds": round(total_training_time, 2),
        "total_epochs_trained": len(history),
        "best_metrics": best_metrics
    }
    save_metrics_report(summary, str(exp_path / "train_summary.json"))

    return best_model_path, history, summary
