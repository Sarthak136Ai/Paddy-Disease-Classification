"""
Model Evaluation and Inference Benchmarking on Untouched Test Set.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.confusion_matrix import plot_and_save_confusion_matrix
from src.evaluation.metrics import compute_metrics
from src.training.losses import get_loss_function
from src.utils.checkpoint import load_checkpoint, save_metrics_report
from src.utils.logger import setup_logger

logger = setup_logger("evaluator")


def evaluate_on_test_set(
    model: nn.Module,
    model_name: str,
    checkpoint_path: str,
    test_loader: DataLoader,
    class_names: List[str],
    output_dir: str,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Evaluate a saved model checkpoint against the test dataset:
    - Multi-class accuracy, precision, recall, macro/weighted F1
    - Per-class breakdown
    - Confusion matrices (raw counts and normalized)
    - Inference latency benchmark (ms per image)
    - Model parameter count and disk size
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading checkpoint for [{model_name}] from {checkpoint_path}...")
    load_checkpoint(checkpoint_path, model, device=str(device))
    model.eval()

    criterion = get_loss_function("CrossEntropyLoss")
    running_loss = 0.0
    total_samples = 0

    all_preds: List[int] = []
    all_targets: List[int] = []
    all_probs: List[np.ndarray] = []

    # Inference timing
    start_time = time.time()

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, targets)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_preds.extend(preds.tolist())
            all_targets.extend(targets.cpu().numpy().tolist())
            all_probs.append(probs)

    total_test_time = time.time() - start_time
    latency_ms_per_image = (total_test_time / max(total_samples, 1)) * 1000.0

    test_loss = running_loss / max(total_samples, 1)
    metrics = compute_metrics(
        y_true=all_targets,
        y_pred=all_preds,
        class_names=class_names
    )
    metrics["test_loss"] = round(test_loss, 4)
    metrics["latency_ms_per_image"] = round(latency_ms_per_image, 2)
    metrics["total_test_samples"] = total_samples

    # Model Size and Parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    checkpoint_size_mb = Path(checkpoint_path).stat().st_size / (1024 * 1024) if Path(checkpoint_path).exists() else 0.0

    metrics["total_parameters"] = total_params
    metrics["trainable_parameters"] = trainable_params
    metrics["checkpoint_size_mb"] = round(checkpoint_size_mb, 2)
    metrics["model_name"] = model_name

    logger.info(
        f"Test Results for [{model_name}] -> "
        f"Loss: {test_loss:.4f} | "
        f"Accuracy: {metrics['accuracy']:.4f} | "
        f"Macro F1: {metrics['f1_macro']:.4f} | "
        f"Latency: {latency_ms_per_image:.2f} ms/img | "
        f"Params: {total_params:,}"
    )

    # Confusion Matrices
    cm_path_raw = str(out_path / f"{model_name}_test_confusion_matrix.png")
    cm_path_norm = str(out_path / f"{model_name}_test_confusion_matrix_norm.png")
    
    plot_and_save_confusion_matrix(
        y_true=all_targets,
        y_pred=all_preds,
        class_names=class_names,
        output_path=cm_path_raw,
        title=f"{model_name} — Test Set Confusion Matrix (Counts)"
    )
    plot_and_save_confusion_matrix(
        y_true=all_targets,
        y_pred=all_preds,
        class_names=class_names,
        output_path=cm_path_norm,
        title=f"{model_name} — Test Set Confusion Matrix (Normalized)",
        normalize=True
    )

    # Save metrics report JSON
    metrics_file = str(out_path / f"{model_name}_test_metrics.json")
    save_metrics_report(metrics, metrics_file)

    return metrics
