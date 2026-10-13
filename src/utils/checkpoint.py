"""
Checkpointing, Training History Tracking, and Visualization Utility.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from src.utils.logger import setup_logger

logger = setup_logger("checkpoint_utils")


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, Any],
    filepath: str,
    class_names: Optional[List[str]] = None,
    scheduler: Optional[Any] = None
) -> None:
    """Save PyTorch model weights and state."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "class_names": class_names,
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, filepath)
    logger.info(f"Checkpoint saved to {filepath}")


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = "cpu"
) -> Dict[str, Any]:
    """Load model weights and state from checkpoint."""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    logger.info(f"Loaded checkpoint from {filepath} (Epoch {checkpoint.get('epoch', 'N/A')})")
    return checkpoint


def save_training_history_and_plots(
    history: List[Dict[str, Any]],
    output_dir: str,
    model_name: str
) -> None:
    """
    Save training history CSV and generate loss/accuracy training curve plots.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Save CSV
    df = pd.DataFrame(history)
    csv_file = out_path / "training_history.csv"
    df.to_csv(csv_file, index=False)
    logger.info(f"Training history saved to {csv_file}")

    epochs = df["epoch"].values

    # 2. Training and Validation Loss Plot
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, df["train_loss"], marker="o", label="Train Loss", color="#1f77b4", linewidth=2)
    plt.plot(epochs, df["val_loss"], marker="s", label="Validation Loss", color="#d62728", linewidth=2)
    plt.title(f"{model_name} — Training & Validation Loss", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=11, fontweight="bold")
    plt.ylabel("Loss (CrossEntropy)", fontsize=11, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path / "training_loss.png", dpi=300)
    plt.close()

    # 3. Training and Validation Accuracy Plot
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, df["train_acc"], marker="o", label="Train Accuracy", color="#2ca02c", linewidth=2)
    plt.plot(epochs, df["val_acc"], marker="s", label="Validation Accuracy", color="#ff7f0e", linewidth=2)
    plt.title(f"{model_name} — Training & Validation Accuracy", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=11, fontweight="bold")
    plt.ylabel("Accuracy", fontsize=11, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path / "training_accuracy.png", dpi=300)
    plt.close()

    # 4. Validation Macro F1 Score Plot
    if "val_f1_macro" in df.columns:
        plt.figure(figsize=(9, 5))
        plt.plot(epochs, df["val_f1_macro"], marker="^", label="Val Macro F1", color="#9467bd", linewidth=2)
        plt.title(f"{model_name} — Validation Macro F1 Score", fontsize=13, fontweight="bold", pad=12)
        plt.xlabel("Epoch", fontsize=11, fontweight="bold")
        plt.ylabel("Macro F1", fontsize=11, fontweight="bold")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(out_path / "validation_f1.png", dpi=300)
        plt.close()


def save_metrics_report(metrics: Dict[str, Any], filepath: str) -> None:
    """Save metrics dictionary as JSON file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Evaluation metrics saved to {filepath}")
