"""
Validation and Evaluation Epoch Loop.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import compute_metrics


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: Optional[List[str]] = None
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """
    Run full validation over a dataloader and compute loss, metrics, and prediction arrays.
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0

    all_preds: List[int] = []
    all_targets: List[int] = []
    all_probs: List[np.ndarray] = []

    with torch.no_grad():
        for images, targets in dataloader:
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

    epoch_loss = running_loss / max(total_samples, 1)
    all_probs_arr = np.concatenate(all_probs, axis=0) if all_probs else np.array([])

    metrics = compute_metrics(
        y_true=all_targets,
        y_pred=all_preds,
        class_names=class_names
    )
    metrics["loss"] = round(epoch_loss, 4)

    return metrics, np.array(all_targets), np.array(all_preds), all_probs_arr
