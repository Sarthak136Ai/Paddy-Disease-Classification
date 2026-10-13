"""
Comprehensive Evaluation Metrics for Paddy Leaf Disease Classification.
Computes Accuracy, Precision, Recall, F1-Score (Macro & Weighted), and Per-Class Metrics.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


def compute_metrics(
    y_true: Union[np.ndarray, List[int], torch.Tensor],
    y_pred: Union[np.ndarray, List[int], torch.Tensor],
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Calculate full multi-class classification evaluation metrics.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    accuracy = float(accuracy_score(y_true, y_pred))
    precision_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    recall_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    precision_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    recall_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # Per-class metrics
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0).tolist()
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0).tolist()
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    report_str = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0
    )

    per_class_metrics = {}
    if class_names is not None:
        for idx, name in enumerate(class_names):
            per_class_metrics[name] = {
                "precision": round(per_class_precision[idx], 4),
                "recall": round(per_class_recall[idx], 4),
                "f1_score": round(per_class_f1[idx], 4)
            }

    return {
        "accuracy": round(accuracy, 4),
        "precision_macro": round(precision_macro, 4),
        "recall_macro": round(recall_macro, 4),
        "f1_macro": round(f1_macro, 4),
        "precision_weighted": round(precision_weighted, 4),
        "recall_weighted": round(recall_weighted, 4),
        "f1_weighted": round(f1_weighted, 4),
        "per_class": per_class_metrics,
        "classification_report_dict": report_dict,
        "classification_report_str": report_str
    }
