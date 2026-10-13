"""
Confusion Matrix computation and visualization module.
"""

from pathlib import Path
from typing import List, Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch


def plot_and_save_confusion_matrix(
    y_true: Union[np.ndarray, List[int], torch.Tensor],
    y_pred: Union[np.ndarray, List[int], torch.Tensor],
    class_names: List[str],
    output_path: str,
    title: str = "Confusion Matrix",
    normalize: bool = False
) -> np.ndarray:
    """
    Generate, format, and save a high-resolution confusion matrix plot.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    
    plt.figure(figsize=(10, 8))
    if normalize:
        cm_display = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
        fmt = ".2f"
        vmax = 1.0
        cbar_label = "Normalized Proportion"
    else:
        cm_display = cm
        fmt = "d"
        vmax = None
        cbar_label = "Sample Count"

    clean_names = [name.replace("_", " ").title() for name in class_names]
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=clean_names,
        yticklabels=clean_names,
        cbar_kws={"label": cbar_label},
        linewidths=0.5
    )

    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Disease Class", fontsize=11, fontweight="bold")
    plt.ylabel("True Disease Class", fontsize=11, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()

    return cm
