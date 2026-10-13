"""
Evaluation package initialization.
"""
from src.evaluation.metrics import compute_metrics
from src.evaluation.confusion_matrix import plot_and_save_confusion_matrix
from src.evaluation.evaluate import evaluate_on_test_set

__all__ = [
    "compute_metrics",
    "plot_and_save_confusion_matrix",
    "evaluate_on_test_set"
]
