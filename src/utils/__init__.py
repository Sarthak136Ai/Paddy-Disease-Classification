"""
Utils package initialization.
"""
from src.utils.logger import setup_logger

try:
    from src.utils.seed import set_seed
except ImportError:
    set_seed = None

try:
    from src.utils.checkpoint import (
        save_checkpoint,
        load_checkpoint,
        save_training_history_and_plots,
        save_metrics_report
    )
except ImportError:
    save_checkpoint = None
    load_checkpoint = None
    save_training_history_and_plots = None
    save_metrics_report = None

__all__ = [
    "set_seed",
    "setup_logger",
    "save_checkpoint",
    "load_checkpoint",
    "save_training_history_and_plots",
    "save_metrics_report"
]
