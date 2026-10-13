"""
Early Stopping utility to halt training when validation metric stops improving.
"""

from typing import Dict, Any


class EarlyStopping:
    """
    Early stopping monitor.
    """
    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.001,
        monitor: str = "val_loss",
        mode: str = "min"
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode.lower()
        self.counter = 0
        self.best_score = None
        self.early_stop = False

        assert self.mode in ["min", "max"], "mode must be 'min' or 'max'"

    def __call__(self, current_value: float) -> bool:
        if self.best_score is None:
            self.best_score = current_value
            return False

        if self.mode == "min":
            improved = current_value < (self.best_score - self.min_delta)
        else:
            improved = current_value > (self.best_score + self.min_delta)

        if improved:
            self.best_score = current_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def state_dict(self) -> Dict[str, Any]:
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "monitor": self.monitor,
            "mode": self.mode,
            "counter": self.counter,
            "best_score": self.best_score,
            "early_stop": self.early_stop
        }
