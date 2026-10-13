"""
Loss Functions for Paddy Leaf Disease Classification.
"""

import torch
import torch.nn as nn
from typing import Optional


def get_loss_function(
    loss_type: str = "CrossEntropyLoss",
    weight: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0
) -> nn.Module:
    """
    Factory function for classification loss functions.
    """
    if loss_type.lower() in ["crossentropy", "crossentropyloss", "ce"]:
        return nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
