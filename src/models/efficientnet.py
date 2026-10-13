"""
EfficientNet-B0 Architecture for Paddy Leaf Disease Classification.
"""

from typing import Tuple
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class EfficientNetB0Transfer(nn.Module):
    """
    EfficientNet-B0 Transfer Learning Model.
    """
    def __init__(
        self,
        num_classes: int = 10,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout_rate: float = 0.3
    ):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_num_parameters(self) -> Tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


def build_efficientnet_b0(num_classes: int = 10, pretrained: bool = True, **kwargs) -> EfficientNetB0Transfer:
    return EfficientNetB0Transfer(num_classes=num_classes, pretrained=pretrained, **kwargs)
