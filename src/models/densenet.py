"""
DenseNet121 Architecture for Paddy Leaf Disease Classification.
"""

from typing import Tuple
import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights


class DenseNet121Transfer(nn.Module):
    """
    DenseNet121 Transfer Learning Model.
    """
    def __init__(
        self,
        num_classes: int = 10,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout_rate: float = 0.3
    ):
        super().__init__()
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        self.backbone = densenet121(weights=weights)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        in_features = self.backbone.classifier.in_features
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


def build_densenet121(num_classes: int = 10, pretrained: bool = True, **kwargs) -> DenseNet121Transfer:
    return DenseNet121Transfer(num_classes=num_classes, pretrained=pretrained, **kwargs)
