"""
MobileNetV3 (Large) Architecture for Paddy Leaf Disease Classification.
"""

from typing import Tuple
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights


class MobileNetV3Transfer(nn.Module):
    """
    MobileNetV3-Large Transfer Learning Model.
    """
    def __init__(
        self,
        num_classes: int = 10,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout_rate: float = 0.3
    ):
        super().__init__()
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        self.backbone = mobilenet_v3_large(weights=weights)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_num_parameters(self) -> Tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


def build_mobilenet_v3(num_classes: int = 10, pretrained: bool = True, **kwargs) -> MobileNetV3Transfer:
    return MobileNetV3Transfer(num_classes=num_classes, pretrained=pretrained, **kwargs)
