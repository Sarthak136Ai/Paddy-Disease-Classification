"""
ResNet50 Architecture for Paddy Leaf Disease Classification.
Supports ImageNet pre-trained weights, classifier replacement, and fine-tuning.
"""

from typing import Tuple
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNet50Transfer(nn.Module):
    """
    ResNet50 Transfer Learning Model.
    """
    def __init__(
        self,
        num_classes: int = 10,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout_rate: float = 0.3
    ):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = resnet50(weights=weights)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_num_parameters(self) -> Tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


def build_resnet50(num_classes: int = 10, pretrained: bool = True, **kwargs) -> ResNet50Transfer:
    return ResNet50Transfer(num_classes=num_classes, pretrained=pretrained, **kwargs)
