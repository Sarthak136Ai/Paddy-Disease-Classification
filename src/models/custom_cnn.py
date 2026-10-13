"""
Custom Convolutional Neural Network (Baseline Architecture).
Built from scratch with 4 convolutional feature extraction blocks, batch normalization,
adaptive global pooling, and regularized classification head.
"""

from typing import Dict, Any, Tuple
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Standard convolutional block: Conv2D -> BatchNorm2D -> ReLU -> MaxPool2D -> Dropout.
    """
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_rate)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CustomCNN(nn.Module):
    """
    Custom Baseline CNN for 10-class Paddy Leaf Disease Classification.
    """
    def __init__(self, num_classes: int = 10, in_channels: int = 3, dropout_rate: float = 0.4):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32, dropout_rate=0.05),   # Output: (32, 112, 112)
            ConvBlock(32, 64, dropout_rate=0.1),            # Output: (64, 56, 56)
            ConvBlock(64, 128, dropout_rate=0.15),          # Output: (128, 28, 28)
            ConvBlock(128, 256, dropout_rate=0.2)           # Output: (256, 14, 14)
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_pool(x)
        logits = self.classifier(x)
        return logits

    def get_num_parameters(self) -> Tuple[int, int]:
        """Return (total_params, trainable_params)."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


def build_custom_cnn(num_classes: int = 10, **kwargs) -> CustomCNN:
    """Factory helper for Custom CNN baseline."""
    kwargs.pop("pretrained", None)
    kwargs.pop("freeze_backbone", None)
    return CustomCNN(num_classes=num_classes, **kwargs)
