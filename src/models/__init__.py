"""
Models Package and Model Factory for Paddy Leaf Disease Classification.
"""

from typing import Any, Dict, Tuple
import torch.nn as nn

from src.models.custom_cnn import CustomCNN, build_custom_cnn
from src.models.resnet import ResNet50Transfer, build_resnet50
from src.models.efficientnet import EfficientNetB0Transfer, build_efficientnet_b0
from src.models.densenet import DenseNet121Transfer, build_densenet121
from src.models.mobilenet import MobileNetV3Transfer, build_mobilenet_v3


MODEL_REGISTRY = {
    "custom_cnn": build_custom_cnn,
    "resnet50": build_resnet50,
    "efficientnet_b0": build_efficientnet_b0,
    "densenet121": build_densenet121,
    "mobilenet_v3": build_mobilenet_v3,
}


def get_model(model_name: str, num_classes: int = 10, **kwargs) -> nn.Module:
    """
    Factory function to instantiate models by registered name.
    """
    key = model_name.lower().replace("-", "_")
    if key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model architecture '{model_name}'. Available options: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[key](num_classes=num_classes, **kwargs)


__all__ = [
    "CustomCNN",
    "build_custom_cnn",
    "ResNet50Transfer",
    "build_resnet50",
    "EfficientNetB0Transfer",
    "build_efficientnet_b0",
    "DenseNet121Transfer",
    "build_densenet121",
    "MobileNetV3Transfer",
    "build_mobilenet_v3",
    "get_model",
    "MODEL_REGISTRY"
]
