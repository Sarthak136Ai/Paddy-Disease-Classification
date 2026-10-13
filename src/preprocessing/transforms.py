"""
Image Preprocessing and Data Augmentation Pipelines.
Provides standardized torchvision transform pipelines for training, validation, and testing,
along with un-normalization utilities for visualization and explainability.
"""

from typing import Tuple
import torch
from torchvision import transforms


# Standard ImageNet normalization parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(
    image_size: Tuple[int, int] = (224, 224),
    mean: list = IMAGENET_MEAN,
    std: list = IMAGENET_STD,
    hflip_prob: float = 0.5,
    vflip_prob: float = 0.2,
    rotation_deg: int = 15,
    brightness_jitter: float = 0.2,
    contrast_jitter: float = 0.2,
    saturation_jitter: float = 0.2
) -> transforms.Compose:
    """
    Construct training transformations with robust geometric and photometric augmentations.
    """
    return transforms.Compose([
        transforms.Resize((int(image_size[0] * 1.14), int(image_size[1] * 1.14))),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=hflip_prob),
        transforms.RandomVerticalFlip(p=vflip_prob),
        transforms.RandomRotation(degrees=rotation_deg),
        transforms.ColorJitter(
            brightness=brightness_jitter,
            contrast=contrast_jitter,
            saturation=saturation_jitter
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


def get_val_transforms(
    image_size: Tuple[int, int] = (224, 224),
    mean: list = IMAGENET_MEAN,
    std: list = IMAGENET_STD
) -> transforms.Compose:
    """
    Construct deterministic validation transformations (no random augmentations).
    """
    return transforms.Compose([
        transforms.Resize((int(image_size[0] * 1.14), int(image_size[1] * 1.14))),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


def get_test_transforms(
    image_size: Tuple[int, int] = (224, 224),
    mean: list = IMAGENET_MEAN,
    std: list = IMAGENET_STD
) -> transforms.Compose:
    """
    Construct deterministic test transformations (identical to validation).
    """
    return transforms.Compose([
        transforms.Resize((int(image_size[0] * 1.14), int(image_size[1] * 1.14))),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


def denormalize_tensor(
    tensor: torch.Tensor,
    mean: list = IMAGENET_MEAN,
    std: list = IMAGENET_STD
) -> torch.Tensor:
    """
    Invert standard normalization on a (C, H, W) or (B, C, H, W) tensor for display/plotting.
    Clamps output to [0, 1].
    """
    mean_t = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std_t = torch.tensor(std, dtype=tensor.dtype, device=tensor.device)

    if tensor.ndim == 3:
        # Shape: (C, H, W)
        mean_t = mean_t.view(-1, 1, 1)
        std_t = std_t.view(-1, 1, 1)
    elif tensor.ndim == 4:
        # Shape: (B, C, H, W)
        mean_t = mean_t.view(1, -1, 1, 1)
        std_t = std_t.view(1, -1, 1, 1)

    denorm = tensor * std_t + mean_t
    return torch.clamp(denorm, 0.0, 1.0)
