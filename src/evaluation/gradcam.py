"""
Grad-CAM (Gradient-weighted Class Activation Mapping) Module.
Provides visual explainability by highlighting spatial leaf regions and lesions
influencing the deep learning model's classification decisions.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.preprocessing.transforms import get_val_transforms, denormalize_tensor


class GradCAM:
    """
    Grad-CAM implementation supporting CNN backbones (ResNet, EfficientNet, DenseNet, MobileNet, Custom CNN).
    """
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer or self._find_default_target_layer(model)
        
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        self._register_hooks()

    def _find_default_target_layer(self, model: nn.Module) -> nn.Module:
        """Heuristic to locate final convolutional layer for various architectures."""
        # Check ResNet
        if hasattr(model, "backbone") and hasattr(model.backbone, "layer4"):
            return model.backbone.layer4[-1]
        elif hasattr(model, "layer4"):
            return model.layer4[-1]

        # Check DenseNet
        if hasattr(model, "backbone") and hasattr(model.backbone, "features") and hasattr(model.backbone.features, "denseblock4"):
            return model.backbone.features.denseblock4
        elif hasattr(model, "features") and hasattr(model.features, "denseblock4"):
            return model.features.denseblock4

        # Check EfficientNet / MobileNet
        if hasattr(model, "backbone") and hasattr(model.backbone, "features"):
            return model.backbone.features[-1]
        elif hasattr(model, "features"):
            if isinstance(model.features, nn.Sequential):
                return model.features[-1]
            return model.features

        # Fallback: scan for last Conv2d
        last_conv = None
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        if last_conv is not None:
            return last_conv

        raise ValueError("Could not automatically determine target layer for Grad-CAM.")

    def _register_hooks(self) -> None:
        def forward_hook(module, input, output):
            if isinstance(output, torch.Tensor):
                self.activations = output.clone()
            else:
                self.activations = output

        def backward_hook(module, grad_input, grad_output):
            if isinstance(grad_output, tuple) and len(grad_output) > 0:
                self.gradients = grad_output[0].clone()
            else:
                self.gradients = grad_output

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int, float]:
        """
        Generate 2D Grad-CAM heatmap normalized to [0, 1].
        input_tensor: Shape (1, 3, H, W)
        """
        self.model.zero_grad()
        
        output = self.model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        
        if target_class is None:
            target_class = int(torch.argmax(output, dim=1).item())

        confidence = float(probabilities[0, target_class].item())

        score = output[0, target_class]
        score.backward(retain_graph=True)

        gradients = self.gradients.detach() # Shape: (1, C, H', W')
        activations = self.activations.detach() # Shape: (1, C, H', W')

        # Global average pooling of gradients over spatial dimensions
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True) # (1, C, 1, 1)

        # Weighted combination of forward activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True) # (1, 1, H', W')
        cam = torch.relu(cam) # Apply ReLU to keep only features that have positive influence

        # Upsample CAM to input dimensions
        cam = nn.functional.interpolate(
            cam,
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode="bilinear",
            align_corners=False
        )

        cam_np = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max - cam_min > 1e-8:
            cam_normalized = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_normalized = np.zeros_like(cam_np)

        return cam_normalized, target_class, confidence


def overlay_cam_on_image(
    image: Union[Image.Image, np.ndarray],
    cam_mask: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Superimpose Grad-CAM heatmap over the original RGB image.
    Returns: (heatmap_rgb_uint8, superimposed_rgb_uint8)
    """
    if isinstance(image, Image.Image):
        img_np = np.array(image.convert("RGB"))
    else:
        img_np = image.copy()

    # Resize CAM mask to image dimensions
    h, w = img_np.shape[:2]
    cam_resized = cv2.resize(cam_mask, (w, h))

    # Convert normalized CAM to uint8 heatmap
    heatmap_uint8 = np.uint8(255 * cam_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Superimpose
    superimposed = np.float32(heatmap_rgb) * alpha + np.float32(img_np) * (1.0 - alpha)
    superimposed = np.uint8(np.clip(superimposed, 0, 255))

    return heatmap_rgb, superimposed
