"""
Inference and Prediction Pipeline for Paddy Leaf Disease Detection.
Provides top-k predictions, confidence estimates, and optional Grad-CAM visual explanations.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import yaml

from src.evaluation.gradcam import GradCAM, overlay_cam_on_image
from src.models import get_model
from src.preprocessing.transforms import get_val_transforms
from src.utils.checkpoint import load_checkpoint
from src.utils.logger import setup_logger

logger = setup_logger("predictor")


class PaddyDiseasePredictor:
    """
    End-to-End Predictor for Paddy Leaf Disease Classification.
    """
    def __init__(
        self,
        model_name: str,
        checkpoint_path: str,
        config_path: str = "configs/config.yaml",
        device: Optional[str] = None
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.class_names = self.config["project"]["class_names"]
        self.num_classes = len(self.class_names)
        self.image_size = tuple(self.config["dataset"].get("image_size", [224, 224]))

        # Instantiate Model
        self.model = get_model(model_name, num_classes=self.num_classes, pretrained=False)
        load_checkpoint(checkpoint_path, self.model, device=str(self.device))
        self.model.to(self.device)
        self.model.eval()

        self.transform = get_val_transforms(
            image_size=self.image_size,
            mean=self.config["preprocessing"].get("mean", [0.485, 0.456, 0.406]),
            std=self.config["preprocessing"].get("std", [0.229, 0.224, 0.225])
        )

        self.gradcam = GradCAM(self.model)
        logger.info(f"PaddyDiseasePredictor initialized with model '{model_name}' on {self.device}.")

    def predict(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        top_k: int = 3,
        generate_gradcam: bool = False
    ) -> Dict[str, Any]:
        """
        Predict disease class for a given input image.
        """
        if isinstance(image_input, (str, Path)):
            pil_image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            pil_image = Image.fromarray(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_image = image_input.convert("RGB")
        else:
            raise TypeError("Unsupported image input type.")

        # Transform and batch
        tensor_img = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor_img)
            probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        top_indices = np.argsort(probabilities)[::-1][:top_k]
        predicted_idx = int(top_indices[0])
        predicted_class = self.class_names[predicted_idx]
        confidence = float(probabilities[predicted_idx])

        top_k_list = [
            {
                "class_id": int(idx),
                "class_name": self.class_names[idx],
                "display_name": self.class_names[idx].replace("_", " ").title(),
                "confidence": float(round(probabilities[idx], 4)),
                "percentage": f"{probabilities[idx] * 100:.2f}%"
            }
            for idx in top_indices
        ]

        all_probs_dict = {
            self.class_names[i]: float(round(probabilities[i], 4))
            for i in range(self.num_classes)
        }

        result: Dict[str, Any] = {
            "predicted_class": predicted_class,
            "predicted_display_name": predicted_class.replace("_", " ").title(),
            "confidence": round(confidence, 4),
            "confidence_percentage": f"{confidence * 100:.2f}%",
            "top_k": top_k_list,
            "all_probabilities": all_probs_dict
        }

        if generate_gradcam:
            cam_mask, _, _ = self.gradcam.generate_cam(tensor_img, target_class=predicted_idx)
            heatmap_rgb, superimposed_rgb = overlay_cam_on_image(pil_image, cam_mask)
            result["gradcam_mask"] = cam_mask
            result["gradcam_heatmap"] = heatmap_rgb
            result["gradcam_superimposed"] = superimposed_rgb

        return result
