"""
Web Predictor Module for Paddy Leaf Disease Flask Web Application.
Handles image preprocessing, model execution, Grad-CAM generation, and result formatting.
"""

import base64
import io
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
from PIL import Image
import torch

from flask_app.models.model_loader import ModelCache
from flask_app.models.registry import CLASS_INFO, MODEL_METADATA, CLASS_NAMES
from src.utils.logger import setup_logger

logger = setup_logger("flask_predictor")


def image_to_base64_jpeg(image: Union[Image.Image, np.ndarray], quality: int = 90) -> str:
    """Convert PIL Image or uint8 NumPy array to base64 JPEG data URL."""
    if isinstance(image, np.ndarray):
        pil_img = Image.fromarray(image.astype(np.uint8))
    else:
        pil_img = image

    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def predict_paddy_leaf(
    model_key: str,
    image_input: Union[str, Path, Image.Image, bytes, io.BytesIO],
    top_k: int = 3,
    generate_gradcam: bool = True
) -> Dict[str, Any]:
    """
    Run complete inference pipeline on a paddy leaf image.
    
    Returns:
    - predicted_class: str
    - display_name: str
    - confidence: float
    - confidence_pct: str
    - top_predictions: List[Dict]
    - inference_time_ms: float
    - disease_info: Dict (scientific name, symptoms, management)
    - original_image_b64: str
    - gradcam_heatmap_b64: Optional[str]
    - gradcam_superimposed_b64: Optional[str]
    - model_name: str
    - device: str
    """
    # 1. Load image to PIL
    if isinstance(image_input, bytes):
        pil_image = Image.open(io.BytesIO(image_input)).convert("RGB")
    elif isinstance(image_input, (str, Path)):
        pil_image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, io.BytesIO):
        pil_image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        pil_image = image_input.convert("RGB")
    else:
        raise ValueError("Unsupported image input format.")

    # 2. Get cached model predictor
    predictor = ModelCache.get_predictor(model_key)
    meta = MODEL_METADATA[model_key]

    # 3. Time the inference pass
    device = predictor.device
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    raw_result = predictor.predict(
        pil_image,
        top_k=top_k,
        generate_gradcam=generate_gradcam
    )

    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    live_latency_ms = round((t1 - t0) * 1000.0, 2)

    # 4. Extract class metadata
    pred_class = raw_result["predicted_class"]
    disease_meta = CLASS_INFO.get(pred_class, {
        "name": pred_class.replace("_", " ").title(),
        "scientific_name": "N/A",
        "pathogen_type": "Unknown",
        "symptoms": "No specific symptoms recorded.",
        "management": "Consult local agricultural extension service."
    })

    # 5. Format Top-K with colors and percentages
    top_k_formatted = []
    for item in raw_result["top_k"]:
        item_class = item["class_name"]
        item_info = CLASS_INFO.get(item_class, {})
        top_k_formatted.append({
            "class_name": item_class,
            "display_name": item_info.get("name", item["display_name"]),
            "pathogen_type": item_info.get("pathogen_type", "Unknown"),
            "confidence": item["confidence"],
            "confidence_pct": f"{item['confidence'] * 100:.2f}%",
            "percentage_num": round(item["confidence"] * 100, 2)
        })

    # 6. Base64 representations
    original_b64 = image_to_base64_jpeg(pil_image)
    gradcam_heat_b64 = None
    gradcam_overlay_b64 = None

    if generate_gradcam and "gradcam_heatmap" in raw_result:
        gradcam_heat_b64 = image_to_base64_jpeg(raw_result["gradcam_heatmap"])
        gradcam_overlay_b64 = image_to_base64_jpeg(raw_result["gradcam_superimposed"])

    return {
        "model_key": model_key,
        "model_name": meta["name"],
        "device": str(device).upper(),
        "inference_time_ms": live_latency_ms,
        "predicted_class": pred_class,
        "display_name": disease_meta.get("name", pred_class.replace("_", " ").title()),
        "confidence": raw_result["confidence"],
        "confidence_pct": f"{raw_result['confidence'] * 100:.2f}%",
        "top_predictions": top_k_formatted,
        "all_probabilities": raw_result["all_probabilities"],
        "disease_info": disease_meta,
        "original_image_b64": original_b64,
        "gradcam_heatmap_b64": gradcam_heat_b64,
        "gradcam_superimposed_b64": gradcam_overlay_b64,
        "is_healthy": pred_class == "normal"
    }
