"""
Prediction Blueprint: Disease Detection Lab, Image Upload, Sample Evaluation, and Grad-CAM Visualizations.
"""

import io
from pathlib import Path
from typing import Any, Dict, List
from flask import Blueprint, render_template, request, jsonify, current_app
from PIL import Image
from werkzeug.utils import secure_filename

from flask_app.config import Config
from flask_app.models.predictor import predict_paddy_leaf, image_to_base64_jpeg
from flask_app.models.registry import MODEL_METADATA, CLASS_INFO, CLASS_NAMES
from src.utils.logger import setup_logger

logger = setup_logger("prediction_routes")

prediction_bp = Blueprint("prediction", __name__)


def allowed_file(filename: str) -> bool:
    """Check if uploaded file has an allowed image extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in Config.ALLOWED_EXTENSIONS


def get_available_samples() -> List[Dict[str, Any]]:
    """Scan and load metadata and thumbnail base64 for built-in sample leaves."""
    samples = []
    samples_dir = Config.SAMPLES_DIR
    if samples_dir.exists():
        sample_files = sorted(samples_dir.glob("sample_*.jpg"))
        for f in sample_files:
            # Filename format: sample_{id}_{class_name}.jpg
            parts = f.stem.split("_", 2)
            if len(parts) >= 3:
                sample_id = parts[1]
                class_name = parts[2]
            else:
                sample_id = "0"
                class_name = "unknown"

            class_meta = CLASS_INFO.get(class_name, {})
            try:
                with Image.open(f) as img:
                    # Create lightweight thumbnail
                    thumb = img.copy()
                    thumb.thumbnail((160, 160))
                    b64_thumb = image_to_base64_jpeg(thumb, quality=80)
                
                samples.append({
                    "sample_id": sample_id,
                    "filename": f.name,
                    "class_name": class_name,
                    "display_name": class_meta.get("name", class_name.replace("_", " ").title()),
                    "pathogen_type": class_meta.get("pathogen_type", "Unknown"),
                    "thumbnail_b64": b64_thumb
                })
            except Exception as e:
                logger.warning(f"Could not load sample thumbnail for {f}: {e}")
    return samples


@prediction_bp.route("/prediction")
@prediction_bp.route("/predict")
def prediction_page():
    """Render the main Disease Detection Studio."""
    samples = get_available_samples()
    default_model = "resnet50"

    return render_template(
        "prediction.html",
        models=MODEL_METADATA,
        default_model=default_model,
        classes=CLASS_INFO,
        samples=samples,
        device=Config.DEVICE,
        device_name=Config.DEVICE_NAME
    )


@prediction_bp.route("/api/predict", methods=["POST"])
def api_predict():
    """AJAX API endpoint for paddy leaf disease prediction."""
    try:
        # 1. Model Selection
        model_key = request.form.get("model", "resnet50").strip().lower()
        if model_key not in MODEL_METADATA:
            return jsonify({
                "success": False,
                "error": f"Invalid model key '{model_key}'. Available models: {list(MODEL_METADATA.keys())}"
            }), 400

        # 2. Grad-CAM Flag
        generate_gradcam = request.form.get("gradcam", "true").lower() in ("true", "1", "yes")

        # 3. Handle File Upload
        if "file" not in request.files or request.files["file"].filename == "":
            return jsonify({"success": False, "error": "No image file provided in upload request."}), 400

        file = request.files["file"]
        if not allowed_file(file.filename):
            return jsonify({
                "success": False,
                "error": f"Unsupported file type. Allowed formats: {', '.join(Config.ALLOWED_EXTENSIONS)}"
            }), 400

        # Read file bytes into memory
        image_bytes = file.read()
        try:
            # Validate image can be opened with PIL
            with Image.open(io.BytesIO(image_bytes)) as test_img:
                test_img.verify()
        except Exception as e:
            return jsonify({"success": False, "error": f"Uploaded file is not a valid image: {e}"}), 400

        # Run inference
        result = predict_paddy_leaf(
            model_key=model_key,
            image_input=image_bytes,
            top_k=3,
            generate_gradcam=generate_gradcam
        )

        return jsonify({
            "success": True,
            "data": result
        })

    except FileNotFoundError as e:
        logger.error(f"Checkpoint not found: {e}")
        return jsonify({"success": False, "error": str(e)}), 404
    except Exception as e:
        logger.exception(f"Inference error: {e}")
        return jsonify({"success": False, "error": f"Server inference error: {str(e)}"}), 500


@prediction_bp.route("/api/predict-sample", methods=["POST"])
def api_predict_sample():
    """AJAX API endpoint for testing built-in sample leaf images."""
    try:
        data = request.get_json() or {}
        filename = data.get("filename", "")
        model_key = data.get("model", "resnet50").strip().lower()
        generate_gradcam = data.get("gradcam", True)

        if not filename:
            return jsonify({"success": False, "error": "No sample filename specified."}), 400

        safe_name = secure_filename(filename)
        sample_path = Config.SAMPLES_DIR / safe_name
        if not sample_path.exists():
            return jsonify({"success": False, "error": f"Sample image '{safe_name}' not found."}), 404

        if model_key not in MODEL_METADATA:
            return jsonify({"success": False, "error": f"Invalid model '{model_key}'."}), 400

        result = predict_paddy_leaf(
            model_key=model_key,
            image_input=sample_path,
            top_k=3,
            generate_gradcam=generate_gradcam
        )

        return jsonify({
            "success": True,
            "data": result
        })

    except Exception as e:
        logger.exception(f"Sample prediction error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@prediction_bp.route("/api/samples", methods=["GET"])
def api_samples():
    """Return JSON list of all sample leaf images."""
    return jsonify({
        "success": True,
        "samples": get_available_samples()
    })
