"""
Performance Blueprint: Model Performance, Per-Class Metrics, Confusion Matrices, and Live Benchmarking Suite.
"""

from typing import Any, Dict
from flask import Blueprint, render_template, request, jsonify, current_app

from flask_app.config import Config
from flask_app.models.model_loader import benchmark_model
from flask_app.models.registry import (
    MODEL_METADATA,
    CLASS_INFO,
    CLASS_NAMES,
    load_recorded_benchmark_metrics
)
from src.utils.logger import setup_logger

logger = setup_logger("performance_routes")

performance_bp = Blueprint("performance", __name__)


@performance_bp.route("/model-performance")
def model_performance():
    """Render Model Performance comparison dashboard with Chart.js charts and metrics table."""
    benchmarks = load_recorded_benchmark_metrics()
    
    # Prepare data payloads for Chart.js
    model_keys = list(MODEL_METADATA.keys())
    model_labels = [MODEL_METADATA[k]["name"] for k in model_keys]
    accuracies = [benchmarks[k]["accuracy_pct"] if benchmarks[k]["accuracy"] is not None else 0 for k in model_keys]
    f1_scores = [round(benchmarks[k]["f1_macro"] * 100, 2) if benchmarks[k]["f1_macro"] != "N/A" else 0 for k in model_keys]
    latencies = [benchmarks[k]["recorded_latency_ms"] if benchmarks[k]["recorded_latency_ms"] != "N/A" else 0 for k in model_keys]
    params_millions = [round(MODEL_METADATA[k]["total_parameters"] / 1e6, 2) for k in model_keys]

    # Scatter plot data points: {x: params_M, y: accuracy_pct, r: latency_bubble_radius}
    scatter_data = []
    for k in model_keys:
        acc = benchmarks[k]["accuracy_pct"]
        lat = benchmarks[k]["recorded_latency_ms"]
        if acc != "N/A" and lat != "N/A":
            scatter_data.append({
                "model_name": MODEL_METADATA[k]["name"],
                "model_key": k,
                "x": round(MODEL_METADATA[k]["total_parameters"] / 1e6, 2),
                "y": acc,
                "latency": lat,
                "params": MODEL_METADATA[k]["total_parameters"],
                "r": max(6, min(24, int(lat * 1.5)))  # scale bubble radius
            })

    chart_payload = {
        "labels": model_labels,
        "accuracies": accuracies,
        "f1_scores": f1_scores,
        "latencies": latencies,
        "params_m": params_millions,
        "scatter_data": scatter_data
    }

    return render_template(
        "performance.html",
        models=MODEL_METADATA,
        benchmarks=benchmarks,
        charts=chart_payload,
        device=Config.DEVICE,
        device_name=Config.DEVICE_NAME
    )


@performance_bp.route("/class-performance")
def class_performance():
    """Render Per-Class Performance breakdown across all 10 disease categories."""
    benchmarks = load_recorded_benchmark_metrics()
    selected_model = request.args.get("model", "resnet50").strip().lower()
    if selected_model not in MODEL_METADATA:
        selected_model = "resnet50"

    current_benchmark = benchmarks.get(selected_model, {})
    per_class_metrics = current_benchmark.get("per_class", {})

    # Build structured class table rows
    class_rows = []
    for class_name in CLASS_NAMES:
        meta = CLASS_INFO.get(class_name, {})
        metrics = per_class_metrics.get(class_name, {})
        class_rows.append({
            "class_key": class_name,
            "display_name": meta.get("name", class_name.replace("_", " ").title()),
            "pathogen_type": meta.get("pathogen_type", "Unknown"),
            "precision": metrics.get("precision", "N/A"),
            "precision_pct": f"{metrics.get('precision', 0.0) * 100:.2f}%" if "precision" in metrics else "N/A",
            "recall": metrics.get("recall", "N/A"),
            "recall_pct": f"{metrics.get('recall', 0.0) * 100:.2f}%" if "recall" in metrics else "N/A",
            "f1_score": metrics.get("f1_score", "N/A"),
            "f1_pct": f"{metrics.get('f1_score', 0.0) * 100:.2f}%" if "f1_score" in metrics else "N/A",
            "support": metrics.get("support", "N/A")
        })

    return render_template(
        "class_performance.html",
        models=MODEL_METADATA,
        benchmarks=benchmarks,
        selected_model=selected_model,
        current_meta=MODEL_METADATA[selected_model],
        current_benchmark=current_benchmark,
        class_rows=class_rows,
        classes=CLASS_INFO,
        device=Config.DEVICE,
        device_name=Config.DEVICE_NAME
    )


@performance_bp.route("/confusion-matrix")
def confusion_matrix():
    """Render Confusion Matrix visualizer with model selector and error analysis."""
    benchmarks = load_recorded_benchmark_metrics()
    selected_model = request.args.get("model", "resnet50").strip().lower()
    if selected_model not in MODEL_METADATA:
        selected_model = "resnet50"

    current_meta = MODEL_METADATA[selected_model]
    current_benchmark = benchmarks.get(selected_model, {})

    return render_template(
        "confusion_matrix.html",
        models=MODEL_METADATA,
        benchmarks=benchmarks,
        selected_model=selected_model,
        current_meta=current_meta,
        current_benchmark=current_benchmark,
        device=Config.DEVICE,
        device_name=Config.DEVICE_NAME
    )


@performance_bp.route("/model-selection")
def model_selection():
    """Render Model Selection and Live Hardware Benchmarking Suite."""
    benchmarks = load_recorded_benchmark_metrics()
    selected_model = request.args.get("model", "resnet50").strip().lower()
    if selected_model not in MODEL_METADATA:
        selected_model = "resnet50"

    return render_template(
        "model_selection.html",
        models=MODEL_METADATA,
        benchmarks=benchmarks,
        selected_model=selected_model,
        current_meta=MODEL_METADATA[selected_model],
        current_benchmark=benchmarks.get(selected_model, {}),
        device=Config.DEVICE,
        device_name=Config.DEVICE_NAME
    )


@performance_bp.route("/api/benchmark", methods=["POST"])
def api_benchmark():
    """Run real-time latency benchmarking on the active server device."""
    try:
        data = request.get_json() or {}
        model_key = data.get("model", "resnet50").strip().lower()
        warm_up = int(data.get("warm_up", 15))
        iterations = int(data.get("iterations", 60))

        # Clamp iterations to reasonable bounds
        warm_up = max(5, min(warm_up, 30))
        iterations = max(10, min(iterations, 200))

        if model_key not in MODEL_METADATA:
            return jsonify({"success": False, "error": f"Invalid model '{model_key}'."}), 400

        result = benchmark_model(model_key, warm_up=warm_up, iterations=iterations)
        
        # Attach recorded benchmark for comparison
        benchmarks = load_recorded_benchmark_metrics()
        recorded = benchmarks.get(model_key, {})
        result["recorded_latency_ms"] = recorded.get("recorded_latency_ms", "N/A")
        result["accuracy_pct"] = recorded.get("accuracy_pct", "N/A")
        result["f1_macro"] = recorded.get("f1_macro", "N/A")

        return jsonify({"success": True, "data": result})

    except Exception as e:
        logger.exception(f"Benchmarking error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@performance_bp.route("/api/metrics", methods=["GET"])
def api_metrics():
    """Return all recorded benchmark metrics as JSON."""
    benchmarks = load_recorded_benchmark_metrics()
    return jsonify({"success": True, "data": benchmarks})


@performance_bp.route("/api/metrics/<model_key>", methods=["GET"])
def api_model_metrics(model_key: str):
    """Return recorded metrics for a single model."""
    benchmarks = load_recorded_benchmark_metrics()
    if model_key not in benchmarks:
        return jsonify({"success": False, "error": f"Model '{model_key}' not found."}), 404
    return jsonify({"success": True, "data": benchmarks[model_key]})
