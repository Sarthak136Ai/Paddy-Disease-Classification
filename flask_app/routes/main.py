"""
Main Blueprint: Dashboard Homepage and About Pages.
"""

from flask import Blueprint, render_template, current_app
from flask_app.config import Config
from flask_app.models.registry import MODEL_METADATA, CLASS_INFO, CLASS_NAMES, load_recorded_benchmark_metrics

main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def index():
    """Render Dashboard Homepage."""
    benchmarks = load_recorded_benchmark_metrics()
    
    # Identify top performers based strictly on verified test metrics
    best_acc_model = max(benchmarks.values(), key=lambda x: x.get("accuracy") or 0.0)
    best_latency_model = min(
        [b for b in benchmarks.values() if isinstance(b.get("recorded_latency_ms"), (int, float))],
        key=lambda x: x["recorded_latency_ms"],
        default=None
    )

    dataset_stats = {
        "total_images": 10407,
        "classes_count": len(CLASS_NAMES),
        "models_count": len(MODEL_METADATA),
        "split_ratio": "70% Train / 15% Val / 15% Test",
        "test_images": 1562,
        "peak_accuracy": f"{best_acc_model.get('accuracy_pct', 'N/A')}%",
        "peak_accuracy_model": best_acc_model.get("model_name", "N/A"),
        "fastest_latency": f"{best_latency_model.get('recorded_latency_ms', 'N/A')} ms" if best_latency_model else "N/A",
        "fastest_latency_model": best_latency_model.get("model_name", "N/A") if best_latency_model else "N/A"
    }

    return render_template(
        "index.html",
        models=MODEL_METADATA,
        benchmarks=benchmarks,
        classes=CLASS_INFO,
        stats=dataset_stats,
        device=Config.DEVICE,
        device_name=Config.DEVICE_NAME
    )


@main_bp.route("/about")
def about():
    """Render Academic Background and Project Documentation."""
    return render_template(
        "about.html",
        models=MODEL_METADATA,
        classes=CLASS_INFO,
        device=Config.DEVICE,
        device_name=Config.DEVICE_NAME
    )
