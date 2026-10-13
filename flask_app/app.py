"""
Flask Application Factory for Paddy Leaf Disease Detection and Model Performance Analysis.
"""

import os
from pathlib import Path
from flask import Flask, send_from_directory, render_template

from flask_app.config import Config
from flask_app.routes import main_bp, prediction_bp, performance_bp
from flask_app.models.registry import MODEL_METADATA, CLASS_NAMES, CLASS_INFO
from src.utils.logger import setup_logger

logger = setup_logger("flask_app")


def create_app(config_class=Config) -> Flask:
    """Create and configure the Flask web application."""
    app = Flask(
        __name__,
        static_folder=str(Config.STATIC_FOLDER),
        template_folder=str(Config.TEMPLATE_FOLDER)
    )
    app.config.from_object(config_class)

    # Ensure upload directory exists
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    # Register Blueprints
    app.register_blueprint(main_bp)
    app.register_blueprint(prediction_bp)
    app.register_blueprint(performance_bp)

    # Static routes for experiment artifacts and sample images
    @app.route("/results/metrics/<path:filename>")
    def serve_metrics_file(filename: str):
        """Serve generated confusion matrices and metric plots."""
        return send_from_directory(str(Config.METRICS_DIR), filename)

    @app.route("/data/samples/<path:filename>")
    def serve_sample_image(filename: str):
        """Serve built-in test sample leaf images."""
        return send_from_directory(str(Config.SAMPLES_DIR), filename)

    @app.route("/results/plots/<path:filename>")
    def serve_plots_file(filename: str):
        """Serve general evaluation plots."""
        return send_from_directory(str(Config.PLOTS_DIR), filename)

    # Global Template Context
    @app.context_processor
    def inject_global_context():
        return {
            "app_title": "Paddy Leaf Disease Detection & Research Dashboard",
            "all_models": MODEL_METADATA,
            "all_classes": CLASS_INFO,
            "class_names": CLASS_NAMES,
            "device_type": Config.DEVICE.upper(),
            "device_name": Config.DEVICE_NAME
        }

    # Error Handlers
    @app.errorhandler(404)
    def page_not_found(e):
        return render_template("404.html", error=str(e)), 404

    @app.errorhandler(500)
    def internal_server_error(e):
        logger.error(f"Internal server error: {e}")
        return render_template("500.html", error=str(e)), 500

    @app.errorhandler(413)
    def file_too_large(e):
        return render_template("413.html", error="File exceeds maximum upload size (16 MB)."), 413

    logger.info(f"Flask Application initialized on device: {Config.DEVICE} ({Config.DEVICE_NAME})")
    return app


if __name__ == "__main__":
    flask_app = create_app()
    flask_app.run(host="127.0.0.1", port=5000, debug=False)
