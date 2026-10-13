"""
Routes package initialization for Paddy Leaf Disease Flask Web Application.
"""

from flask_app.routes.main import main_bp
from flask_app.routes.prediction import prediction_bp
from flask_app.routes.performance import performance_bp

__all__ = ["main_bp", "prediction_bp", "performance_bp"]
