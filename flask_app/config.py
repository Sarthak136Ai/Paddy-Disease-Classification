"""
Configuration module for the Paddy Leaf Disease Flask Web Application.
"""

import os
from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

class Config:
    """Flask Application Configuration."""
    SECRET_KEY = os.environ.get("SECRET_KEY", "paddy-leaf-disease-detection-secret-key-2026")
    
    # Upload Settings
    UPLOAD_FOLDER = BASE_DIR / "uploads"
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}
    
    # Project Paths
    EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
    METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
    PLOTS_DIR = PROJECT_ROOT / "results" / "plots"
    SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"
    CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
    
    # Hardware Device Detection
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DEVICE_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Host CPU"
    
    # Static & Template folders
    STATIC_FOLDER = BASE_DIR / "static"
    TEMPLATE_FOLDER = BASE_DIR / "templates"
