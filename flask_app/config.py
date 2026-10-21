"""
Configuration module for the Paddy Leaf Disease Flask Web Application.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

try:
    import torch
    TORCH_AVAILABLE = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DEVICE_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Host CPU"
except ImportError:
    torch = None
    TORCH_AVAILABLE = False
    DEVICE = "serverless"
    DEVICE_NAME = "Vercel Serverless / Cloud Runtime"


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
    TORCH_AVAILABLE = TORCH_AVAILABLE
    DEVICE = DEVICE
    DEVICE_NAME = DEVICE_NAME
    
    # Static & Template folders
    STATIC_FOLDER = BASE_DIR / "static"
    TEMPLATE_FOLDER = BASE_DIR / "templates"
