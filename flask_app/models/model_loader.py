"""
Model Loader, Validation, Singleton Cache, and Live Benchmark Suite.
"""

import os
import time
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch

from flask_app.config import Config
from flask_app.models.registry import MODEL_METADATA, CLASS_NAMES
from src.inference.predict import PaddyDiseasePredictor
from src.utils.logger import setup_logger

logger = setup_logger("flask_model_loader")


class ModelCache:
    """
    Thread-safe Singleton Model Cache.
    Keeps trained PyTorch models loaded in memory on demand to avoid reloading overhead.
    """
    _instance = None
    _lock = threading.Lock()
    _predictors: Dict[str, PaddyDiseasePredictor] = {}

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModelCache, cls).__new__(cls)
                cls._predictors = {}
        return cls._instance

    @classmethod
    def get_predictor(cls, model_key: str) -> PaddyDiseasePredictor:
        """
        Retrieve or instantiate a PaddyDiseasePredictor for the given model key.
        Performs thorough validation on initial load.
        """
        if model_key not in MODEL_METADATA:
            raise ValueError(f"Unknown model key '{model_key}'. Available models: {list(MODEL_METADATA.keys())}")

        with cls._lock:
            if model_key in cls._predictors:
                return cls._predictors[model_key]

            meta = MODEL_METADATA[model_key]
            checkpoint_path = Path(meta["checkpoint_path"])

            # 1. Validate checkpoint file exists
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"Checkpoint for model '{meta['name']}' not found at {checkpoint_path}. "
                    "Ensure model weights have been trained and saved."
                )

            # 2. Instantiate and load model
            logger.info(f"Loading model '{meta['name']}' from checkpoint: {checkpoint_path}...")
            try:
                predictor = PaddyDiseasePredictor(
                    model_name=model_key,
                    checkpoint_path=str(checkpoint_path),
                    config_path=str(Config.CONFIG_PATH),
                    device=Config.DEVICE
                )
            except Exception as e:
                logger.error(f"Failed to instantiate predictor for '{model_key}': {e}")
                raise RuntimeError(f"Error loading model '{meta['name']}': {e}")

            # 3. Validate model integrity via dummy forward pass
            dummy_input = torch.randn(1, 3, 224, 224, device=predictor.device)
            try:
                with torch.no_grad():
                    dummy_output = predictor.model(dummy_input)
                    if dummy_output.shape != (1, len(CLASS_NAMES)):
                        raise ValueError(
                            f"Model output shape mismatch! Expected (1, {len(CLASS_NAMES)}), got {dummy_output.shape}"
                        )
                logger.info(f"Model '{meta['name']}' passed forward-pass validation successfully.")
            except Exception as e:
                logger.error(f"Model '{meta['name']}' failed forward-pass validation: {e}")
                raise RuntimeError(f"Validation failed for model '{meta['name']}': {e}")

            cls._predictors[model_key] = predictor
            return predictor

    @classmethod
    def preload_all(cls) -> Dict[str, bool]:
        """Preload all available registered models into memory."""
        status = {}
        for key in MODEL_METADATA:
            try:
                cls.get_predictor(key)
                status[key] = True
            except Exception as e:
                logger.warning(f"Could not preload model '{key}': {e}")
                status[key] = False
        return status


def benchmark_model(
    model_key: str,
    warm_up: int = 15,
    iterations: int = 60
) -> Dict[str, Any]:
    """
    Measure actual live inference latency on the server's current hardware (GPU/CPU).
    Applies warm-up passes to eliminate cold-start artifacts, then times multiple iterations.
    
    Returns detailed benchmark statistics:
    - mean_latency_ms
    - median_latency_ms
    - std_latency_ms
    - min_latency_ms
    - max_latency_ms
    - fps (throughput)
    - hardware details
    """
    predictor = ModelCache.get_predictor(model_key)
    model = predictor.model
    device = predictor.device
    model.eval()

    dummy_tensor = torch.randn(1, 3, 224, 224, device=device)
    is_cuda = device.type == "cuda"

    # 1. Warm-up Phase
    logger.info(f"Starting warm-up ({warm_up} passes) for model '{model_key}' on {device}...")
    with torch.no_grad():
        for _ in range(warm_up):
            _ = model(dummy_tensor)
            if is_cuda:
                torch.cuda.synchronize()

    # 2. Timed Iterations Phase
    logger.info(f"Running timed benchmark ({iterations} iterations) for model '{model_key}'...")
    latencies = []

    with torch.no_grad():
        for _ in range(iterations):
            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            _ = model(dummy_tensor)

            if is_cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            latencies.append((t1 - t0) * 1000.0)  # convert to milliseconds

    latencies_np = np.array(latencies)
    mean_lat = float(np.mean(latencies_np))
    median_lat = float(np.median(latencies_np))
    std_lat = float(np.std(latencies_np))
    min_lat = float(np.min(latencies_np))
    max_lat = float(np.max(latencies_np))
    fps = float(1000.0 / mean_lat) if mean_lat > 0 else 0.0

    meta = MODEL_METADATA[model_key]

    return {
        "model_key": model_key,
        "model_name": meta["name"],
        "mean_latency_ms": round(mean_lat, 2),
        "median_latency_ms": round(median_lat, 2),
        "std_latency_ms": round(std_lat, 3),
        "min_latency_ms": round(min_lat, 2),
        "max_latency_ms": round(max_lat, 2),
        "fps": round(fps, 1),
        "iterations": iterations,
        "warm_up_passes": warm_up,
        "device": str(device).upper(),
        "device_name": Config.DEVICE_NAME,
        "input_shape": [1, 3, 224, 224],
        "batch_size": 1,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
