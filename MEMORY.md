# Project Memory

## Project
Paddy Leaf Disease Detection Using Deep Learning and Transfer Learning & Flask Web Application

## Dataset
Hugging Face: `anthony2261/paddy-disease-classification`
- Total Images: 10,407 (0 corrupted, 72 duplicate groups)
- Splits: Stratified 70% Train (7,284), 15% Validation (1,561), 15% Test (1,562), Seed = 42
- Classes (10): `bacterial_leaf_blight`, `bacterial_leaf_streak`, `bacterial_panicle_blight`, `blast`, `brown_spot`, `dead_heart`, `downy_mildew`, `hispa`, `normal`, `tungro`
- Varieties: 10 cultivars (`ADT45` [67.19%], `KarnatakaPonni` [9.49%], etc.)
- Plant Age: 45 to 82 days (mean: 64.04 days)
- Image Resolution: Standardized to 224x224 RGB tensors with ImageNet normalization

## Environment & Hardware
- Python: 3.10.0 (64-bit)
- PyTorch: 2.12.0+cu130, Torchvision: 0.27.0+cu130
- Flask: 3.1.3, Werkzeug: 3.1.3
- GPU Acceleration: NVIDIA GeForce RTX 2050 (4GB VRAM)
- Key Packages: `datasets` 5.0.1, `scikit-learn` 1.7.2, `pandas` 2.3.3, `matplotlib` 3.7.5, `seaborn` 0.13.2, `opencv-python` 4.11.0

## Current Stage
Flask Web Application & Core ML Research Pipeline Complete (All 10 Phases Verified)

## Verified Benchmark Results (Untouched Test Set, N=1,562)

| Model | Test Accuracy (%) | Macro F1 | Weighted F1 | Parameters | Recorded GPU Latency | Size (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet50** | **96.93%** | **0.9696** | **0.9692** | 23,528,522 | 9.44 ms/img | 90.1 MB |
| **EfficientNet-B0** | **96.73%** | **0.9654** | **0.9674** | **4,020,358** | **7.08 ms/img** | **15.6 MB** |
| **MobileNetV3** | 94.37% | 0.9371 | 0.9440 | 4,214,842 | 11.63 ms/img | 16.3 MB |
| **DenseNet121** | 92.57% | 0.9199 | 0.9257 | 6,964,106 | 9.85 ms/img | 27.2 MB |
| **Custom CNN (Baseline)** | 45.39% | 0.3163 | 0.4150 | 423,338 | 10.82 ms/img | 1.7 MB |

## Architectural Decisions & Model Selection
- **Best Overall Classifier**: `ResNet50` achieves top Macro F1 (0.9696) and highest per-class recall.
- **Best Production/Edge Model**: `EfficientNet-B0` is selected as recommended deployment model due to 5.8x parameter reduction (4.02M params vs 23.53M) and lowest inference latency (7.08 ms) while retaining 96.73% test accuracy.
- **Explainability**: Grad-CAM attention maps successfully localize fungal blast spindle lesions and bacterial streak chlorosis across all 5 architectures.

## Flask Application Architecture
- **Factory & Config**: `flask_app/app.py`, `flask_app/config.py`, `app.py`
- **Models & Registry**: `flask_app/models/registry.py`, `model_loader.py` (Singleton ModelCache + Live Benchmarker), `predictor.py`
- **Blueprints**: `main_bp` (`/`, `/about`), `prediction_bp` (`/predict`, `/api/predict`, `/api/predict-sample`), `performance_bp` (`/model-performance`, `/class-performance`, `/confusion-matrix`, `/model-selection`, `/api/benchmark`)
- **Templates & Static**: Bespoke agricultural research design system, Chart.js visualizations, drag & drop uploader, Grad-CAM tabbed viewer.
- **Tests**: `flask_app/test_app.py` (12 automated unit/integration tests passing).

## Launch Commands
- Launch Flask Web App: `python app.py` (serves on `http://127.0.0.1:5000`)
- Run Flask Test Suite: `python -m unittest flask_app/test_app.py`
- Single-image CLI prediction: `python scripts/predict.py --image path/to/leaf.jpg --model efficientnet_b0 --gradcam`

## Last Updated
2026-10-02 11:06:00
