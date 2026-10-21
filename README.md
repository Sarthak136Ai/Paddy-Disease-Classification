# Paddy Leaf Disease Detection & Multi-Model Performance Analysis System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.12+-EE4C2C.svg)](https://pytorch.org/)
[![Flask 3.1+](https://img.shields.io/badge/Web_App-Flask_3.1-000000.svg)](https://flask.palletsprojects.com/)
[![CUDA Accelerated](https://img.shields.io/badge/CUDA-NVIDIA_RTX_2050-green.svg)](https://developer.nvidia.com/cuda-zone)
[![HuggingFace Dataset](https://img.shields.io/badge/Dataset-anthony2261/paddy--disease--classification-yellow.svg)](https://huggingface.co/datasets/anthony2261/paddy-disease-classification)

An end-to-end, reproducible, research-grade computer vision system for automated classification, explainable diagnosis, and multi-architecture benchmarking of paddy/rice leaf pathologies.

---

## 1. Executive Summary & Problem Statement

Rice (*Oryza sativa*) is the staple sustenance for over half the world's population. Foliar infections—encompassing fungal, bacterial, viral, and pest-induced pathologies—lead to 15–30% annual yield reductions globally.

This project implements a rigorous, leak-free deep learning and transfer learning comparative study across 10,407 field images. We evaluate five distinct architectures:
1. **ResNet50** (ImageNet transfer learning) — *Peak diagnostic test accuracy* (**96.93%**, Macro F1: **0.9696**)
2. **EfficientNet-B0** (Compound scaled transfer learning) — *Production deployment winner* (**96.73%** accuracy, **7.08 ms** GPU latency, **4.02M** parameters)
3. **MobileNetV3-Large** (Edge-optimized inverted residuals) — **94.37%** accuracy
4. **DenseNet121** (Dense feature concatenation) — **92.57%** accuracy
5. **Custom CNN Baseline** (4-stage CNN trained from scratch) — **45.39%** accuracy (*empirical proof of transfer learning necessity*)

---

## 2. Dataset Overview

* **Identifier**: [`anthony2261/paddy-disease-classification`](https://huggingface.co/datasets/anthony2261/paddy-disease-classification)
* **Volume**: 10,407 RGB Images
* **Classes**: 10 Pathological Categories (9 Diseases + 1 Healthy Control)
* **Splitting Protocol**: Stratified 70% Train (7,284) / 15% Validation (1,561) / 15% Test (1,562), fixed random seed `42`.

### The 10 Pathological Classes:
1. **Bacterial Leaf Blight** (*Xanthomonas oryzae pv. oryzae*)
2. **Bacterial Leaf Streak** (*Xanthomonas oryzae pv. oryzicola*)
3. **Bacterial Panicle Blight** (*Burkholderia glumae*)
4. **Rice Blast** (*Magnaporthe oryzae*)
5. **Brown Spot** (*Bipolaris oryzae*)
6. **Dead Heart** (Yellow Stem Borer damage — *Scirpophaga incertulas*)
7. **Downy Mildew** (*Sclerophthora macrospora*)
8. **Rice Hispa** (Scraping beetle damage — *Dicladispa armigera*)
9. **Healthy Control (Normal)** (*Oryza sativa* clean epidermis)
10. **Rice Tungro Disease** (Viral complex transmitted by *Nephotettix virescens*)

---

## 3. Verified Benchmark Results

All metrics are verified from held-out test split evaluation artifacts ($N=1,562$ unseen images):

| Model Architecture | Type | Test Accuracy | Macro F1 | Macro Precision | Macro Recall | GPU Latency | Parameters | Size |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet50** | Residual Transfer Learning | **96.93%** | **0.9696** | **0.9694** | **0.9700** | 9.44 ms | 23.53M | 90.1 MB |
| **EfficientNet-B0** | Compound Scaling (Recommended) | **96.73%** | **0.9654** | 0.9642 | 0.9668 | **7.08 ms** | **4.02M** | **15.6 MB** |
| **MobileNetV3** | Edge-Optimized Inverted Residuals | **94.37%** | **0.9371** | 0.9333 | 0.9420 | 11.63 ms | 4.21M | 16.3 MB |
| **DenseNet121** | Densely Connected Convolutions | **92.57%** | **0.9199** | 0.9207 | 0.9203 | 9.85 ms | 6.96M | 27.2 MB |
| **Custom CNN** | Scratch Baseline (No Pretraining) | **45.39%** | **0.3163** | 0.3395 | 0.3332 | 10.82 ms | 0.42M | 1.7 MB |

> [!NOTE]
> **Offline Evaluation vs. Live Inference**:
> * **Recorded Evaluation**: Test-split metrics (Accuracy, Macro F1, Confusion Matrices) computed strictly on the static 1,562 test split images.
> * **Live Hardware Benchmark**: Real-time multi-pass timing runs executed on the host GPU/CPU with 15 warm-up iterations.

---

## 4. Flask Web Application Architecture

The Flask web application provides a responsive, research-oriented user interface for live leaf diagnosis and multi-model performance analysis.

### Application Pages & Features:
* **Dashboard (`/`)**: High-level research metrics (10,407 images, 10 classes, 5 models, peak accuracy), model highlight cards, and disease catalog.
* **Disease Detection Lab (`/predict`)**: 
  * Drag-and-drop or file browser upload for any rice leaf image.
  * Instant Test Sample Ribbon (10 one-click test leaves).
  * Real-time model selection dropdown.
  * Top-3 softmax confidence distributions with animated bars.
  * Agronomic symptoms and field management advice.
  * Interactive **Grad-CAM Attention Heatmap** (Superimposed, Heatmap alone, and Original image tabs).
* **Model Performance (`/model-performance`)**:
  * Complete verified benchmark table.
  * Chart 1: Grouped Bar Chart (Test Accuracy vs. Macro F1).
  * Chart 2: Horizontal Bar Chart (Inference Latency in ms/image).
  * Chart 3: Bubble / Scatter Plot (Accuracy vs. Parameter Count, bubble radius = latency).
  * Objective evidence-based model analysis.
* **Class Metrics (`/class-performance`)**: Model-switchable 10-class breakdown table with Precision, Recall, F1-Score, Support, and interactive F1 bar chart.
* **Confusion Matrix Gallery (`/confusion-matrix`)**: Interactive model switcher displaying high-resolution Normalized and Raw Confusion Matrices with misclassification analysis.
* **Model Selection & Live Benchmarking (`/model-selection`)**: Objective trade-off comparison matrix and an interactive real-time multi-pass hardware benchmark runner.
* **About & Documentation (`/about`)**: Dataset provenance, architecture specifications, Grad-CAM theory, and agricultural diagnostic disclaimer.

---

## 5. Installation & Setup

### Prerequisites
* Python 3.10 or higher
* CUDA 11.8 / 12.x compatible GPU (Optional, CPU is fully supported with automatic detection)

### 1. Clone Repository & Install Dependencies
```bash
git clone https://github.com/your-username/paddy-disease.git
cd paddy-disease

# Install required packages
pip install -r requirements.txt
```

### 2. Verify Pre-Trained Weights & Metric Artifacts
All 5 model checkpoints are stored in `experiments/<model_name>/best_model.pth` and evaluated metrics are in `results/metrics/`.

---

## 6. Running the Applications

### Launch the Flask Web Application (Recommended)
```bash
python app.py
```
Open your browser and navigate to:
```text
http://127.0.0.1:5000
```

### Run the Automated Test Suite
```bash
python -m unittest flask_app/test_app.py
```

### Run Single-Image CLI Predictor with Grad-CAM
```bash
python scripts/predict.py --image data/samples/sample_3_blast.jpg --model resnet50 --gradcam
```

### Launch Alternative Streamlit Application
```bash
streamlit run app/streamlit_app.py
```

---

## 7. Project Directory Structure

```text
paddy-disease/
├── app.py                             # Root entry point for Flask Web Application
├── README.md                          # Complete research & system documentation
├── MEMORY.md                          # Persistent project memory & state tracker
├── progress_tracker.md                # Phase execution progress checklist
├── requirements.txt                   # Environment dependencies (Flask, PyTorch, etc.)
│
├── flask_app/                         # Flask Web Application Package
│   ├── app.py                         # Application factory & route registration
│   ├── config.py                      # Application & hardware configuration
│   ├── test_app.py                    # Automated test suite (12 unit/integration tests)
│   ├── models/
│   │   ├── registry.py                # Model metadata, class info & metric loaders
│   │   ├── model_loader.py            # Singleton cache & live latency benchmarker
│   │   └── predictor.py               # Web inference engine & base64 image serializer
│   ├── routes/
│   │   ├── main.py                    # Routes: /, /about
│   │   ├── prediction.py              # Routes: /predict, /api/predict, /api/predict-sample
│   │   └── performance.py             # Routes: /model-performance, /class-performance,
│   │                                  #         /confusion-matrix, /model-selection, /api/benchmark
│   ├── templates/
│   │   ├── base.html                  # Master layout with navbar, status badge & footer
│   │   ├── index.html                 # Dashboard homepage
│   │   ├── prediction.html            # Disease Diagnosis Lab with Grad-CAM viewer
│   │   ├── performance.html           # Performance comparison charts & metrics table
│   │   ├── class_performance.html     # Per-class 10-disease classification report
│   │   ├── confusion_matrix.html      # Raw & normalized confusion matrices
│   │   ├── model_selection.html       # Selection matrix & live benchmarking tool
│   │   ├── about.html                 # Research specs & disclaimer
│   │   ├── 404.html, 500.html, 413.html # Custom error pages
│   └── static/
│       ├── css/style.css              # Custom agricultural research design system
│       └── js/
│           ├── main.js                # Mobile nav & toast alerts
│           ├── prediction.js          # Drag & drop upload, sample picker, AJAX predictor
│           └── charts.js              # Chart.js visualization configs & benchmark runner
│
├── configs/
│   └── config.yaml                    # Global pipeline configuration
├── data/
│   ├── samples/                       # 10 verified test sample leaf images (sample_0..9)
│   └── splits/
│       └── split_indices.json         # Persisted stratified 70/15/15 split indices
├── experiments/                       # Trained model checkpoints (best_model.pth)
│   ├── resnet50/
│   ├── efficientnet_b0/
│   ├── mobilenet_v3/
│   ├── densenet121/
│   └── custom_cnn/
├── results/
│   ├── metrics/                       # JSON metrics, confusion matrices & comparison tables
│   └── plots/                         # Exploratory data plots & Grad-CAM visual grids
├── src/                               # Core Python ML library
│   ├── data/                          # HF loader & PyTorch datasets
│   ├── models/                        # PyTorch model definitions
│   ├── preprocessing/                 # Augmentations & transforms
│   ├── training/                      # Trainer, loss functions, early stopping
│   ├── evaluation/                    # Metrics calculator & Grad-CAM generator
│   ├── inference/                     # Predictor pipeline
│   └── utils/                         # Checkpointing & logging
└── scripts/                           # Standalone CLI tools & verification scripts
```

---

## 8. Deploying to Vercel

The application is pre-configured for instantaneous deployment on **Vercel** serverless functions via `api/index.py` and `vercel.json`.

### Option A: 1-Click Git Integration (Recommended)
1. Push this repository to your GitHub account (`git push origin main`).
2. Navigate to [vercel.com/new](https://vercel.com/new).
3. Import your `Paddy-Disease-Classification` repository.
4. Keep the default settings (**Framework Preset: Other**, **Root Directory: `./`**).
5. Click **Deploy**. Vercel will install the serverless requirements and launch your live application at `https://<your-project>.vercel.app`!

### Option B: Deploy via Vercel CLI
```bash
# 1. Install or run Vercel CLI
npx vercel

# 2. Deploy to Production
npx vercel --prod
```

> **Note on Serverless Architecture**:
> - Vercel serverless functions enforce a strict **250 MB** bundle limit. The production `requirements.txt` includes lightweight dependencies (`Flask`, `Pillow`, `numpy`, `Werkzeug`).
> - For full local GPU/CUDA model training and PyTorch experiments, install: `pip install -r requirements-train.txt`.

---

## 9. Troubleshooting & FAQ

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| `CUDA out of memory` during live benchmark | High concurrent VRAM allocation | The application automatically falls back to CPU or reduces batch size ($B=1$). |
| `Unsupported file format` on upload | Uploaded non-image or corrupted file | Supported formats: `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`. |
| `Port 5000 already in use` | Another process listening on port 5000 | Kill existing process or modify port in `app.py` (`app.run(port=5001)`). |

---

## 10. Agricultural & Ethical Disclaimer

This application is developed strictly for **academic, research, and educational purposes**. Machine learning predictions and Grad-CAM attention heatmaps are statistical approximations and must not replace professional on-site phytosanitary scouting or chemical prescription by certified agronomists.
