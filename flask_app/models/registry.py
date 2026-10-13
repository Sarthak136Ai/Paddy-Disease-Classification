"""
Model Registry and Metadata Definition for Paddy Leaf Disease Flask Web Application.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# 10 Paddy Disease Classes and Agronomic Guidelines
CLASS_INFO = {
    "bacterial_leaf_blight": {
        "id": 0,
        "name": "Bacterial Leaf Blight",
        "scientific_name": "Xanthomonas oryzae pv. oryzae",
        "pathogen_type": "Bacterial",
        "symptoms": "Water-soaked lesions that enlarge, turn yellow to grayish-white along leaf margins, causing seedling wilt and blighted foliage.",
        "management": "Apply balanced fertilization (avoid excessive nitrogen), use certified resistant varieties (e.g. IR64), ensure field drainage."
    },
    "bacterial_leaf_streak": {
        "id": 1,
        "name": "Bacterial Leaf Streak",
        "scientific_name": "Xanthomonas oryzae pv. oryzicola",
        "pathogen_type": "Bacterial",
        "symptoms": "Narrow, brownish interveinal streaks running parallel to leaf veins, with tiny yellow bacterial exudate beads under humid conditions.",
        "management": "Refrain from high nitrogen inputs, control irrigation water flow across fields, apply copper-based bactericides if severe."
    },
    "bacterial_panicle_blight": {
        "id": 2,
        "name": "Bacterial Panicle Blight",
        "scientific_name": "Burkholderia glumae",
        "pathogen_type": "Bacterial",
        "symptoms": "Discolored florets, upright standing unfilled panicles, dark brown lesion rings at panicle base and florets.",
        "management": "Plant certified disease-free seeds, avoid late planting in high-temperature regions, apply recommended protective foliar sprays at boot stage."
    },
    "blast": {
        "id": 3,
        "name": "Rice Blast",
        "scientific_name": "Magnaporthe oryzae (Pyricularia oryzae)",
        "pathogen_type": "Fungal",
        "symptoms": "Spindle-shaped (diamond/eye-like) necrotic spots with grayish centers and dark brown margins on leaves, collar, and panicle neck.",
        "management": "Apply recommended systemic fungicides (e.g. Tricyclazole, Isoprothiolane), maintain proper water levels, avoid dense planting."
    },
    "brown_spot": {
        "id": 4,
        "name": "Brown Spot",
        "scientific_name": "Bipolaris oryzae (Cochliobolus miyabeanus)",
        "pathogen_type": "Fungal",
        "symptoms": "Small, circular to oval dark-brown spots with prominent yellow halos distributed across the leaf blade.",
        "management": "Correct nutrient deficiencies (apply Potassium, Silicon, Zinc), treat seeds with fungicides before sowing, improve soil health."
    },
    "dead_heart": {
        "id": 5,
        "name": "Dead Heart",
        "scientific_name": "Scirpophaga incertulas (Yellow Stem Borer)",
        "pathogen_type": "Pest-Induced",
        "symptoms": "Drying and death of the central vegetative tiller shoot caused by internal larvae feeding.",
        "management": "Deploy pheromone traps for monitoring, release egg parasitoids (Trichogramma japonicum), apply targeted systemic insecticides if ETL is breached."
    },
    "downy_mildew": {
        "id": 6,
        "name": "Downy Mildew",
        "scientific_name": "Sclerophthora macrospora",
        "pathogen_type": "Oomycete",
        "symptoms": "Stunted seedlings, yellow-green chlorotic leaf mottling, twisted leaves, and deformed, upright floral structures.",
        "management": "Improve nursery and field drainage, remove infected plants and alternate grass weed hosts, ensure clean nursery beds."
    },
    "hispa": {
        "id": 7,
        "name": "Rice Hispa",
        "scientific_name": "Dicladispa armigera",
        "pathogen_type": "Pest-Induced",
        "symptoms": "Parallel white feeding streaks along leaf epidermis scraped by adult beetles and grub leaf-mining damage.",
        "management": "Clip leaf tips containing eggs/grubs before transplanting, sweep-net adult beetles, spray recommended contact insecticides if needed."
    },
    "normal": {
        "id": 8,
        "name": "Healthy Paddy Leaf",
        "scientific_name": "Oryza sativa (Clean Epidermis)",
        "pathogen_type": "Healthy Control",
        "symptoms": "Vibrant, uniform green coloration, clear epidermal surface, devoid of necrotic lesions, spots, or pest injury.",
        "management": "Continue regular crop monitoring, balanced NPK application, and standard integrated pest and disease management."
    },
    "tungro": {
        "id": 9,
        "name": "Rice Tungro Disease",
        "scientific_name": "Rice Tungro Bacilliform & Spherical Virus Complex",
        "pathogen_type": "Viral Complex (Vector: Nephotettix virescens)",
        "symptoms": "Severe plant stunting, reduced tillering, yellow to orange-yellow leaf discoloration beginning at tips and extending downward.",
        "management": "Control green leafhopper vectors using recommended insecticides, practice synchronous planting, destroy infected stubble."
    }
}

CLASS_NAMES = list(CLASS_INFO.keys())

# Model Registry Metadata
MODEL_METADATA = {
    "resnet50": {
        "name": "ResNet50",
        "key": "resnet50",
        "type": "Transfer Learning (CNN)",
        "backbone": "ResNet-50 ImageNet Pretrained",
        "checkpoint_path": str(PROJECT_ROOT / "experiments" / "resnet50" / "best_model.pth"),
        "metrics_path": str(PROJECT_ROOT / "results" / "metrics" / "resnet50_test_metrics.json"),
        "confusion_matrix_raw": "results/metrics/resnet50_test_confusion_matrix.png",
        "confusion_matrix_norm": "results/metrics/resnet50_test_confusion_matrix_norm.png",
        "total_parameters": 23528522,
        "input_size": [224, 224],
        "description": "Deep 50-layer residual network utilizing skip connections for optimal feature representation. Achieved highest overall test accuracy and macro F1 score."
    },
    "efficientnet_b0": {
        "name": "EfficientNet-B0",
        "key": "efficientnet_b0",
        "type": "Transfer Learning (Compound Scaled)",
        "backbone": "EfficientNet-B0 ImageNet Pretrained",
        "checkpoint_path": str(PROJECT_ROOT / "experiments" / "efficientnet_b0" / "best_model.pth"),
        "metrics_path": str(PROJECT_ROOT / "results" / "metrics" / "efficientnet_b0_test_metrics.json"),
        "confusion_matrix_raw": "results/metrics/efficientnet_b0_test_confusion_matrix.png",
        "confusion_matrix_norm": "results/metrics/efficientnet_b0_test_confusion_matrix_norm.png",
        "total_parameters": 4020358,
        "input_size": [224, 224],
        "description": "Compound scaled architecture balancing network depth, width, and resolution. Recommended for production deployment due to 5.8x parameter efficiency and fastest inference latency."
    },
    "mobilenet_v3": {
        "name": "MobileNetV3",
        "key": "mobilenet_v3",
        "type": "Transfer Learning (Mobile/Edge)",
        "backbone": "MobileNetV3-Large ImageNet Pretrained",
        "checkpoint_path": str(PROJECT_ROOT / "experiments" / "mobilenet_v3" / "best_model.pth"),
        "metrics_path": str(PROJECT_ROOT / "results" / "metrics" / "mobilenet_v3_test_metrics.json"),
        "confusion_matrix_raw": "results/metrics/mobilenet_v3_test_confusion_matrix.png",
        "confusion_matrix_norm": "results/metrics/mobilenet_v3_test_confusion_matrix_norm.png",
        "total_parameters": 4214842,
        "input_size": [224, 224],
        "description": "Lightweight vision backbone featuring inverted residuals with Hard-Swish non-linearities and Squeeze-and-Excitation attention modules."
    },
    "densenet121": {
        "name": "DenseNet121",
        "key": "densenet121",
        "type": "Transfer Learning (Densely Connected)",
        "backbone": "DenseNet-121 ImageNet Pretrained",
        "checkpoint_path": str(PROJECT_ROOT / "experiments" / "densenet121" / "best_model.pth"),
        "metrics_path": str(PROJECT_ROOT / "results" / "metrics" / "densenet121_test_metrics.json"),
        "confusion_matrix_raw": "results/metrics/densenet121_test_confusion_matrix.png",
        "confusion_matrix_norm": "results/metrics/densenet121_test_confusion_matrix_norm.png",
        "total_parameters": 6964106,
        "input_size": [224, 224],
        "description": "Densely connected convolutional network concatenating all preceding feature maps to maximize feature reuse."
    },
    "custom_cnn": {
        "name": "Custom CNN (Baseline)",
        "key": "custom_cnn",
        "type": "Scratch Baseline",
        "backbone": "4-Stage ConvBlock with Global Pooling",
        "checkpoint_path": str(PROJECT_ROOT / "experiments" / "custom_cnn" / "best_model.pth"),
        "metrics_path": str(PROJECT_ROOT / "results" / "metrics" / "custom_cnn_test_metrics.json"),
        "confusion_matrix_raw": "results/metrics/custom_cnn_test_confusion_matrix.png",
        "confusion_matrix_norm": "results/metrics/custom_cnn_test_confusion_matrix_norm.png",
        "total_parameters": 423338,
        "input_size": [224, 224],
        "description": "Trained from scratch baseline without pretrained weights. Demonstrates the critical performance uplift provided by ImageNet transfer learning representations."
    }
}


def load_recorded_benchmark_metrics() -> Dict[str, Dict[str, Any]]:
    """
    Dynamically load verified test metrics for all models from JSON artifacts.
    """
    benchmarks = {}
    for model_key, meta in MODEL_METADATA.items():
        metrics_file = Path(meta["metrics_path"])
        if metrics_file.exists():
            with open(metrics_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                benchmarks[model_key] = {
                    "model_key": model_key,
                    "model_name": meta["name"],
                    "accuracy": data.get("accuracy", 0.0),
                    "accuracy_pct": round(data.get("accuracy", 0.0) * 100, 2),
                    "precision_macro": data.get("precision_macro", 0.0),
                    "recall_macro": data.get("recall_macro", 0.0),
                    "f1_macro": data.get("f1_macro", 0.0),
                    "f1_weighted": data.get("f1_weighted", 0.0),
                    "test_loss": data.get("test_loss", 0.0),
                    "recorded_latency_ms": data.get("latency_ms_per_image", 0.0),
                    "total_parameters": data.get("total_parameters", meta["total_parameters"]),
                    "model_size_mb": data.get("checkpoint_size_mb", 0.0),
                    "per_class": data.get("per_class", {}),
                    "classification_report_dict": data.get("classification_report_dict", {})
                }
        else:
            benchmarks[model_key] = {
                "model_key": model_key,
                "model_name": meta["name"],
                "accuracy": None,
                "accuracy_pct": "N/A",
                "f1_macro": "N/A",
                "recorded_latency_ms": "N/A",
                "total_parameters": meta["total_parameters"],
                "status": "Not Evaluated"
            }
    return benchmarks
