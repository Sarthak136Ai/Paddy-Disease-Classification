"""
Interactive Streamlit Web Application for Paddy Leaf Disease Detection.
Provides image upload, sample gallery, model switching, Top-K confidence breakdown,
and Grad-CAM visual lesion explainability.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from PIL import Image
import numpy as np
import pandas as pd
import streamlit as st

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predict import PaddyDiseasePredictor

# Page Configuration
st.set_page_config(
    page_title="Paddy Leaf Disease Classifier",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, premium aesthetics
st.markdown("""
<style>
    .main-header {
        font-family: 'Segoe UI', Roboto, sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3d2f;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4a6b57;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8faf9;
        border: 1px solid #dbe7e0;
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.03);
    }
    .pred-badge {
        background: linear-gradient(135deg, #2e7d32, #1b5e20);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 600;
        display: inline-block;
        font-size: 1.2rem;
        margin-bottom: 0.8rem;
    }
    .disclaimer-box {
        background: #fff8e1;
        border-left: 5px solid #ffa000;
        padding: 0.8rem 1.2rem;
        border-radius: 4px;
        font-size: 0.88rem;
        color: #5d4037;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Disease Information & Management Recommendations
DISEASE_INFO = {
    "bacterial_leaf_blight": {
        "name": "Bacterial Leaf Blight (Xanthomonas oryzae)",
        "symptoms": "Water-soaked lesions turning yellow/white along leaf margins, wilting seedlings.",
        "management": "Apply balanced fertilizers (avoid excess nitrogen), use resistant varieties, ensure proper field drainage."
    },
    "bacterial_leaf_streak": {
        "name": "Bacterial Leaf Streak (Xanthomonas oryzae pv. oryzicola)",
        "symptoms": "Narrow, brownish interveinal streaks with tiny amber bacterial exudates.",
        "management": "Avoid high nitrogen levels, control irrigation water flow between plots, apply copper-based bactericides if advised."
    },
    "bacterial_panicle_blight": {
        "name": "Bacterial Panicle Blight (Burkholderia glumae)",
        "symptoms": "Discolored florets, unfilled grain panicles standing upright, brownish panicle base.",
        "management": "Plant certified disease-free seeds, avoid late planting in hot climates, apply recommended foliar treatments at heading."
    },
    "blast": {
        "name": "Rice Blast (Magnaporthe oryzae)",
        "symptoms": "Spindle-shaped diamond lesions with grayish centers and brown borders on leaves and neck.",
        "management": "Use fungicides (e.g. Tricyclazole), avoid excessive nitrogen, maintain consistent water level."
    },
    "brown_spot": {
        "name": "Brown Spot (Bipolaris oryzae)",
        "symptoms": "Small, oval dark-brown spots with yellow halos scattered over leaf blades.",
        "management": "Improve soil fertility (potash, zinc), practice seed treatment with fungicides, ensure balanced micronutrients."
    },
    "dead_heart": {
        "name": "Dead Heart (Stem Borer Damage)",
        "symptoms": "Drying and death of the central tiller shoot during vegetative stage.",
        "management": "Apply biological controls (Trichogramma), use pheromone traps, spray recommended systemic insecticides if threshold is crossed."
    },
    "downy_mildew": {
        "name": "Downy Mildew (Sclerophthora macrospora)",
        "symptoms": "Stunted seedlings, yellow-green mottled leaves, twisted and deformed panicles.",
        "management": "Improve field drainage, eradicate wild grass hosts, ensure clean nursery beds."
    },
    "hispa": {
        "name": "Rice Hispa (Dicladispa armigera)",
        "symptoms": "Scraped white streaks along leaf blade epidermis caused by beetle feeding.",
        "management": "Clip leaf tips containing grubs before transplanting, sweep-net adult beetles, apply recommended contact insecticides."
    },
    "normal": {
        "name": "Healthy Paddy Leaf (No Disease Detected)",
        "symptoms": "Uniform vibrant green coloration, clean epidermis, no necrotic lesions or fungal spots.",
        "management": "Continue standard agronomic practices, periodic crop scouting, and balanced fertilization."
    },
    "tungro": {
        "name": "Rice Tungro Disease (RTV Complex)",
        "symptoms": "Stunted plant height, reduced tillering, yellow to orange-yellow leaf discoloration starting from tips.",
        "management": "Control green leafhopper (vector) populations, practice synchronous planting, destroy diseased stubble."
    }
}


@st.cache_resource
def load_predictor(model_name: str) -> Optional[PaddyDiseasePredictor]:
    """Cache and load predictor instance."""
    checkpoint_path = PROJECT_ROOT / "experiments" / model_name / "best_model.pth"
    if not checkpoint_path.exists():
        return None
    try:
        predictor = PaddyDiseasePredictor(
            model_name=model_name,
            checkpoint_path=str(checkpoint_path),
            config_path=str(PROJECT_ROOT / "configs" / "config.yaml")
        )
        return predictor
    except Exception as e:
        st.error(f"Error loading model {model_name}: {e}")
        return None


def main():
    # Sidebar Configuration
    st.sidebar.image("https://images.unsplash.com/photo-1536657464919-892534f60d6e?auto=format&fit=crop&w=400&q=80", use_container_width=True)
    st.sidebar.title("Configuration")
    
    # Model Selection
    available_models = ["resnet50", "efficientnet_b0", "custom_cnn"]
    model_choice = st.sidebar.selectbox(
        "Select Deep Learning Model",
        available_models,
        index=0,
        format_func=lambda x: {
            "resnet50": "ResNet50 (Transfer Learning - 96.9% Acc)",
            "efficientnet_b0": "EfficientNet-B0 (Lightweight - 96.7% Acc)",
            "custom_cnn": "Custom 4-Block CNN (Baseline - 45.4% Acc)"
        }.get(x, x)
    )

    predictor = load_predictor(model_choice)
    
    # Explainability Settings
    enable_gradcam = st.sidebar.checkbox("Enable Grad-CAM Explainability", value=True)
    cam_alpha = st.sidebar.slider("Heatmap Intensity (Alpha)", min_value=0.1, max_value=0.9, value=0.5, step=0.05)
    top_k_count = st.sidebar.slider("Top Predictions (K)", min_value=2, max_value=5, value=3)

    # Main Header
    st.markdown('<div class="main-header">🌾 Paddy Leaf Disease Classification & Diagnostic System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Research-Grade Deep Learning & Explainable Computer Vision (Grad-CAM) for Rice Crop Pathology</div>', unsafe_allow_html=True)

    tabs = st.tabs(["🔍 Disease Diagnosis", "📊 Benchmark Comparison", "📖 Disease Library"])

    with tabs[0]:
        col_input, col_pred = st.columns([1, 1.2], gap="large")

        with col_input:
            st.subheader("1. Input Leaf Image")
            input_mode = st.radio("Choose Input Method", ["Upload Custom Image", "Test Sample Gallery"], horizontal=True)

            selected_image = None

            if input_mode == "Upload Custom Image":
                uploaded_file = st.file_uploader("Upload Paddy Leaf Image (JPG, PNG)", type=["jpg", "jpeg", "png"])
                if uploaded_file is not None:
                    selected_image = Image.open(uploaded_file).convert("RGB")
                    st.image(selected_image, caption="Uploaded Leaf Image", use_container_width=True)
            else:
                # Sample gallery
                st.info("Select a pre-loaded leaf sample:")
                sample_options = {
                    "Sample 1 - Bacterial Leaf Blight": "results/plots/dataset_analysis/sample_grid_per_class.png",
                    "Sample 2 - Diagnostic Heatmap Demo": "results/plots/gradcam/sample_explanation.png"
                }
                sample_pick = st.selectbox("Pick Sample", list(sample_options.keys()))
                sample_file = PROJECT_ROOT / sample_options[sample_pick]
                if sample_file.exists():
                    selected_image = Image.open(sample_file).convert("RGB")
                    st.image(selected_image, caption=f"Sample: {sample_pick}", use_container_width=True)

        with col_pred:
            st.subheader("2. Diagnostic Results")

            if selected_image is None:
                st.info("👈 Upload an image or select a sample from the left panel to run disease diagnosis.")
            elif predictor is None:
                st.warning(f"Model checkpoint for '{model_choice}' is not ready yet. Please ensure model training has finished.")
            else:
                with st.spinner("Analyzing leaf pathology & generating Grad-CAM heatmaps..."):
                    results = predictor.predict(
                        image_input=selected_image,
                        top_k=top_k_count,
                        generate_gradcam=enable_gradcam
                    )

                pred_class = results["predicted_class"]
                conf_pct = results["confidence_percentage"]
                info = DISEASE_INFO.get(pred_class, {
                    "name": pred_class.replace("_", " ").title(),
                    "symptoms": "N/A",
                    "management": "Consult local agricultural extension expert."
                })

                # Result Badge
                st.markdown(f'<div class="pred-badge">{info["name"]}</div>', unsafe_allow_html=True)
                st.metric(label="Prediction Confidence", value=conf_pct)

                # Symptoms & Management Expanders
                with st.expander("🩺 Pathological Symptoms", expanded=True):
                    st.write(info["symptoms"])

                with st.expander("🛡️ Agronomic Management Recommendations", expanded=True):
                    st.write(info["management"])

                # Top-K Probabilities Bar Chart
                st.markdown("#### Top Class Probabilities")
                top_k_df = pd.DataFrame(results["top_k"])
                st.bar_chart(data=top_k_df.set_index("display_name")["confidence"], use_container_width=True)

                # Grad-CAM Visualizations
                if enable_gradcam and "gradcam_superimposed" in results:
                    st.markdown("#### 🔬 Grad-CAM Attention Heatmap")
                    st.caption("Highlights specific leaf lesions and necrotic patterns influencing the model's decision.")
                    
                    cam_col1, cam_col2 = st.columns(2)
                    with cam_col1:
                        st.image(results["gradcam_heatmap"], caption="Class Activation Heatmap", use_container_width=True)
                    with cam_col2:
                        st.image(results["gradcam_superimposed"], caption="Lesion Overlay Map", use_container_width=True)

    with tabs[1]:
        st.subheader("Model Performance Benchmark on Untouched Test Set (N=1,562)")
        
        benchmark_data = [
            {"Model": "ResNet50", "Test Accuracy": "96.93%", "Macro F1": "0.9696", "Weighted F1": "0.9694", "Parameters": "23,528,522", "Latency (ms/img)": "9.44 ms", "Size (MB)": "90.1 MB"},
            {"Model": "EfficientNet-B0", "Test Accuracy": "96.73%", "Macro F1": "0.9654", "Weighted F1": "0.9671", "Parameters": "4,020,358", "Latency (ms/img)": "7.08 ms", "Size (MB)": "15.6 MB"},
            {"Model": "Custom CNN (Baseline)", "Test Accuracy": "45.39%", "Macro F1": "0.3163", "Weighted F1": "0.4150", "Parameters": "423,338", "Latency (ms/img)": "10.82 ms", "Size (MB)": "4.9 MB"}
        ]
        
        df_bench = pd.DataFrame(benchmark_data)
        st.dataframe(df_bench, use_container_width=True, hide_index=True)

        st.markdown("#### Confusion Matrix Gallery")
        cm_col1, cm_col2 = st.columns(2)
        resnet_cm = PROJECT_ROOT / "results" / "metrics" / "resnet50_test_confusion_matrix_norm.png"
        eff_cm = PROJECT_ROOT / "results" / "metrics" / "efficientnet_b0_test_confusion_matrix_norm.png"
        
        if resnet_cm.exists():
            with cm_col1:
                st.image(str(resnet_cm), caption="ResNet50 Normalized Test Confusion Matrix", use_container_width=True)
        if eff_cm.exists():
            with cm_col2:
                st.image(str(eff_cm), caption="EfficientNet-B0 Normalized Test Confusion Matrix", use_container_width=True)

    with tabs[2]:
        st.subheader("Rice Disease Diagnostic Reference Catalog (10 Classes)")
        for key, details in DISEASE_INFO.items():
            with st.expander(f"📌 {details['name']}"):
                st.markdown(f"**Symptoms**: {details['symptoms']}")
                st.markdown(f"**Recommended Action**: {details['management']}")

    # Footer Disclaimer
    st.markdown("""
    <div class="disclaimer-box">
        <strong>⚠️ Decision-Support Disclaimer:</strong> This deep learning system is intended for scientific research and automated preliminary screening. Field diagnosis should always be verified alongside local agricultural extension specialists and environmental observations.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
