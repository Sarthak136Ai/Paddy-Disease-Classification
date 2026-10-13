"""
Root entry point for running the Paddy Leaf Disease Flask Web Application.

Usage:
    python app.py
"""

from flask_app.app import create_app

app = create_app()

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  PADDY LEAF DISEASE DETECTION & RESEARCH DASHBOARD (FLASK)")
    print("=" * 70)
    print("  Local URL: http://127.0.0.1:5000")
    print("  Dataset:   anthony2261/paddy-disease-classification (10,407 images)")
    print("  Models:    ResNet50, EfficientNet-B0, MobileNetV3, DenseNet121, Custom CNN")
    print("=" * 70 + "\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
