"""
CLI Script to run disease prediction and Grad-CAM visualization on an image.
"""

import argparse
import json
import sys
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predict import PaddyDiseasePredictor
from src.utils.logger import setup_logger

logger = setup_logger("predict_cli")


def parse_args():
    parser = argparse.ArgumentParser(description="Predict Paddy Leaf Disease from Image")
    parser.add_argument("--image", type=str, required=True, help="Path to input leaf image")
    parser.add_argument("--model", type=str, default="resnet50", help="Model name")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--top_k", type=int, default=3, help="Top-K predictions to return")
    parser.add_argument("--gradcam", action="store_true", help="Generate and save Grad-CAM explanation")
    parser.add_argument("--output", type=str, default=None, help="Output path for Grad-CAM image")
    return parser.parse_args()


def main():
    args = parse_args()
    
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = str(PROJECT_ROOT / "experiments" / args.model / "best_model.pth")

    predictor = PaddyDiseasePredictor(
        model_name=args.model,
        checkpoint_path=checkpoint_path,
        config_path=str(PROJECT_ROOT / "configs" / "config.yaml")
    )

    result = predictor.predict(
        image_input=args.image,
        top_k=args.top_k,
        generate_gradcam=args.gradcam
    )

    print("\n" + "=" * 50)
    print(f"Predicted Disease: {result['predicted_display_name']}")
    print(f"Confidence:        {result['confidence_percentage']}")
    print("-" * 50)
    print("Top Predictions:")
    for i, item in enumerate(result["top_k"], 1):
        print(f"  {i}. {item['display_name']:<25} {item['percentage']}")
    print("=" * 50 + "\n")

    if args.gradcam and "gradcam_superimposed" in result:
        out_file = args.output or str(PROJECT_ROOT / "results" / "predictions" / "gradcam_result.png")
        Path(out_file).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(result["gradcam_superimposed"]).save(out_file)
        logger.info(f"Grad-CAM explanation saved to {out_file}")


if __name__ == "__main__":
    main()
