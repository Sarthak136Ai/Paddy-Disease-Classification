"""
CLI Script to evaluate a trained checkpoint on the untouched test split.
"""

import argparse
import sys
from pathlib import Path
import yaml
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import get_dataloaders
from src.evaluation.evaluate import evaluate_on_test_set
from src.models import get_model
from src.utils.logger import setup_logger

logger = setup_logger("evaluate_cli")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Paddy Leaf Disease Model on Test Set")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["custom_cnn", "resnet50", "efficientnet_b0", "densenet121", "mobilenet_v3"],
        help="Model architecture name"
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to best_model.pth")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save evaluation results")
    return parser.parse_args()


def main():
    args = parse_args()
    
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load DataLoaders
    _, _, test_loader, class_names = get_dataloaders(config)

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = str(PROJECT_ROOT / "experiments" / args.model / "best_model.pth")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = str(PROJECT_ROOT / "results" / "metrics")

    # Instantiate Model
    num_classes = len(class_names)
    model = get_model(args.model, num_classes=num_classes, pretrained=False)

    metrics = evaluate_on_test_set(
        model=model,
        model_name=args.model,
        checkpoint_path=checkpoint_path,
        test_loader=test_loader,
        class_names=class_names,
        output_dir=output_dir,
        device=device
    )

    logger.info(f"Evaluation complete for {args.model}!")
    logger.info(f"Test Accuracy: {metrics['accuracy']:.4f} | Test Macro F1: {metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
