"""
CLI Script to train a deep learning model on Paddy Leaf Disease dataset.
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
from src.models import get_model
from src.training.train import train_model
from src.utils.logger import setup_logger
from src.utils.seed import set_seed

logger = setup_logger("train_cli")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Paddy Leaf Disease Detection Model")
    parser.add_argument(
        "--model",
        type=str,
        default="custom_cnn",
        choices=["custom_cnn", "resnet50", "efficientnet_b0", "densenet121", "mobilenet_v3"],
        help="Model architecture name"
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs from config")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size from config")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze backbone for feature extraction")
    parser.add_argument("--pretrained", action="store_true", default=True, help="Use ImageNet pretrained weights")
    return parser.parse_args()


def main():
    args = parse_args()
    
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Overrides
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr

    seed = config["project"].get("seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Load DataLoaders
    train_loader, val_loader, _, class_names = get_dataloaders(config)

    # Instantiate Model
    num_classes = len(class_names)
    logger.info(f"Instantiating model '{args.model}' with {num_classes} classes...")
    
    model_kwargs = {"num_classes": num_classes}
    if args.model != "custom_cnn":
        model_kwargs["pretrained"] = args.pretrained
        model_kwargs["freeze_backbone"] = args.freeze_backbone

    model = get_model(args.model, **model_kwargs)
    
    total_p, train_p = model.get_num_parameters()
    logger.info(f"Model parameters -> Total: {total_p:,}, Trainable: {train_p:,}")

    experiment_dir = str(PROJECT_ROOT / "experiments" / args.model)

    best_model_path, history, summary = train_model(
        model=model,
        model_name=args.model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_names=class_names,
        config=config,
        experiment_dir=experiment_dir,
        device=device
    )

    logger.info(f"Model training process complete! Checkpoint: {best_model_path}")


if __name__ == "__main__":
    main()
