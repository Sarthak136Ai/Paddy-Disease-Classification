"""
CLI script to generate reproducible stratified splits.
"""
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.create_splits import create_stratified_splits
from src.utils.logger import setup_logger

logger = setup_logger("create_splits_cli")

def main():
    logger.info("Initiating dataset splitting...")
    split_info = create_stratified_splits(
        dataset_name="anthony2261/paddy-disease-classification",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
        output_path=str(PROJECT_ROOT / "data" / "splits" / "split_indices.json")
    )
    logger.info("Split generation completed successfully!")
    logger.info(f"Train: {split_info['split_counts']['train']} samples")
    logger.info(f"Val:   {split_info['split_counts']['val']} samples")
    logger.info(f"Test:  {split_info['split_counts']['test']} samples")

if __name__ == "__main__":
    main()
