"""
CLI Script to execute dataset verification and generate analysis report.
"""
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.verify_dataset import load_raw_dataset, verify_and_analyze_dataset
from src.utils.seed import set_seed
from src.utils.logger import setup_logger

logger = setup_logger("verify_dataset_script")

def main():
    logger.info("Initializing Paddy Leaf Disease Dataset Verification...")
    set_seed(42)
    
    ds = load_raw_dataset("anthony2261/paddy-disease-classification")
    report = verify_and_analyze_dataset(
        ds=ds,
        output_dir=str(PROJECT_ROOT / "results"),
        plots_dir=str(PROJECT_ROOT / "results" / "plots" / "dataset_analysis")
    )
    logger.info("Dataset verification completed successfully!")
    logger.info(f"Verified {report['total_images']} images across {report['num_classes']} classes.")

if __name__ == "__main__":
    main()
