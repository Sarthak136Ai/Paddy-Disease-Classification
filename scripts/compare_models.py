"""
CLI Script to generate model comparison benchmark tables and plots.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.compare_models import generate_comparison_table_and_plots
from src.utils.logger import setup_logger

logger = setup_logger("compare_cli")

def main():
    logger.info("Compiling model benchmarks...")
    df = generate_comparison_table_and_plots(
        metrics_dir=str(PROJECT_ROOT / "results" / "metrics"),
        plots_dir=str(PROJECT_ROOT / "results" / "plots" / "model_comparison")
    )
    print("\n" + "=" * 90)
    print("FINAL RESEARCH BENCHMARK COMPARISON TABLE (UNTOUCHED TEST SET N=1,562)")
    print("=" * 90)
    print(df.to_string(index=False))
    print("=" * 90 + "\n")

if __name__ == "__main__":
    main()
