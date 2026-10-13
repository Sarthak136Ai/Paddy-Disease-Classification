"""
Stratified Dataset Splitting Module.
Generates and persists reproducible 70% train / 15% validation / 15% test split indices
based on disease class labels.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from datasets import Dataset, load_dataset
from sklearn.model_selection import train_test_split

from src.utils.logger import setup_logger
from src.utils.seed import set_seed

logger = setup_logger("create_splits")


def create_stratified_splits(
    dataset_name: str = "anthony2261/paddy-disease-classification",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    output_path: str = "data/splits/split_indices.json"
) -> Dict[str, Any]:
    """
    Generate stratified train/val/test splits and save index mapping to disk.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Split ratios must sum to 1.0"
    set_seed(random_seed)

    logger.info(f"Loading dataset '{dataset_name}' for splitting...")
    ds = load_dataset(dataset_name)["train"]
    total_samples = len(ds)
    logger.info(f"Total dataset samples: {total_samples}")

    class_names = ds.features["label"].names
    labels = np.array(ds["label"])
    indices = np.arange(total_samples)

    # 1st split: Train vs Temp (Val + Test)
    temp_ratio = val_ratio + test_ratio
    train_idx, temp_idx, y_train, y_temp = train_test_split(
        indices,
        labels,
        test_size=temp_ratio,
        stratify=labels,
        random_state=random_seed
    )

    # 2nd split: Val vs Test
    val_rel_ratio = val_ratio / temp_ratio
    val_idx, test_idx, y_val, y_test = train_test_split(
        temp_idx,
        y_temp,
        test_size=(1.0 - val_rel_ratio),
        stratify=y_temp,
        random_state=random_seed
    )

    train_indices = sorted(train_idx.tolist())
    val_indices = sorted(val_idx.tolist())
    test_indices = sorted(test_idx.tolist())

    logger.info(f"Split sizes -> Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")

    # Verify no overlaps
    assert set(train_indices).isdisjoint(set(val_indices)), "Data leakage: overlap between train and val!"
    assert set(train_indices).isdisjoint(set(test_indices)), "Data leakage: overlap between train and test!"
    assert set(val_indices).isdisjoint(set(test_indices)), "Data leakage: overlap between val and test!"
    assert len(train_indices) + len(val_indices) + len(test_indices) == total_samples, "Total sample count mismatch!"

    # Calculate class counts per split
    train_counts = Counter(labels[train_indices])
    val_counts = Counter(labels[val_indices])
    test_counts = Counter(labels[test_indices])

    per_class_summary = {
        cls_name: {
            "train": train_counts[i],
            "val": val_counts[i],
            "test": test_counts[i],
            "total": int(train_counts[i] + val_counts[i] + test_counts[i]),
            "train_pct": round(train_counts[i] / len(train_indices) * 100, 2),
            "val_pct": round(val_counts[i] / len(val_indices) * 100, 2),
            "test_pct": round(test_counts[i] / len(test_indices) * 100, 2),
        }
        for i, cls_name in enumerate(class_names)
    }

    split_payload = {
        "dataset_name": dataset_name,
        "random_seed": random_seed,
        "stratified": True,
        "total_samples": total_samples,
        "split_counts": {
            "train": len(train_indices),
            "val": len(val_indices),
            "test": len(test_indices)
        },
        "split_percentages": {
            "train": round(len(train_indices) / total_samples * 100, 2),
            "val": round(len(val_indices) / total_samples * 100, 2),
            "test": round(len(test_indices) / total_samples * 100, 2)
        },
        "class_names": class_names,
        "per_class_summary": per_class_summary,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(split_payload, f, indent=2)

    logger.info(f"Split indices successfully saved to {out_file.resolve()}")
    return split_payload


def load_split_indices(split_path: str = "data/splits/split_indices.json") -> Dict[str, Any]:
    """Load persisted split indices from disk."""
    with open(split_path, "r", encoding="utf-8") as f:
        return json.load(f)
