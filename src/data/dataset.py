"""
PyTorch Dataset and DataLoader Pipeline for Paddy Leaf Disease Classification.
"""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from datasets import Dataset, load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset as TorchDataset
import yaml

from src.data.create_splits import load_split_indices, create_stratified_splits
from src.preprocessing.transforms import (
    get_train_transforms,
    get_val_transforms,
    get_test_transforms,
)
from src.utils.logger import setup_logger

logger = setup_logger("paddy_dataset")


class PaddyDataset(TorchDataset):
    """
    PyTorch Dataset wrapper for Hugging Face Paddy Leaf Disease dataset.
    """
    def __init__(
        self,
        hf_dataset: Dataset,
        indices: List[int],
        transform: Optional[Callable] = None,
        return_metadata: bool = False
    ):
        self.hf_dataset = hf_dataset
        self.indices = indices
        self.transform = transform
        self.return_metadata = return_metadata

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, int], Tuple[torch.Tensor, int, Dict[str, Any]]]:
        actual_idx = self.indices[idx]
        item = self.hf_dataset[actual_idx]

        image = item["image"]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = int(item["label"])

        if self.return_metadata:
            metadata = {
                "dataset_idx": actual_idx,
                "variety": int(item.get("variety", -1)),
                "age": int(item.get("age", -1))
            }
            return image, label, metadata

        return image, label


def get_dataloaders(
    config_or_path: Union[str, Dict[str, Any]],
    return_metadata: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Build and return Train, Validation, and Test DataLoaders.
    """
    if isinstance(config_or_path, str):
        with open(config_or_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        config = config_or_path

    hf_path = config["dataset"].get("hf_path", "anthony2261/paddy-disease-classification")
    image_size = tuple(config["dataset"].get("image_size", [224, 224]))
    split_file = config["dataset"].get("split_file", "data/splits/split_indices.json")
    batch_size = config["training"].get("batch_size", 32)
    num_workers = config["training"].get("num_workers", 2)
    pin_memory = torch.cuda.is_available()

    # Ensure split file exists
    if not Path(split_file).exists():
        logger.info(f"Split file '{split_file}' not found. Generating stratified splits...")
        create_stratified_splits(
            dataset_name=hf_path,
            random_seed=config["project"].get("seed", 42),
            output_path=split_file
        )

    split_data = load_split_indices(split_file)
    train_indices = split_data["train_indices"]
    val_indices = split_data["val_indices"]
    test_indices = split_data["test_indices"]
    class_names = split_data["class_names"]

    logger.info(f"Loading raw dataset '{hf_path}'...")
    raw_ds = load_dataset(hf_path)["train"]

    # Construct Transforms
    train_transform = get_train_transforms(
        image_size=image_size,
        mean=config["preprocessing"].get("mean", [0.485, 0.456, 0.406]),
        std=config["preprocessing"].get("std", [0.229, 0.224, 0.225]),
        hflip_prob=config["preprocessing"]["augmentation"].get("horizontal_flip_prob", 0.5),
        vflip_prob=config["preprocessing"]["augmentation"].get("vertical_flip_prob", 0.2),
        rotation_deg=config["preprocessing"]["augmentation"].get("rotation_degrees", 15)
    )
    val_transform = get_val_transforms(
        image_size=image_size,
        mean=config["preprocessing"].get("mean", [0.485, 0.456, 0.406]),
        std=config["preprocessing"].get("std", [0.229, 0.224, 0.225])
    )
    test_transform = get_test_transforms(
        image_size=image_size,
        mean=config["preprocessing"].get("mean", [0.485, 0.456, 0.406]),
        std=config["preprocessing"].get("std", [0.229, 0.224, 0.225])
    )

    # Instantiate Datasets
    train_dataset = PaddyDataset(
        hf_dataset=raw_ds,
        indices=train_indices,
        transform=train_transform,
        return_metadata=return_metadata
    )
    val_dataset = PaddyDataset(
        hf_dataset=raw_ds,
        indices=val_indices,
        transform=val_transform,
        return_metadata=return_metadata
    )
    test_dataset = PaddyDataset(
        hf_dataset=raw_ds,
        indices=test_indices,
        transform=test_transform,
        return_metadata=return_metadata
    )

    # Windows safe workers configuration
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    logger.info(
        f"DataLoaders created -> Train Batches: {len(train_loader)} ({len(train_dataset)} samples), "
        f"Val Batches: {len(val_loader)} ({len(val_dataset)} samples), "
        f"Test Batches: {len(test_loader)} ({len(test_dataset)} samples)"
    )

    return train_loader, val_loader, test_loader, class_names
