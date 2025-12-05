"""
Dataset and DataLoader utilities for SAM3 LoRA fine-tuning.

This module provides dataset classes compatible with the SAM3 training pipeline.
"""

import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class LoRASAM3Dataset(Dataset):
    """
    Dataset for SAM3 LoRA fine-tuning using COCO format annotations.

    Args:
        img_folder: Path to image folder
        ann_file: Path to COCO format annotation file
        transforms: Optional transforms to apply
        max_ann_per_img: Maximum annotations per image
    """

    def __init__(
        self,
        img_folder: str,
        ann_file: str,
        transforms: Optional[Callable] = None,
        max_ann_per_img: int = 200,
    ):
        self.img_folder = img_folder
        self.ann_file = ann_file
        self.transforms = transforms
        self.max_ann_per_img = max_ann_per_img

        # Load annotations
        self.data = self._load_annotations()

    def _load_annotations(self) -> List[Dict[str, Any]]:
        """Load and parse COCO format annotations."""
        with open(self.ann_file, "r") as f:
            coco_data = json.load(f)

        # Build image id to annotations mapping
        img_to_anns = {}
        for ann in coco_data["annotations"]:
            img_id = ann.get("image_id", ann.get("id"))
            if img_id not in img_to_anns:
                img_to_anns[img_id] = []
            img_to_anns[img_id].append(ann)

        # Build dataset
        data = []
        for img_info in coco_data["images"]:
            img_id = img_info.get("id", img_info.get("image_id"))
            anns = img_to_anns.get(img_id, [])

            # Skip images with no annotations
            if len(anns) == 0:
                continue

            # Limit annotations per image
            if len(anns) > self.max_ann_per_img:
                anns = anns[: self.max_ann_per_img]

            data.append(
                {
                    "image": img_info,
                    "annotations": anns,
                }
            )

        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single data sample.

        Returns:
            Dictionary containing:
                - image: PIL Image or tensor (after transforms)
                - annotations: List of annotation dicts
                - image_info: Image metadata
        """
        item = self.data[idx]

        # Load image
        img_path = os.path.join(self.img_folder, item["image"]["file_name"])
        image = Image.open(img_path).convert("RGB")

        # Prepare sample
        sample = {
            "image": image,
            "annotations": item["annotations"],
            "image_info": item["image"],
        }

        # Apply transforms if provided
        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for batching samples.

    Args:
        batch: List of samples from dataset

    Returns:
        Batched data
    """
    # Stack images if they're tensors
    if isinstance(batch[0]["image"], torch.Tensor):
        images = torch.stack([item["image"] for item in batch])
    else:
        images = [item["image"] for item in batch]

    return {
        "images": images,
        "annotations": [item["annotations"] for item in batch],
        "image_info": [item["image_info"] for item in batch],
    }


def create_dataloaders(
    train_img_folder: str,
    train_ann_file: str,
    val_img_folder: str,
    val_ann_file: str,
    batch_size: int = 1,
    num_workers: int = 4,
    pin_memory: bool = True,
    train_transforms: Optional[Callable] = None,
    val_transforms: Optional[Callable] = None,
    max_ann_per_img: int = 200,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders.

    Args:
        train_img_folder: Path to training images
        train_ann_file: Path to training annotations
        val_img_folder: Path to validation images
        val_ann_file: Path to validation annotations
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory
        train_transforms: Training transforms
        val_transforms: Validation transforms
        max_ann_per_img: Maximum annotations per image

    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Create datasets
    train_dataset = LoRASAM3Dataset(
        img_folder=train_img_folder,
        ann_file=train_ann_file,
        transforms=train_transforms,
        max_ann_per_img=max_ann_per_img,
    )

    val_dataset = LoRASAM3Dataset(
        img_folder=val_img_folder,
        ann_file=val_ann_file,
        transforms=val_transforms,
        max_ann_per_img=max_ann_per_img,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=False,
    )

    return train_loader, val_loader
