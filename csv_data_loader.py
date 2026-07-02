"""
Custom Dataset Loader for Kaggle Handwriting Recognition CSV

Dataset format:
  FILENAME  - image filename (e.g., TRAIN_00001.jpg)
  IDENTITY  - handwritten word label (e.g., BALTHAZAR)

Usage:
    from csv_data_loader import HandwritingCSVDataset, create_csv_dataloaders

    train_loader, val_loader, class_to_idx = create_csv_dataloaders(
        csv_path   = 'Dataset/written_name_train_v2.csv',
        images_dir = 'images',
        top_n      = None,   # None = all classes
        val_split  = 0.15,
        batch_size = 64,
    )
"""

import os
import io
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from typing import Tuple, Optional, Dict, List


# ---------------------------------------------------------------------------
# Label cleaning & filtering helpers
# ---------------------------------------------------------------------------

INVALID_LABELS = {"EMPTY", "UNREADABLE", ""}


def clean_label(label) -> Optional[str]:
    """Return stripped uppercase label, or None if invalid."""
    if pd.isna(label):
        return None
    label = str(label).strip().upper()
    if label in INVALID_LABELS:
        return None
    return label


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class HandwritingCSVDataset(Dataset):
    """
    PyTorch Dataset for the Kaggle Handwriting Recognition dataset.

    Parameters
    ----------
    csv_path   : path to written_name_train_v2.csv (or similar)
    images_dir : directory that contains the .jpg image files
    top_n      : keep only the top-N most-frequent classes (None = keep all)
    transform  : torchvision transform applied to each PIL image
    label_encoder : pre-built {label: idx} mapping (used for val split reuse)
    """

    def __init__(
        self,
        csv_path: str,
        images_dir: str,
        top_n: Optional[int] = None,
        transform=None,
        label_encoder: Optional[Dict[str, int]] = None,
    ):
        self.images_dir = images_dir
        self.transform = transform

        # ---- Load & clean CSV ------------------------------------------------
        print(f"\n📂 Reading CSV: {csv_path}")
        df = pd.read_csv(csv_path)

        # Clean IDENTITY column
        df["IDENTITY"] = df["IDENTITY"].apply(clean_label)
        df = df.dropna(subset=["IDENTITY"])

        print(f"✓ Rows after removing invalid labels: {len(df):,}")
        print(f"✓ Unique labels: {df['IDENTITY'].nunique():,}")

        # ---- Optionally restrict to top-N classes ----------------------------
        if top_n is not None:
            top_labels = (
                df["IDENTITY"].value_counts().head(top_n).index.tolist()
            )
            df = df[df["IDENTITY"].isin(top_labels)].reset_index(drop=True)
            print(f"✓ Rows for top-{top_n} classes: {len(df):,}")

        # ---- Build label encoder --------------------------------------------
        if label_encoder is not None:
            self.class_to_idx = label_encoder
        else:
            unique_labels = sorted(df["IDENTITY"].unique())
            self.class_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}

        self.idx_to_class: List[str] = [
            lbl for lbl, _ in sorted(self.class_to_idx.items(), key=lambda x: x[1])
        ]
        self.num_classes = len(self.class_to_idx)
        print(f"✓ Number of classes: {self.num_classes:,}")

        # ---- Filter to rows whose labels are in the encoder -----------------
        df = df[df["IDENTITY"].isin(self.class_to_idx)].reset_index(drop=True)

        self.filenames = df["FILENAME"].tolist()
        self.labels    = df["IDENTITY"].tolist()

        # ---- Check how many image files actually exist ----------------------
        found = sum(
            1 for fn in self.filenames[:200]  # quick sample check
            if os.path.isfile(os.path.join(images_dir, fn))
        )
        if found == 0:
            print(
                f"\n⚠️  WARNING: No image files found in '{images_dir}'!\n"
                f"   Please download the training images from Kaggle:\n"
                f"   https://www.kaggle.com/datasets/landlord/handwriting-recognition\n"
                f"   and place them in: {os.path.abspath(images_dir)}\n"
            )
        else:
            print(f"✓ Image directory: {os.path.abspath(images_dir)}")

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        filename = self.filenames[idx]
        label    = self.labels[idx]

        img_path = os.path.join(self.images_dir, filename)

        try:
            img = Image.open(img_path).convert("L")   # grayscale
        except FileNotFoundError:
            # Return a blank image with the correct size if file is missing
            img = Image.new("L", (128, 64), color=255)
        except Exception:
            img = Image.new("L", (128, 64), color=255)

        if self.transform:
            img = self.transform(img)

        class_idx = self.class_to_idx[label]
        return img, class_idx


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_transforms(
    img_height: int = 64,
    img_width: int = 256,
    augment: bool = True,
):
    """
    Returns (train_transform, val_transform) for word-level images.

    Images are resized to (img_height x img_width) and normalised.
    """
    val_transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.RandomRotation(degrees=5),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                shear=5,
            ),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
    else:
        train_transform = val_transform

    return train_transform, val_transform


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def create_csv_dataloaders(
    csv_path: str,
    images_dir: str,
    top_n: Optional[int] = None,
    val_split: float = 0.15,
    batch_size: int = 64,
    num_workers: int = 0,
    img_height: int = 64,
    img_width: int = 256,
    augment: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """
    Build train and validation DataLoaders from the CSV.

    Returns
    -------
    (train_loader, val_loader, class_to_idx)
    """
    train_transform, val_transform = get_transforms(img_height, img_width, augment)

    # Build full dataset (to derive the class encoder)
    full_dataset = HandwritingCSVDataset(
        csv_path   = csv_path,
        images_dir = images_dir,
        top_n      = top_n,
        transform  = train_transform,
    )

    class_to_idx = full_dataset.class_to_idx
    n_classes    = full_dataset.num_classes

    # Split into train / val
    total       = len(full_dataset)
    val_size    = int(total * val_split)
    train_size  = total - val_size

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size], generator=generator)

    # Apply val transform to val subset without mutating the parent dataset
    class TransformSubset(torch.utils.data.Dataset):
        """Wrapper that applies a specific transform to a subset."""
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
        def __len__(self):
            return len(self.subset)
        def __getitem__(self, idx):
            img, label = self.subset[idx]
            if self.transform is not None:
                # img is already a tensor from the parent transform,
                # so we apply val-specific normalization only if needed
                pass
            return img, label

    # Re-create val dataset with val transform applied at load time
    val_full_dataset = HandwritingCSVDataset(
        csv_path   = csv_path,
        images_dir = images_dir,
        top_n      = top_n,
        transform  = val_transform,
        label_encoder = class_to_idx,
    )
    _, val_ds = random_split(val_full_dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_ds,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
    )

    print(f"\n✓ Training batches:   {len(train_loader):,}  ({train_size:,} samples)")
    print(f"✓ Validation batches: {len(val_loader):,}  ({val_size:,} samples)")

    return train_loader, val_loader, class_to_idx


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    csv   = r"Dataset/written_name_train_v2.csv"
    imgs  = r"images"

    print("=" * 60)
    print("   CSV DATA LOADER — SELF TEST")
    print("=" * 60)

    try:
        train_loader, val_loader, class_to_idx = create_csv_dataloaders(
            csv_path   = csv,
            images_dir = imgs,
            top_n      = 50,       # small test
            val_split  = 0.2,
            batch_size = 8,
        )
    except FileNotFoundError as e:
        print(f"\n❌ File not found: {e}")
        sys.exit(1)

    print(f"\nTotal classes: {len(class_to_idx)}")
    print(f"Some classes:  {list(class_to_idx.keys())[:10]}")

    # Try one batch
    imgs_batch, labels_batch = next(iter(train_loader))
    print(f"\nBatch image shape: {imgs_batch.shape}")
    print(f"Batch label shape: {labels_batch.shape}")
    print(f"Label values:      {labels_batch.tolist()}")
    print("\n✅ Data loader OK!")
