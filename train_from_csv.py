"""
Training Script for Handwriting Recognition using CSV Dataset

Trains a word-level CNN on the Kaggle Handwriting Recognition dataset.

Usage
-----
# Train on ALL classes (requires images downloaded):
  python train_from_csv.py \
      --csv    "Dataset/written_name_train_v2.csv" \
      --images "images"

# Train on top-100 most-common names:
  python train_from_csv.py \
      --csv   "Dataset/written_name_train_v2.csv" \
      --images "images" \
      --top-n 100

# Full options:
  python train_from_csv.py --help
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# ── project imports ──────────────────────────────────────────────────────────
from csv_data_loader import create_csv_dataloaders
from model_word import WordCNN, get_word_model_summary


# ═══════════════════════════════════════════════════════════════════════════
# Single epoch helpers
# ═══════════════════════════════════════════════════════════════════════════

def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> Tuple[float, float]:
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    bar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Train]", leave=False)
    for images, labels in bar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total   += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        bar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{100.*correct/total:.2f}%",
        )

    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="Validating", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss    = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted  = outputs.max(1)
        total   += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, 100.0 * correct / total


# ═══════════════════════════════════════════════════════════════════════════
# Visualisation helpers
# ═══════════════════════════════════════════════════════════════════════════

def save_training_curves(
    history: Dict[str, List[float]],
    save_path: str,
) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=4)
    ax1.plot(epochs, history["val_loss"],   "r-o", label="Val Loss",   markersize=4)
    ax1.set_title("Loss per Epoch", fontsize=13)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], "b-o", label="Train Acc", markersize=4)
    ax2.plot(epochs, history["val_acc"],   "r-o", label="Val Acc",   markersize=4)
    ax2.set_title("Accuracy per Epoch", fontsize=13)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Handwriting Recognition — Training Curves", fontsize=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Training curves → {save_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main training loop
# ═══════════════════════════════════════════════════════════════════════════

def train(args) -> Dict:
    # ── device ───────────────────────────────────────────────────────────
    if args.no_cuda or not torch.cuda.is_available():
        device = torch.device("cpu")
        print("ℹ️  Using CPU")
    else:
        device = torch.device("cuda")
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(args.save_dir,   exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── data ─────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("   📂 LOADING DATASET")
    print("="*60)

    top_n_label = "ALL" if args.top_n is None else str(args.top_n)

    train_loader, val_loader, class_to_idx = create_csv_dataloaders(
        csv_path   = args.csv,
        images_dir = args.images,
        top_n      = args.top_n,
        val_split  = args.val_split,
        batch_size = args.batch_size,
        num_workers= args.workers,
        img_height = args.img_height,
        img_width  = args.img_width,
        augment    = not args.no_augment,
    )

    num_classes = len(class_to_idx)

    # Save class mapping
    mapping_path = os.path.join(args.save_dir, "class_to_idx.json")
    with open(mapping_path, "w") as f:
        json.dump(class_to_idx, f, indent=2)
    print(f"✓ Class mapping saved → {mapping_path}")

    # ── model ─────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("   🏗️  BUILDING MODEL")
    print("="*60)
    model = WordCNN(num_classes=num_classes, dropout=args.dropout)
    print(get_word_model_summary(model))
    model = model.to(device)

    # ── optimizer, scheduler, loss ────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(
        model.parameters(),
        lr           = args.lr,
        weight_decay = 1e-4,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    # ── training ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("   🚀 STARTING TRAINING")
    print("="*60)
    print(f"  CSV:          {args.csv}")
    print(f"  Images dir:   {args.images}")
    print(f"  Top-N:        {top_n_label}")
    print(f"  Classes:      {num_classes:,}")
    print(f"  Epochs:       {args.epochs}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  LR:           {args.lr}")
    print(f"  Val split:    {args.val_split:.0%}")
    print(f"  Image size:   {args.img_height} x {args.img_width}")
    print(f"  Device:       {device}")
    print("="*60)

    history: Dict[str, List[float]] = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
    }
    best_val_acc = 0.0
    start_time   = time.time()

    for epoch in range(1, args.epochs + 1):
        print(f"\n📊 Epoch {epoch}/{args.epochs}  (LR={optimizer.param_groups[0]['lr']:.2e})")
        print("-" * 50)

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")

        # Checkpoint best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(args.save_dir, "csv_model_best.pth")
            torch.save(
                {
                    "epoch":            epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc":          val_acc,
                    "val_loss":         val_loss,
                    "num_classes":      num_classes,
                    "class_to_idx":     class_to_idx,
                    "img_height":       args.img_height,
                    "img_width":        args.img_width,
                    "top_n":            args.top_n,
                },
                ckpt_path,
            )
            print(f"  ✓ Best model saved  (val_acc={val_acc:.2f}%) → {ckpt_path}")

        # Save training curves after each epoch (overwriters previous)
        curves_path = os.path.join(args.output_dir, "csv_training_curves.png")
        save_training_curves(history, curves_path)

    elapsed = time.time() - start_time

    # ── final model ───────────────────────────────────────────────────────
    final_path = os.path.join(args.save_dir, "csv_model_final.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "val_acc":          val_acc,
            "num_classes":      num_classes,
            "class_to_idx":     class_to_idx,
            "img_height":       args.img_height,
            "img_width":        args.img_width,
            "top_n":            args.top_n,
        },
        final_path,
    )

    print("\n" + "="*60)
    print("   🎉 TRAINING COMPLETE!")
    print("="*60)
    print(f"  Total Time:    {elapsed/60:.1f} minutes")
    print(f"  Best Val Acc:  {best_val_acc:.2f}%")
    print(f"  Final Val Acc: {val_acc:.2f}%")
    print(f"  Best model  → {ckpt_path}")
    print(f"  Final model → {final_path}")
    print(f"  Curves      → {curves_path}")
    print("="*60)

    return history


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Train a CNN on the Kaggle Handwriting Recognition CSV dataset."
    )

    # Data
    p.add_argument("--csv",       type=str,
                   default=r"Dataset/written_name_train_v2.csv",
                   help="Path to written_name_train_v2.csv")
    p.add_argument("--images",    type=str,
                   default="images",
                   help="Directory containing the .jpg image files")
    p.add_argument("--top-n",     type=int, default=None,
                   help="Train on only the top-N most-common classes (default: ALL)")
    p.add_argument("--val-split", type=float, default=0.15,
                   help="Fraction of data used for validation (default: 0.15)")
    p.add_argument("--img-height",type=int, default=64,
                   help="Resize image height (default: 64)")
    p.add_argument("--img-width", type=int, default=256,
                   help="Resize image width (default: 256)")

    # Training
    p.add_argument("--epochs",    type=int,   default=20,
                   help="Number of training epochs (default: 20)")
    p.add_argument("--batch-size",type=int,   default=64,
                   help="Batch size (default: 64)")
    p.add_argument("--lr",        type=float, default=1e-3,
                   help="Initial learning rate (default: 0.001)")
    p.add_argument("--dropout",   type=float, default=0.4,
                   help="Dropout rate in classifier head (default: 0.4)")

    # Misc
    p.add_argument("--no-cuda",    action="store_true",
                   help="Disable CUDA (use CPU)")
    p.add_argument("--no-augment", action="store_true",
                   help="Disable training data augmentation")
    p.add_argument("--workers",    type=int, default=0,
                   help="Number of DataLoader workers (default: 0)")
    p.add_argument("--save-dir",   type=str, default="models",
                   help="Directory to save model checkpoints (default: models/)")
    p.add_argument("--output-dir", type=str, default="outputs",
                   help="Directory to save training plots (default: outputs/)")

    return p.parse_args()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("   ✍️  HANDWRITING WORD RECOGNITION — CNN TRAINING")
    print("   Dataset: Kaggle written_name_train_v2.csv")
    print("="*60)

    args = parse_args()

    # Basic sanity check
    if not os.path.isfile(args.csv):
        print(f"\n❌ CSV not found: {args.csv}")
        print("   Please provide the correct path via --csv")
        sys.exit(1)

    if not os.path.isdir(args.images):
        print(f"\n⚠️  Images directory not found: {args.images}")
        print("   Please download the image files from Kaggle and place them in:")
        print(f"   {os.path.abspath(args.images)}")
        print("   URL: https://www.kaggle.com/datasets/landlord/handwriting-recognition")
        print("\n   Training will continue but images will be blank placeholders.\n")

    train(args)
