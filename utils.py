"""
Utility Functions for MNIST Digit Recognition

This module provides helper functions for data loading, visualization,
and evaluation.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Tuple, List, Optional


def get_data_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get data transformations for training and testing.
    
    Returns:
        Tuple of (train_transform, test_transform)
    """
    # Training transforms with data augmentation
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),  # Slight rotation for augmentation
        transforms.RandomAffine(
            degrees=0, 
            translate=(0.1, 0.1),  # Small translations
            scale=(0.9, 1.1)  # Slight scaling
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    # Test transforms (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    return train_transform, test_transform


def load_mnist_data(
    batch_size: int = 64,
    data_dir: str = './data',
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    Load MNIST dataset and create data loaders.
    
    Args:
        batch_size: Number of samples per batch
        data_dir: Directory to store/load the dataset
        num_workers: Number of worker processes for data loading
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    train_transform, test_transform = get_data_transforms()
    
    # Download and load training data
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )
    
    # Download and load test data
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"✓ Loaded {len(train_dataset):,} training samples")
    print(f"✓ Loaded {len(test_dataset):,} test samples")
    
    return train_loader, test_loader


def plot_training_history(
    history: dict,
    save_path: Optional[str] = None
) -> None:
    """
    Plot training and validation metrics.
    
    Args:
        history: Dictionary containing 'train_loss', 'val_loss', 
                'train_acc', 'val_acc'
        save_path: Path to save the figure (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Set style
    plt.style.use('default')
    colors = {'train': '#2ecc71', 'val': '#e74c3c'}
    
    # Plot Loss
    ax1 = axes[0]
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'o-', color=colors['train'], 
             linewidth=2, markersize=6, label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 's-', color=colors['val'], 
             linewidth=2, markersize=6, label='Validation Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor('#f8f9fa')
    
    # Plot Accuracy
    ax2 = axes[1]
    ax2.plot(epochs, history['train_acc'], 'o-', color=colors['train'], 
             linewidth=2, markersize=6, label='Training Accuracy')
    ax2.plot(epochs, history['val_acc'], 's-', color=colors['val'], 
             linewidth=2, markersize=6, label='Validation Accuracy')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        print(f"✓ Saved training curves to {save_path}")
    
    plt.show()


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    save_path: Optional[str] = None
) -> None:
    """
    Plot confusion matrix for classification results.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        save_path: Path to save the figure (optional)
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=range(10),
        yticklabels=range(10),
        ax=ax,
        square=True,
        linewidths=0.5,
        cbar_kws={'shrink': 0.8}
    )
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"✓ Saved confusion matrix to {save_path}")
    
    plt.show()


def plot_sample_predictions(
    images: torch.Tensor,
    true_labels: torch.Tensor,
    pred_labels: torch.Tensor,
    probabilities: torch.Tensor,
    num_samples: int = 16,
    save_path: Optional[str] = None
) -> None:
    """
    Plot sample predictions with confidence scores.
    
    Args:
        images: Batch of images
        true_labels: Ground truth labels
        pred_labels: Predicted labels
        probabilities: Prediction probabilities
        num_samples: Number of samples to display
        save_path: Path to save the figure (optional)
    """
    num_samples = min(num_samples, len(images))
    cols = 4
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    axes = axes.flatten()
    
    for i in range(num_samples):
        ax = axes[i]
        
        # Get image (denormalize)
        img = images[i].squeeze().cpu().numpy()
        img = img * 0.3081 + 0.1307  # Denormalize
        img = np.clip(img, 0, 1)
        
        # Get labels and confidence
        true_label = true_labels[i].item()
        pred_label = pred_labels[i].item()
        confidence = probabilities[i][pred_label].item() * 100
        
        # Plot image
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        
        # Set title with color based on correctness
        correct = true_label == pred_label
        color = '#2ecc71' if correct else '#e74c3c'
        title = f'True: {true_label} | Pred: {pred_label}\nConf: {confidence:.1f}%'
        ax.set_title(title, fontsize=10, color=color, fontweight='bold')
    
    # Hide empty subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Sample Predictions', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"✓ Saved sample predictions to {save_path}")
    
    plt.show()


def get_device() -> torch.device:
    """
    Get the best available device (CUDA, MPS, or CPU).
    
    Returns:
        torch.device object
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("✓ Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device('cpu')
        print("✓ Using CPU")
    
    return device


def print_classification_report(y_true: List[int], y_pred: List[int]) -> None:
    """
    Print detailed classification report.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
    """
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(
        y_true, 
        y_pred, 
        target_names=[f'Digit {i}' for i in range(10)]
    ))
