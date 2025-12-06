"""
Data Loading Utilities for EMNIST and MNIST Datasets

Supports:
- MNIST: Digits (0-9)
- EMNIST Letters: Uppercase letters (A-Z)
- EMNIST Balanced: Balanced alphanumeric (47 classes)
- EMNIST ByClass: Full alphanumeric (62 classes)
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from typing import Tuple, Literal, Optional

DatasetType = Literal['mnist', 'letters', 'balanced', 'byclass']


def get_data_transforms(augment: bool = True) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get data transformations for training and testing.
    
    Args:
        augment: Whether to apply data augmentation for training
        
    Returns:
        Tuple of (train_transform, test_transform)
    """
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomAffine(
                degrees=0, 
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    return train_transform, test_transform


def load_dataset(
    dataset_type: DatasetType = 'mnist',
    batch_size: int = 64,
    data_dir: str = './data',
    num_workers: int = 0,
    augment: bool = True,
    val_split: float = 0.0
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Load dataset and create data loaders.
    
    Args:
        dataset_type: Type of dataset to load
        batch_size: Number of samples per batch
        data_dir: Directory to store/load the dataset
        num_workers: Number of worker processes
        augment: Whether to apply data augmentation
        val_split: Fraction of training data to use for validation (0 = no validation split)
        
    Returns:
        Tuple of (train_loader, test_loader, val_loader or None)
    """
    train_transform, test_transform = get_data_transforms(augment)
    
    print(f"\n📂 Loading {dataset_type.upper()} dataset...")
    
    if dataset_type == 'mnist':
        train_dataset = datasets.MNIST(
            root=data_dir, train=True, download=True, transform=train_transform
        )
        test_dataset = datasets.MNIST(
            root=data_dir, train=False, download=True, transform=test_transform
        )
        
    elif dataset_type == 'letters':
        # EMNIST Letters: 26 classes (A-Z)
        train_dataset = datasets.EMNIST(
            root=data_dir, split='letters', train=True, download=True, transform=train_transform
        )
        test_dataset = datasets.EMNIST(
            root=data_dir, split='letters', train=False, download=True, transform=test_transform
        )
        # EMNIST letters are 1-indexed (1-26), need to adjust to 0-indexed
        train_dataset.targets = train_dataset.targets - 1
        test_dataset.targets = test_dataset.targets - 1
        
    elif dataset_type == 'balanced':
        # EMNIST Balanced: 47 classes (balanced distribution)
        train_dataset = datasets.EMNIST(
            root=data_dir, split='balanced', train=True, download=True, transform=train_transform
        )
        test_dataset = datasets.EMNIST(
            root=data_dir, split='balanced', train=False, download=True, transform=test_transform
        )
        
    elif dataset_type == 'byclass':
        # EMNIST ByClass: 62 classes (full alphanumeric)
        train_dataset = datasets.EMNIST(
            root=data_dir, split='byclass', train=True, download=True, transform=train_transform
        )
        test_dataset = datasets.EMNIST(
            root=data_dir, split='byclass', train=False, download=True, transform=test_transform
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # Optional validation split
    val_loader = None
    if val_split > 0:
        val_size = int(len(train_dataset) * val_split)
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = random_split(
            train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        print(f"✓ Validation samples: {val_size:,}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    train_size = len(train_dataset)
    test_size = len(test_dataset)
    
    print(f"✓ Training samples: {train_size:,}")
    print(f"✓ Test samples: {test_size:,}")
    
    # Get number of classes
    if hasattr(train_dataset, 'classes'):
        num_classes = len(train_dataset.classes)
    elif hasattr(train_dataset, 'dataset') and hasattr(train_dataset.dataset, 'classes'):
        num_classes = len(train_dataset.dataset.classes)
    else:
        # Infer from targets
        if hasattr(train_dataset, 'targets'):
            num_classes = len(torch.unique(train_dataset.targets))
        else:
            num_classes = {'mnist': 10, 'letters': 26, 'balanced': 47, 'byclass': 62}[dataset_type]
    
    print(f"✓ Number of classes: {num_classes}")
    
    return train_loader, test_loader, val_loader


def get_dataset_info(dataset_type: DatasetType) -> dict:
    """
    Get information about a dataset type.
    
    Args:
        dataset_type: Type of dataset
        
    Returns:
        Dictionary with dataset information
    """
    info = {
        'mnist': {
            'name': 'MNIST',
            'description': 'Handwritten Digits',
            'num_classes': 10,
            'labels': '0-9',
            'train_samples': 60000,
            'test_samples': 10000
        },
        'letters': {
            'name': 'EMNIST Letters',
            'description': 'Handwritten Letters',
            'num_classes': 26,
            'labels': 'A-Z',
            'train_samples': 124800,
            'test_samples': 20800
        },
        'balanced': {
            'name': 'EMNIST Balanced',
            'description': 'Balanced Alphanumeric',
            'num_classes': 47,
            'labels': '0-9, A-Z, some lowercase',
            'train_samples': 112800,
            'test_samples': 18800
        },
        'byclass': {
            'name': 'EMNIST ByClass',
            'description': 'Full Alphanumeric',
            'num_classes': 62,
            'labels': '0-9, A-Z, a-z',
            'train_samples': 697932,
            'test_samples': 116323
        }
    }
    return info.get(dataset_type, info['mnist'])


def visualize_dataset_samples(
    dataset_type: DatasetType = 'mnist',
    num_samples: int = 20
) -> None:
    """
    Visualize random samples from a dataset.
    
    Args:
        dataset_type: Type of dataset
        num_samples: Number of samples to display
    """
    import matplotlib.pyplot as plt
    from model_extended import get_class_info, index_to_label
    
    _, test_transform = get_data_transforms(augment=False)
    
    if dataset_type == 'mnist':
        dataset = datasets.MNIST('./data', train=True, download=True, transform=test_transform)
    elif dataset_type == 'letters':
        dataset = datasets.EMNIST('./data', split='letters', train=True, download=True, transform=test_transform)
        dataset.targets = dataset.targets - 1
    elif dataset_type == 'balanced':
        dataset = datasets.EMNIST('./data', split='balanced', train=True, download=True, transform=test_transform)
    else:
        dataset = datasets.EMNIST('./data', split='byclass', train=True, download=True, transform=test_transform)
    
    # Random indices
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    cols = 5
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 2.5 * rows))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        image, label = dataset[idx]
        img = image.squeeze().numpy()
        img = img * 0.3081 + 0.1307  # Denormalize
        
        char_label = index_to_label(label, dataset_type)
        
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'{char_label} ({label})', fontsize=10)
        axes[i].axis('off')
    
    # Hide empty subplots
    for i in range(len(indices), len(axes)):
        axes[i].axis('off')
    
    info = get_dataset_info(dataset_type)
    plt.suptitle(f"{info['name']} - {info['description']}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("   DATASET INFORMATION")
    print("="*60)
    
    for dtype in ['mnist', 'letters', 'balanced', 'byclass']:
        info = get_dataset_info(dtype)
        print(f"\n📊 {info['name']}:")
        print(f"   Description: {info['description']}")
        print(f"   Classes: {info['num_classes']} ({info['labels']})")
        print(f"   Train: {info['train_samples']:,} | Test: {info['test_samples']:,}")
