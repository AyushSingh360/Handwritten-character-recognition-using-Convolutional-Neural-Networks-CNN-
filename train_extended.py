"""
Unified Training Script for MNIST/EMNIST Recognition

Supports training models for:
- Digits only (MNIST): 0-9
- Letters only (EMNIST Letters): A-Z
- Alphanumeric (EMNIST Balanced): 47 classes
- Full Alphanumeric (EMNIST ByClass): 62 classes
"""

import os
import time
import argparse
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from typing import Tuple, Dict

from model_extended import AlphanumericNet, create_model, get_model_summary, get_class_info
from data_loader import load_dataset, get_dataset_info
from utils import (
    get_device,
    plot_training_history,
    plot_confusion_matrix,
    print_classification_report
)


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc='Training', leave=False)
    
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })
    
    return running_loss / total, 100. * correct / total


def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, list, list]:
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_true = []
    all_pred = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validating', leave=False):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_true.extend(labels.cpu().numpy())
            all_pred.extend(predicted.cpu().numpy())
    
    return running_loss / total, 100. * correct / total, all_true, all_pred


def train(
    dataset_type: str = 'mnist',
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    device: torch.device = None,
    save_dir: str = './models',
    output_dir: str = './outputs'
) -> Dict:
    """
    Complete training loop.
    
    Args:
        dataset_type: 'mnist', 'letters', 'balanced', or 'byclass'
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to use
        save_dir: Directory to save models
        output_dir: Directory to save visualizations
        
    Returns:
        Training history
    """
    if device is None:
        device = get_device()
    
    # Load dataset
    train_loader, test_loader, _ = load_dataset(
        dataset_type=dataset_type,
        batch_size=batch_size
    )
    
    # Create model
    model = create_model(dataset_type)
    model = model.to(device)
    
    class_info = get_class_info(dataset_type)
    print(get_model_summary(model, dataset_type))
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print(f"🚀 TRAINING {dataset_type.upper()} MODEL")
    print("="*60)
    print(f"  Dataset:       {class_info['description']}")
    print(f"  Classes:       {class_info['num_classes']}")
    print(f"  Epochs:        {epochs}")
    print(f"  Batch Size:    {batch_size}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Device:        {device}")
    print("="*60 + "\n")
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        print(f"\n📊 Epoch {epoch}/{epochs}")
        print("-" * 40)
        
        # Training
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validation
        val_loss, val_acc, y_true, y_pred = validate(
            model, test_loader, criterion, device
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_name = f'{dataset_type}_model_best.pth'
            model_path = os.path.join(save_dir, model_name)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'dataset_type': dataset_type,
                'num_classes': class_info['num_classes']
            }, model_path)
            print(f"  ✓ Saved best model (accuracy: {val_acc:.2f}%)")
    
    elapsed_time = time.time() - start_time
    
    # Final results
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETE!")
    print("="*60)
    print(f"  Total Time:     {elapsed_time/60:.2f} minutes")
    print(f"  Best Val Acc:   {best_val_acc:.2f}%")
    print(f"  Final Val Acc:  {val_acc:.2f}%")
    print("="*60)
    
    # Save final model
    model_name = f'{dataset_type}_model.pth'
    final_model_path = os.path.join(save_dir, model_name)
    torch.save({
        'model_state_dict': model.state_dict(),
        'val_acc': val_acc,
        'dataset_type': dataset_type,
        'num_classes': class_info['num_classes']
    }, final_model_path)
    print(f"\n✓ Saved final model to {final_model_path}")
    
    # Generate visualizations
    print("\n📈 Generating visualizations...")
    
    plot_training_history(
        history, 
        save_path=os.path.join(output_dir, f'{dataset_type}_training_curves.png')
    )
    
    plot_confusion_matrix(
        y_true, y_pred,
        save_path=os.path.join(output_dir, f'{dataset_type}_confusion_matrix.png')
    )
    
    return history


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(
        description='Train CNN for character recognition'
    )
    parser.add_argument(
        '--dataset', type=str, default='mnist',
        choices=['mnist', 'letters', 'balanced', 'byclass'],
        help='Dataset type (default: mnist)'
    )
    parser.add_argument(
        '--epochs', type=int, default=10,
        help='Number of training epochs (default: 10)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=64,
        help='Batch size for training (default: 64)'
    )
    parser.add_argument(
        '--lr', type=float, default=0.001,
        help='Learning rate (default: 0.001)'
    )
    parser.add_argument(
        '--no-cuda', action='store_true',
        help='Disable CUDA even if available'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*60)
    print("   🔤 ALPHANUMERIC CHARACTER RECOGNITION")
    print("   Convolutional Neural Network Training")
    print("="*60)
    
    # Get device
    if args.no_cuda:
        device = torch.device('cpu')
        print("Using CPU (CUDA disabled)")
    else:
        device = get_device()
    
    # Dataset info
    info = get_dataset_info(args.dataset)
    print(f"\n📊 Dataset: {info['name']}")
    print(f"   Description: {info['description']}")
    print(f"   Classes: {info['num_classes']} ({info['labels']})")
    
    # Train
    history = train(
        dataset_type=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=device
    )
    
    print(f"\n✅ All done! Model saved to ./models/{args.dataset}_model.pth")
    print("   Run 'python app.py' to launch the interactive demo!")


if __name__ == "__main__":
    main()
