"""
Training Script for MNIST Digit Recognition

This script handles the complete training pipeline including:
- Data loading and preprocessing
- Model training with validation
- Progress tracking and metrics
- Model checkpointing
- Visualization generation
"""

import os
import time
import argparse
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from typing import Tuple, Dict, List

from model import MNISTNet, get_model_summary
from utils import (
    load_mnist_data,
    get_device,
    plot_training_history,
    plot_confusion_matrix,
    plot_sample_predictions,
    print_classification_report
)


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """
    Train for one epoch.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use
        
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc='Training', leave=False)
    
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, List[int], List[int], torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Validate the model.
    
    Args:
        model: Neural network model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to use
        
    Returns:
        Tuple of (average_loss, accuracy, true_labels, pred_labels, 
                  sample_images, sample_probs)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_true = []
    all_pred = []
    all_probs = []
    sample_images = None
    sample_true = None
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validating', leave=False):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Store for analysis
            all_true.extend(labels.cpu().numpy())
            all_pred.extend(predicted.cpu().numpy())
            all_probs.append(probs.cpu())
            
            # Save first batch for visualization
            if sample_images is None:
                sample_images = images.cpu()
                sample_true = labels.cpu()
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    all_probs = torch.cat(all_probs, dim=0)
    
    return (epoch_loss, epoch_acc, all_true, all_pred, 
            sample_images, sample_true, all_probs[:len(sample_images)])


def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    epochs: int = 10,
    learning_rate: float = 0.001,
    device: torch.device = None,
    save_dir: str = './models',
    output_dir: str = './outputs'
) -> Dict:
    """
    Complete training loop.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to use
        save_dir: Directory to save model checkpoints
        output_dir: Directory to save visualizations
        
    Returns:
        Dictionary containing training history
    """
    if device is None:
        device = get_device()
    
    model = model.to(device)
    
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
    print("🚀 STARTING TRAINING")
    print("="*60)
    print(f"  Epochs:        {epochs}")
    print(f"  Batch Size:    {train_loader.batch_size}")
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
        val_loss, val_acc, y_true, y_pred, sample_imgs, sample_labels, sample_probs = validate(
            model, val_loader, criterion, device
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch results
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = os.path.join(save_dir, 'mnist_cnn_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
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
    final_model_path = os.path.join(save_dir, 'mnist_cnn.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'val_acc': val_acc
    }, final_model_path)
    print(f"\n✓ Saved final model to {final_model_path}")
    
    # Generate visualizations
    print("\n📈 Generating visualizations...")
    
    # Training curves
    plot_training_history(
        history, 
        save_path=os.path.join(output_dir, 'training_curves.png')
    )
    
    # Confusion matrix
    plot_confusion_matrix(
        y_true, y_pred,
        save_path=os.path.join(output_dir, 'confusion_matrix.png')
    )
    
    # Sample predictions
    with torch.no_grad():
        model.eval()
        sample_imgs_device = sample_imgs.to(device)
        sample_outputs = model(sample_imgs_device)
        sample_preds = torch.argmax(sample_outputs, dim=1).cpu()
        sample_probs = torch.softmax(sample_outputs, dim=1).cpu()
    
    plot_sample_predictions(
        sample_imgs, sample_labels, sample_preds, sample_probs,
        num_samples=16,
        save_path=os.path.join(output_dir, 'sample_predictions.png')
    )
    
    # Classification report
    print_classification_report(y_true, y_pred)
    
    return history


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(
        description='Train CNN for MNIST digit classification'
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
    print("   🔢 MNIST HANDWRITTEN DIGIT RECOGNITION")
    print("   Convolutional Neural Network Training")
    print("="*60)
    
    # Get device
    if args.no_cuda:
        device = torch.device('cpu')
        print("Using CPU (CUDA disabled)")
    else:
        device = get_device()
    
    # Load data
    print("\n📂 Loading MNIST dataset...")
    train_loader, test_loader = load_mnist_data(batch_size=args.batch_size)
    
    # Create model
    print("\n🏗️ Building CNN model...")
    model = MNISTNet()
    print(get_model_summary(model))
    
    # Train
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=args.epochs,
        learning_rate=args.lr,
        device=device
    )
    
    print("\n✅ All done! Model saved to ./models/mnist_cnn.pth")
    print("   Run 'python demo.py' to launch the interactive demo!")


if __name__ == "__main__":
    main()
