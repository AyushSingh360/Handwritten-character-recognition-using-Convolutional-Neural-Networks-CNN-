"""
Prediction Module for MNIST Digit Recognition

This module provides functions for loading a trained model
and making predictions on new images.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from typing import Tuple, Optional, Union
import matplotlib.pyplot as plt

from model import MNISTNet
from utils import get_device


class MNISTPredictor:
    """
    Predictor class for MNIST digit classification.
    
    Handles model loading, image preprocessing, and inference.
    """
    
    def __init__(
        self, 
        model_path: str = './models/mnist_cnn.pth',
        device: Optional[torch.device] = None
    ):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to the trained model checkpoint
            device: Device to use for inference
        """
        self.device = device if device else get_device()
        self.model = self._load_model(model_path)
        self.transform = self._get_transform()
        
    def _load_model(self, model_path: str) -> MNISTNet:
        """Load the trained model from checkpoint."""
        if not os.path.exists(model_path):
            # Try best model
            best_model_path = model_path.replace('.pth', '_best.pth')
            if os.path.exists(best_model_path):
                model_path = best_model_path
            else:
                raise FileNotFoundError(
                    f"Model not found at {model_path}. "
                    "Please train the model first using 'python train.py'"
                )
        
        model = MNISTNet()
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Loaded model from {model_path}")
            if 'val_acc' in checkpoint:
                print(f"  Model validation accuracy: {checkpoint['val_acc']:.2f}%")
        else:
            model.load_state_dict(checkpoint)
            print(f"✓ Loaded model from {model_path}")
        
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def _get_transform(self) -> transforms.Compose:
        """Get preprocessing transforms for inference."""
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    def preprocess_image(
        self, 
        image: Union[str, Image.Image, np.ndarray]
    ) -> torch.Tensor:
        """
        Preprocess an image for inference.
        
        Args:
            image: Image path, PIL Image, or numpy array
            
        Returns:
            Preprocessed tensor ready for inference
        """
        # Load image if path is provided
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            # Handle different array formats
            if image.ndim == 2:
                image = Image.fromarray(image.astype('uint8'), mode='L')
            elif image.ndim == 3:
                if image.shape[2] == 4:  # RGBA
                    image = Image.fromarray(image.astype('uint8'), mode='RGBA').convert('L')
                else:  # RGB
                    image = Image.fromarray(image.astype('uint8'), mode='RGB').convert('L')
            else:
                raise ValueError(f"Unexpected array shape: {image.shape}")
        
        # Convert to grayscale if needed
        if image.mode != 'L':
            image = image.convert('L')
        
        # Apply transforms
        tensor = self.transform(image)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def predict(
        self, 
        image: Union[str, Image.Image, np.ndarray, torch.Tensor]
    ) -> Tuple[int, float, np.ndarray]:
        """
        Make a prediction on an image.
        
        Args:
            image: Input image (path, PIL Image, numpy array, or tensor)
            
        Returns:
            Tuple of (predicted_digit, confidence, all_probabilities)
        """
        # Preprocess if not already a tensor
        if isinstance(image, torch.Tensor):
            tensor = image
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
        else:
            tensor = self.preprocess_image(image)
        
        tensor = tensor.to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = F.softmax(outputs, dim=1)
            
            confidence, predicted = probabilities.max(1)
            
            predicted_digit = predicted.item()
            confidence_score = confidence.item()
            all_probs = probabilities.squeeze().cpu().numpy()
        
        return predicted_digit, confidence_score, all_probs
    
    def predict_batch(
        self, 
        images: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions on a batch of images.
        
        Args:
            images: Batch of images tensor (N, 1, 28, 28)
            
        Returns:
            Tuple of (predicted_digits, confidences, all_probabilities)
        """
        images = images.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(images)
            probabilities = F.softmax(outputs, dim=1)
            
            confidences, predictions = probabilities.max(1)
            
            predicted_digits = predictions.cpu().numpy()
            confidence_scores = confidences.cpu().numpy()
            all_probs = probabilities.cpu().numpy()
        
        return predicted_digits, confidence_scores, all_probs


def visualize_prediction(
    image: Union[str, Image.Image, np.ndarray],
    predictor: MNISTPredictor
) -> None:
    """
    Visualize a prediction with probability bar chart.
    
    Args:
        image: Input image
        predictor: MNISTPredictor instance
    """
    # Make prediction
    predicted, confidence, probs = predictor.predict(image)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Show image
    ax1 = axes[0]
    if isinstance(image, str):
        img = Image.open(image).convert('L')
    elif isinstance(image, np.ndarray):
        img = image
    else:
        img = np.array(image.convert('L'))
    
    ax1.imshow(img, cmap='gray')
    ax1.set_title(
        f'Predicted: {predicted} (Confidence: {confidence*100:.1f}%)',
        fontsize=14, fontweight='bold'
    )
    ax1.axis('off')
    
    # Show probability bar chart
    ax2 = axes[1]
    colors = ['#3498db'] * 10
    colors[predicted] = '#2ecc71'  # Highlight predicted class
    
    bars = ax2.bar(range(10), probs * 100, color=colors, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Digit', fontsize=12)
    ax2.set_ylabel('Probability (%)', fontsize=12)
    ax2.set_title('Prediction Probabilities', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(10))
    ax2.set_ylim(0, 100)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, prob in zip(bars, probs):
        if prob > 0.05:  # Only label significant probabilities
            ax2.text(
                bar.get_x() + bar.get_width()/2, 
                bar.get_height() + 1,
                f'{prob*100:.1f}%',
                ha='center', va='bottom', fontsize=9
            )
    
    plt.tight_layout()
    plt.show()


def main():
    """Demo the prediction module with test samples."""
    from utils import load_mnist_data
    
    print("\n" + "="*60)
    print("   🔢 MNIST DIGIT PREDICTION DEMO")
    print("="*60)
    
    # Load predictor
    try:
        predictor = MNISTPredictor()
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        return
    
    # Load test data
    print("\n📂 Loading test samples...")
    _, test_loader = load_mnist_data(batch_size=16)
    
    # Get a batch of test images
    images, labels = next(iter(test_loader))
    
    # Make predictions
    print("\n🔮 Making predictions on test samples...")
    predictions, confidences, _ = predictor.predict_batch(images)
    
    # Display results
    print("\n" + "-"*50)
    print(f"{'Sample':^8} | {'True':^6} | {'Pred':^6} | {'Conf':^10} | {'Status':^8}")
    print("-"*50)
    
    correct = 0
    for i in range(len(labels)):
        true_label = labels[i].item()
        pred_label = predictions[i]
        conf = confidences[i] * 100
        
        is_correct = true_label == pred_label
        correct += is_correct
        status = "✓" if is_correct else "✗"
        
        print(f"{i+1:^8} | {true_label:^6} | {pred_label:^6} | {conf:>8.2f}% | {status:^8}")
    
    print("-"*50)
    print(f"Accuracy: {correct}/{len(labels)} ({100*correct/len(labels):.1f}%)")
    print("="*60)
    
    # Visualize first sample
    print("\n📊 Visualizing first sample prediction...")
    visualize_prediction(images[0], predictor)


if __name__ == "__main__":
    main()
