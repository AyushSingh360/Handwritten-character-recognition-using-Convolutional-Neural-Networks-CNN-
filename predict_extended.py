"""
Extended Prediction Module for Alphanumeric Recognition

Supports:
- MNIST: Digits (0-9)
- EMNIST Letters: Letters (A-Z)
- EMNIST Balanced: Alphanumeric (47 classes)
- EMNIST ByClass: Full Alphanumeric (62 classes)
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from typing import Tuple, Optional, Union, Literal

from model_extended import AlphanumericNet, get_class_info, index_to_label, create_model
from utils import get_device

DatasetType = Literal['mnist', 'letters', 'balanced', 'byclass']


class AlphanumericPredictor:
    """
    Predictor class for alphanumeric character classification.
    
    Handles model loading, image preprocessing, and inference
    for digits, letters, or full alphanumeric recognition.
    """
    
    def __init__(
        self, 
        model_path: Optional[str] = None,
        dataset_type: DatasetType = 'mnist',
        device: Optional[torch.device] = None
    ):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to the trained model checkpoint (auto-detect if None)
            dataset_type: Type of model to load ('mnist', 'letters', 'balanced', 'byclass')
            device: Device to use for inference
        """
        self.device = device if device else get_device()
        self.dataset_type = dataset_type
        self.class_info = get_class_info(dataset_type)
        
        # Auto-detect model path
        if model_path is None:
            model_path = self._find_model_path(dataset_type)
        
        self.model = self._load_model(model_path)
        self.transform = self._get_transform()
        
    def _find_model_path(self, dataset_type: str) -> str:
        """Find model path for the given dataset type."""
        possible_paths = [
            f'./models/{dataset_type}_model.pth',
            f'./models/{dataset_type}_model_best.pth',
            './models/mnist_cnn.pth',  # Fallback for backward compatibility
            './models/mnist_cnn_best.pth'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        raise FileNotFoundError(
            f"No model found for {dataset_type}. "
            f"Please train first using: python train_extended.py --dataset {dataset_type}"
        )
        
    def _load_model(self, model_path: str) -> AlphanumericNet:
        """Load the trained model from checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Determine model configuration
        if 'dataset_type' in checkpoint:
            self.dataset_type = checkpoint['dataset_type']
            num_classes = checkpoint.get('num_classes', get_class_info(self.dataset_type)['num_classes'])
        elif 'num_classes' in checkpoint:
            num_classes = checkpoint['num_classes']
        else:
            # Assume MNIST for backward compatibility
            num_classes = 10
            self.dataset_type = 'mnist'
        
        self.class_info = get_class_info(self.dataset_type)
        
        # Create model with correct architecture
        model = AlphanumericNet(num_classes=num_classes, dataset_type=self.dataset_type)
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            acc = checkpoint.get('val_acc', 'N/A')
            print(f"✓ Loaded {self.dataset_type} model from {model_path}")
            if acc != 'N/A':
                print(f"  Model accuracy: {acc:.2f}%")
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
        """Preprocess an image for inference."""
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            if image.ndim == 2:
                image = Image.fromarray(image.astype('uint8'), mode='L')
            elif image.ndim == 3:
                if image.shape[2] == 4:
                    image = Image.fromarray(image.astype('uint8'), mode='RGBA').convert('L')
                else:
                    image = Image.fromarray(image.astype('uint8'), mode='RGB').convert('L')
        
        if hasattr(image, 'mode') and image.mode != 'L':
            image = image.convert('L')
        
        tensor = self.transform(image)
        tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def predict(
        self, 
        image: Union[str, Image.Image, np.ndarray, torch.Tensor]
    ) -> Tuple[str, int, float, np.ndarray]:
        """
        Make a prediction on an image.
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (predicted_label, predicted_index, confidence, all_probabilities)
        """
        if isinstance(image, torch.Tensor):
            tensor = image
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
        else:
            tensor = self.preprocess_image(image)
        
        tensor = tensor.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = F.softmax(outputs, dim=1)
            
            confidence, predicted_idx = probabilities.max(1)
            
            idx = predicted_idx.item()
            label = index_to_label(idx, self.dataset_type)
            conf = confidence.item()
            probs = probabilities.squeeze().cpu().numpy()
        
        return label, idx, conf, probs
    
    def predict_batch(
        self, 
        images: torch.Tensor
    ) -> Tuple[list, np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions on a batch of images.
        
        Args:
            images: Batch of images tensor (N, 1, 28, 28)
            
        Returns:
            Tuple of (predicted_labels, predicted_indices, confidences, all_probabilities)
        """
        images = images.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(images)
            probabilities = F.softmax(outputs, dim=1)
            
            confidences, predictions = probabilities.max(1)
            
            indices = predictions.cpu().numpy()
            labels = [index_to_label(idx, self.dataset_type) for idx in indices]
            confs = confidences.cpu().numpy()
            probs = probabilities.cpu().numpy()
        
        return labels, indices, confs, probs
    
    def get_top_predictions(
        self, 
        image: Union[str, Image.Image, np.ndarray, torch.Tensor],
        top_k: int = 5
    ) -> list:
        """
        Get top-k predictions with labels and confidences.
        
        Args:
            image: Input image
            top_k: Number of top predictions to return
            
        Returns:
            List of (label, confidence) tuples
        """
        _, _, _, probs = self.predict(image)
        
        top_indices = np.argsort(probs)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            label = index_to_label(idx, self.dataset_type)
            conf = probs[idx]
            results.append((label, float(conf)))
        
        return results


# Backward compatibility
MNISTPredictor = lambda path='./models/mnist_cnn.pth': AlphanumericPredictor(path, 'mnist')


def main():
    """Demo the prediction module."""
    from data_loader import load_dataset
    
    print("\n" + "="*60)
    print("   🔤 ALPHANUMERIC PREDICTION DEMO")
    print("="*60)
    
    # Try to load available models
    available_models = []
    for dtype in ['mnist', 'letters', 'balanced', 'byclass']:
        paths = [f'./models/{dtype}_model.pth', f'./models/{dtype}_model_best.pth']
        for p in paths:
            if os.path.exists(p):
                available_models.append(dtype)
                break
    
    # Check for backward compatible MNIST
    if not available_models and os.path.exists('./models/mnist_cnn.pth'):
        available_models.append('mnist')
    
    if not available_models:
        print("\n❌ No trained models found!")
        print("   Train a model first using:")
        print("   python train_extended.py --dataset mnist")
        print("   python train_extended.py --dataset letters")
        print("   python train_extended.py --dataset balanced")
        return
    
    print(f"\n📦 Available models: {', '.join(available_models)}")
    
    # Test first available model
    dtype = available_models[0]
    print(f"\n🔍 Testing {dtype} model...")
    
    try:
        predictor = AlphanumericPredictor(dataset_type=dtype)
        
        # Load test data
        _, test_loader, _ = load_dataset(dtype, batch_size=16)
        images, labels = next(iter(test_loader))
        
        # Make predictions
        pred_labels, pred_indices, confidences, _ = predictor.predict_batch(images)
        
        # Display results
        print("\n" + "-"*60)
        print(f"{'#':^4} | {'True':^8} | {'Pred':^8} | {'Conf':^10} | {'Status':^8}")
        print("-"*60)
        
        correct = 0
        for i in range(len(labels)):
            true_label = index_to_label(labels[i].item(), dtype)
            pred_label = pred_labels[i]
            conf = confidences[i] * 100
            
            is_correct = pred_indices[i] == labels[i].item()
            correct += is_correct
            status = "✓" if is_correct else "✗"
            
            print(f"{i+1:^4} | {true_label:^8} | {pred_label:^8} | {conf:>8.2f}% | {status:^8}")
        
        print("-"*60)
        print(f"Accuracy: {correct}/{len(labels)} ({100*correct/len(labels):.1f}%)")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
