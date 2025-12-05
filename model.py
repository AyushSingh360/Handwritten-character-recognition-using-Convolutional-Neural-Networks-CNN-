"""
CNN Model Architecture for MNIST Digit Recognition

This module defines a Convolutional Neural Network for classifying 
handwritten digits (0-9) from the MNIST dataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTNet(nn.Module):
    """
    A modern CNN architecture for MNIST digit classification.
    
    Architecture:
        - 2 Convolutional blocks with BatchNorm and Dropout
        - 2 Fully connected layers
        - Achieves ~99% accuracy on MNIST
    
    Input: (batch_size, 1, 28, 28) - Grayscale 28x28 images
    Output: (batch_size, 10) - Logits for each digit class
    """
    
    def __init__(self, dropout_rate: float = 0.25):
        """
        Initialize the CNN model.
        
        Args:
            dropout_rate: Dropout probability for regularization
        """
        super(MNISTNet, self).__init__()
        
        # First Convolutional Block
        # Input: (1, 28, 28) -> Output: (32, 13, 13)
        self.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=32, 
            kernel_size=3, 
            stride=1, 
            padding=1
        )
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(p=dropout_rate)
        
        # Second Convolutional Block
        # Input: (32, 14, 14) -> Output: (64, 7, 7)
        self.conv2 = nn.Conv2d(
            in_channels=32, 
            out_channels=64, 
            kernel_size=3, 
            stride=1, 
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout2d(p=dropout_rate)
        
        # Third Convolutional Block
        # Input: (64, 7, 7) -> Output: (128, 3, 3)
        self.conv3 = nn.Conv2d(
            in_channels=64, 
            out_channels=128, 
            kernel_size=3, 
            stride=1, 
            padding=1
        )
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout2d(p=dropout_rate)
        
        # Fully Connected Layers
        # After 3 pooling layers: 28 -> 14 -> 7 -> 3
        # Flattened size: 128 * 3 * 3 = 1152
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.dropout_fc = nn.Dropout(p=0.5)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        
        self.fc3 = nn.Linear(128, 10)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)
            
        Returns:
            Output tensor of shape (batch_size, 10) containing logits
        """
        # First Conv Block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Second Conv Block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Third Conv Block
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully Connected Layers
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout_fc(x)
        
        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = F.relu(x)
        
        x = self.fc3(x)
        
        return x
    
    def predict(self, x: torch.Tensor) -> tuple:
        """
        Make predictions with probabilities.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)
            
        Returns:
            Tuple of (predicted_classes, probabilities)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1)
        return predicted_classes, probabilities


def get_model_summary(model: nn.Module) -> str:
    """
    Get a string summary of the model architecture.
    
    Args:
        model: PyTorch model
        
    Returns:
        String containing model summary
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    summary = f"""
╔══════════════════════════════════════════════════════════╗
║              MNIST CNN Model Summary                     ║
╠══════════════════════════════════════════════════════════╣
║  Total Parameters:      {total_params:>10,}                      ║
║  Trainable Parameters:  {trainable_params:>10,}                      ║
║  Model Size:            {total_params * 4 / 1024 / 1024:>10.2f} MB                   ║
╚══════════════════════════════════════════════════════════╝
"""
    return summary


if __name__ == "__main__":
    # Test the model
    model = MNISTNet()
    print(get_model_summary(model))
    
    # Test forward pass
    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
