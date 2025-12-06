"""
Extended Model Architecture for Alphanumeric Recognition

Supports:
- MNIST: Digits only (0-9) - 10 classes
- EMNIST Letters: Letters only (A-Z) - 26 classes
- EMNIST Balanced: Alphanumeric (0-9 + A-Z + a-z merged) - 47 classes
- EMNIST ByClass: Full alphanumeric (0-9 + A-Z + a-z) - 62 classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal

# Dataset type definitions
DatasetType = Literal['mnist', 'letters', 'balanced', 'byclass']

# Class mappings for different datasets
CLASS_MAPPINGS = {
    'mnist': {
        'num_classes': 10,
        'labels': list('0123456789'),
        'description': 'Digits (0-9)'
    },
    'letters': {
        'num_classes': 26,
        'labels': list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'),
        'description': 'Letters (A-Z)'
    },
    'balanced': {
        'num_classes': 47,
        'labels': list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt'),
        'description': 'Balanced Alphanumeric (47 classes)'
    },
    'byclass': {
        'num_classes': 62,
        'labels': list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'),
        'description': 'Full Alphanumeric (62 classes)'
    }
}


def get_class_info(dataset_type: DatasetType) -> dict:
    """Get class information for a dataset type."""
    return CLASS_MAPPINGS.get(dataset_type, CLASS_MAPPINGS['mnist'])


def index_to_label(index: int, dataset_type: DatasetType) -> str:
    """Convert class index to human-readable label."""
    info = get_class_info(dataset_type)
    if 0 <= index < len(info['labels']):
        return info['labels'][index]
    return f'Unknown({index})'


class AlphanumericNet(nn.Module):
    """
    Extended CNN architecture for alphanumeric character recognition.
    
    Supports variable number of output classes for different datasets:
    - 10 classes: MNIST digits
    - 26 classes: EMNIST letters
    - 47 classes: EMNIST balanced
    - 62 classes: EMNIST byclass
    
    Architecture is deeper than MNISTNet to handle more complex classification.
    """
    
    def __init__(
        self, 
        num_classes: int = 10,
        dropout_rate: float = 0.25,
        dataset_type: DatasetType = 'mnist'
    ):
        """
        Initialize the CNN model.
        
        Args:
            num_classes: Number of output classes
            dropout_rate: Dropout probability for regularization
            dataset_type: Type of dataset ('mnist', 'letters', 'balanced', 'byclass')
        """
        super(AlphanumericNet, self).__init__()
        
        self.num_classes = num_classes
        self.dataset_type = dataset_type
        self.class_info = get_class_info(dataset_type)
        
        # First Convolutional Block
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(p=dropout_rate)
        
        # Second Convolutional Block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout2d(p=dropout_rate)
        
        # Third Convolutional Block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout2d(p=dropout_rate)
        
        # Fourth Convolutional Block (additional depth for alphanumeric)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.dropout4 = nn.Dropout2d(p=dropout_rate)
        
        # Adaptive pooling to ensure consistent output size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))
        
        # Fully Connected Layers
        # Adaptive pool gives us 256 * 2 * 2 = 1024
        self.fc1 = nn.Linear(256 * 2 * 2, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.dropout_fc1 = nn.Dropout(p=0.5)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.dropout_fc2 = nn.Dropout(p=0.3)
        
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        # Conv Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Conv Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Conv Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Conv Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout4(x)
        
        # Adaptive pooling
        x = self.adaptive_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC Layers
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout_fc1(x)
        
        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = F.relu(x)
        x = self.dropout_fc2(x)
        
        x = self.fc3(x)
        
        return x
    
    def predict(self, x: torch.Tensor) -> tuple:
        """Make predictions with probabilities and labels."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
            predicted_indices = torch.argmax(probabilities, dim=1)
            
            # Convert indices to labels
            labels = [index_to_label(idx.item(), self.dataset_type) 
                     for idx in predicted_indices]
            
        return predicted_indices, probabilities, labels


def get_model_summary(model: nn.Module, dataset_type: str = 'mnist') -> str:
    """Get a string summary of the model architecture."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    info = get_class_info(dataset_type)
    
    summary = f"""
╔══════════════════════════════════════════════════════════════╗
║           Alphanumeric CNN Model Summary                     ║
╠══════════════════════════════════════════════════════════════╣
║  Dataset Type:          {dataset_type:>15}                   ║
║  Description:           {info['description']:>30}           ║
║  Number of Classes:     {info['num_classes']:>10}                           ║
║  Total Parameters:      {total_params:>10,}                           ║
║  Trainable Parameters:  {trainable_params:>10,}                           ║
║  Model Size:            {total_params * 4 / 1024 / 1024:>10.2f} MB                        ║
╚══════════════════════════════════════════════════════════════╝
"""
    return summary


def create_model(dataset_type: DatasetType = 'mnist') -> AlphanumericNet:
    """
    Factory function to create model for specific dataset.
    
    Args:
        dataset_type: 'mnist', 'letters', 'balanced', or 'byclass'
        
    Returns:
        AlphanumericNet configured for the dataset
    """
    info = get_class_info(dataset_type)
    return AlphanumericNet(
        num_classes=info['num_classes'],
        dataset_type=dataset_type
    )


# Keep backward compatibility
MNISTNet = lambda: AlphanumericNet(num_classes=10, dataset_type='mnist')


if __name__ == "__main__":
    print("\n" + "="*60)
    print("   ALPHANUMERIC CNN MODEL TEST")
    print("="*60)
    
    # Test each model type
    for dtype in ['mnist', 'letters', 'balanced', 'byclass']:
        model = create_model(dtype)
        model.eval()  # Set to eval mode for testing with batch size 1
        print(get_model_summary(model, dtype))
        
        # Test forward pass
        dummy_input = torch.randn(1, 1, 28, 28)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        print()

