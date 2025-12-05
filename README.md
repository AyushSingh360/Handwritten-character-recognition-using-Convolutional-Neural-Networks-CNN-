# 🔢 MNIST Handwritten Digit Recognition

A classic "Hello World" deep learning project that trains a **Convolutional Neural Network (CNN)** to classify handwritten digits (0-9) using the MNIST dataset.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## ✨ Features

- 🧠 **Modern CNN Architecture** - BatchNorm, Dropout, and 3 convolutional layers
- 📊 **Training Pipeline** - Complete with validation, metrics, and model checkpointing
- 🎨 **Interactive Demo** - Draw digits and get real-time predictions via Gradio
- 📈 **Visualizations** - Training curves, confusion matrix, and sample predictions
- 🚀 **High Accuracy** - Achieves ~99% accuracy on MNIST test set

## 🏗️ Project Structure

```
CNN/
├── requirements.txt    # Project dependencies
├── README.md           # This file
├── model.py            # CNN architecture definition
├── train.py            # Training script
├── predict.py          # Inference/prediction utilities
├── utils.py            # Helper functions
├── demo.py             # Gradio interactive demo
├── models/             # Saved model checkpoints
│   ├── mnist_cnn.pth
│   └── mnist_cnn_best.pth
└── outputs/            # Training visualizations
    ├── training_curves.png
    ├── confusion_matrix.png
    └── sample_predictions.png
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train.py
```

This will:
- Download the MNIST dataset automatically
- Train the CNN for 10 epochs
- Save the best model to `models/mnist_cnn_best.pth`
- Generate visualizations in the `outputs/` folder

**Training Options:**
```bash
python train.py --epochs 20 --batch-size 128 --lr 0.0005
```

### 3. Launch Interactive Demo

```bash
python demo.py
```

Open your browser to `http://127.0.0.1:7860` and draw digits!

### 4. Test Predictions

```bash
python predict.py
```

## 🧠 Model Architecture

```
Input (1, 28, 28) - Grayscale 28×28 image
    │
    ▼
┌─────────────────────────────────────────┐
│  Conv Block 1                           │
│  Conv2D(1→32, 3×3) → BatchNorm → ReLU   │
│  MaxPool(2×2) → Dropout(0.25)           │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Conv Block 2                           │
│  Conv2D(32→64, 3×3) → BatchNorm → ReLU  │
│  MaxPool(2×2) → Dropout(0.25)           │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Conv Block 3                           │
│  Conv2D(64→128, 3×3) → BatchNorm → ReLU │
│  MaxPool(2×2) → Dropout(0.25)           │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Fully Connected Layers                 │
│  Flatten → FC(1152→256) → ReLU          │
│  Dropout(0.5) → FC(256→128) → ReLU      │
│  FC(128→10)                             │
└─────────────────────────────────────────┘
    │
    ▼
Output (10) - Logits for each digit class
```

**Model Statistics:**
- Parameters: ~300,000
- Model Size: ~1.2 MB
- Inference Time: <1ms on GPU, ~5ms on CPU

## 📊 Results

| Metric | Value |
|--------|-------|
| Training Accuracy | ~99.5% |
| Validation Accuracy | ~99.0% |
| Training Time (GPU) | ~2-3 minutes |
| Training Time (CPU) | ~15-20 minutes |

### Sample Outputs

After training, you'll find these visualizations in the `outputs/` folder:

- **Training Curves** - Loss and accuracy over epochs
- **Confusion Matrix** - Classification performance per digit
- **Sample Predictions** - Visual examples with confidence scores

## 📁 Dataset

The [MNIST dataset](http://yann.lecun.com/exdb/mnist/) contains:
- **60,000** training images
- **10,000** test images
- **28×28** grayscale images
- **10 classes** (digits 0-9)

The dataset is downloaded automatically when you run `train.py`.

## 🔧 Configuration

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 10 | Number of training epochs |
| `--batch-size` | 64 | Batch size for training |
| `--lr` | 0.001 | Learning rate |
| `--no-cuda` | False | Disable GPU acceleration |

### Data Augmentation

The training pipeline includes:
- Random rotation (±10°)
- Random translation (±10%)
- Random scaling (90-110%)

## 📚 API Usage

### Using the Predictor

```python
from predict import MNISTPredictor
from PIL import Image

# Load the predictor
predictor = MNISTPredictor()

# Predict from image file
digit, confidence, probs = predictor.predict("digit.png")
print(f"Predicted: {digit} (Confidence: {confidence:.2%})")

# Predict from PIL Image
image = Image.open("digit.png")
digit, confidence, probs = predictor.predict(image)

# Predict from numpy array
import numpy as np
array = np.array(image)
digit, confidence, probs = predictor.predict(array)
```

### Using the Model Directly

```python
import torch
from model import MNISTNet

# Create model
model = MNISTNet()

# Load trained weights
checkpoint = torch.load('./models/mnist_cnn.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Make prediction
input_tensor = torch.randn(1, 1, 28, 28)
output = model(input_tensor)
predicted = torch.argmax(output, dim=1)
```

## 🛠️ Development

### Prerequisites

- Python 3.8+
- CUDA (optional, for GPU acceleration)

### Running Tests

```bash
# Test model architecture
python model.py

# Test prediction module
python predict.py
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/) by Yann LeCun
- [PyTorch](https://pytorch.org/) for the deep learning framework
- [Gradio](https://gradio.app/) for the interactive web interface

---

<p align="center">
  Made with ❤️ for learning deep learning
</p>
