# 🔢 MNIST Handwritten Digit Recognition

A complete deep learning project that trains a **Convolutional Neural Network (CNN)** to classify handwritten digits (0-9) using the MNIST dataset.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-99.3%25-green.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## ✨ Features

- 🧠 **Modern CNN Architecture** - BatchNorm, Dropout, and 3 convolutional layers
- 📊 **Complete Training Pipeline** - Validation, metrics, and model checkpointing
- 🎨 **Web Interface** - Upload images and get predictions via Gradio
- 📦 **Batch Processing** - Process folders of images, export to CSV/JSON
- 📓 **Jupyter Notebook** - Step-by-step walkthrough for learning
- 📈 **Visualizations** - Training curves, confusion matrix, sample predictions
- 🚀 **99.3% Accuracy** - Production-ready performance

## 🏗️ Project Structure

```
CNN/
├── model.py              # CNN architecture definition
├── train.py              # Training script
├── predict.py            # Inference/prediction module
├── utils.py              # Helper functions
├── app.py                # Gradio web interface
├── batch_processor.py    # Batch processing & export
├── mnist_walkthrough.ipynb  # Tutorial notebook
├── requirements.txt      # Dependencies
├── README.md             # This file
├── models/               # Saved model checkpoints
│   ├── mnist_cnn.pth
│   └── mnist_cnn_best.pth
├── outputs/              # Training visualizations
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   └── sample_predictions.png
├── predictions/          # Batch processing results
│   ├── logs/
│   └── reports/
└── data/                 # MNIST dataset (auto-downloaded)
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model (Optional - already trained)

```bash
python train.py --epochs 10
```

### 3. Launch Web Interface

```bash
python app.py
```
Open http://127.0.0.1:7860 in your browser.

### 4. Run Jupyter Notebook

```bash
jupyter notebook mnist_walkthrough.ipynb
```

### 5. Batch Process Images

```python
from batch_processor import BatchProcessor

processor = BatchProcessor()
results = processor.process_folder("./my_digits/")
processor.save_results_csv()
processor.print_summary()
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
- Model Size: ~1.7 MB
- Inference Time: <1ms on GPU, ~5ms on CPU

## 📊 Results

| Metric | Value |
|--------|-------|
| Training Accuracy | ~99.5% |
| Validation Accuracy | **99.3%** |
| Training Time (GPU) | ~2-3 minutes |
| Training Time (CPU) | ~15-20 minutes |

## 📚 Usage Examples

### Basic Prediction

```python
from predict import MNISTPredictor

predictor = MNISTPredictor()
digit, confidence, probs = predictor.predict("digit.png")
print(f"Predicted: {digit} ({confidence:.1%})")
```

### Batch Processing

```python
from batch_processor import BatchProcessor

processor = BatchProcessor()

# Process a folder
results = processor.process_folder("./digits/")

# Export results
processor.save_results_csv()
processor.save_results_json()

# Get summary
processor.print_summary()
```

### Using the Model Directly

```python
import torch
from model import MNISTNet

model = MNISTNet()
checkpoint = torch.load('./models/mnist_cnn.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Make prediction
input_tensor = torch.randn(1, 1, 28, 28)
output = model(input_tensor)
predicted = torch.argmax(output, dim=1)
```

## 🔧 Command Line Options

### Training

```bash
python train.py --epochs 20 --batch-size 128 --lr 0.0005 --no-cuda
```

| Option | Default | Description |
|--------|---------|-------------|
| `--epochs` | 10 | Number of training epochs |
| `--batch-size` | 64 | Batch size for training |
| `--lr` | 0.001 | Learning rate |
| `--no-cuda` | False | Disable GPU acceleration |

### Prediction Demo

```bash
python predict.py
```

### Web Interface

```bash
python app.py
```

## 📁 Dataset

The [MNIST dataset](http://yann.lecun.com/exdb/mnist/) contains:
- **60,000** training images
- **10,000** test images
- **28×28** grayscale images
- **10 classes** (digits 0-9)

The dataset is downloaded automatically on first run.

## 📓 Learning Resources

- **Jupyter Notebook** (`mnist_walkthrough.ipynb`): Step-by-step tutorial covering:
  1. Understanding the MNIST Dataset
  2. Exploring the CNN Architecture
  3. Training the Model
  4. Making Predictions
  5. Visualizing Results
  6. Analyzing Misclassifications

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

# Test batch processor
python batch_processor.py
```

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/) by Yann LeCun
- [PyTorch](https://pytorch.org/) for the deep learning framework
- [Gradio](https://gradio.app/) for the web interface

---

<p align="center">
  Made with ❤️ for learning deep learning
</p>
