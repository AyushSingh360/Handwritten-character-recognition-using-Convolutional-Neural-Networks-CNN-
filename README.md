# 🔤 Handwritten Character Recognition

A complete deep learning project for **handwritten character recognition** using Convolutional Neural Networks (CNN). Supports **digits**, **letters**, and **full alphanumeric** recognition.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-99%25+-green.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## ✨ Features

- 🔢 **Digits (0-9)** - MNIST dataset, 10 classes
- 🔤 **Letters (A-Z)** - EMNIST Letters, 26 classes
- 🔣 **Alphanumeric** - EMNIST Balanced, 47 classes
- 📝 **Full Alphanumeric** - EMNIST ByClass, 62 classes (0-9, A-Z, a-z)
- 🎨 **Web Interface** - Upload and classify with Gradio
- 📦 **Batch Processing** - Process folders, export to CSV/JSON
- 📓 **Jupyter Notebook** - Step-by-step tutorial
- 🚀 **99%+ Accuracy** - Production-ready models

## 🏗️ Project Structure

```
CNN/
├── model.py              # Original MNIST CNN (backward compatible)
├── model_extended.py     # Extended CNN for alphanumeric (10-62 classes)
├── train.py              # Original MNIST training
├── train_extended.py     # Unified training for all datasets
├── predict.py            # Original MNIST predictor
├── predict_extended.py   # Extended predictor for all models
├── data_loader.py        # MNIST/EMNIST data utilities
├── utils.py              # Helper functions
├── app.py                # Web interface
├── batch_processor.py    # Batch processing
├── mnist_walkthrough.ipynb  # Tutorial notebook
├── requirements.txt      # Dependencies
├── README.md             # This file
├── models/               # Trained model checkpoints
│   ├── mnist_cnn.pth
│   ├── letters_model.pth
│   ├── balanced_model.pth
│   └── byclass_model.pth
└── data/                 # Datasets (auto-downloaded)
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Models

```bash
# Digits only (0-9) - fastest
python train_extended.py --dataset mnist --epochs 10

# Letters only (A-Z)
python train_extended.py --dataset letters --epochs 10

# Alphanumeric (47 classes)
python train_extended.py --dataset balanced --epochs 15

# Full Alphanumeric (62 classes) - most comprehensive
python train_extended.py --dataset byclass --epochs 15
```

### 3. Launch Web Interface

```bash
python app.py
```
Open http://127.0.0.1:7860 and select the model type.

## 📊 Supported Datasets

| Dataset | Classes | Characters | Training Samples |
|---------|---------|------------|------------------|
| **MNIST** | 10 | 0-9 | 60,000 |
| **EMNIST Letters** | 26 | A-Z | 124,800 |
| **EMNIST Balanced** | 47 | 0-9, A-Z, some lowercase | 112,800 |
| **EMNIST ByClass** | 62 | 0-9, A-Z, a-z | 697,932 |

## 🧠 Model Architecture

```
Input (1, 28, 28) - Grayscale 28×28 image
    │
    ▼
┌─────────────────────────────────────────┐
│  Conv Block 1: Conv2D(1→32) + BN + ReLU │
│  MaxPool(2×2) + Dropout(0.25)           │
├─────────────────────────────────────────┤
│  Conv Block 2: Conv2D(32→64) + BN + ReLU│
│  MaxPool(2×2) + Dropout(0.25)           │
├─────────────────────────────────────────┤
│  Conv Block 3: Conv2D(64→128) + BN      │
│  MaxPool(2×2) + Dropout(0.25)           │
├─────────────────────────────────────────┤
│  Conv Block 4: Conv2D(128→256) + BN     │
│  Dropout(0.25)                          │
├─────────────────────────────────────────┤
│  FC: 2304 → 512 → 256 → N classes       │
│  (N = 10, 26, 47, or 62)                │
└─────────────────────────────────────────┘
```

**Model Size:** ~2-3 MB depending on output classes

## 📚 Usage Examples

### Basic Prediction

```python
from predict_extended import AlphanumericPredictor

# Load specific model
predictor = AlphanumericPredictor(dataset_type='letters')

# Predict
label, index, confidence, probs = predictor.predict("image.png")
print(f"Predicted: {label} ({confidence:.1%})")

# Get top-5 predictions
top5 = predictor.get_top_predictions("image.png", top_k=5)
for char, conf in top5:
    print(f"  {char}: {conf:.1%}")
```

### Batch Processing

```python
from batch_processor import BatchProcessor

processor = BatchProcessor()
results = processor.process_folder("./handwritten_chars/")
processor.save_results_csv()
processor.print_summary()
```

## 🔧 Command Line Options

```bash
python train_extended.py --dataset TYPE --epochs N --batch-size B --lr RATE

# Options:
#   --dataset   : mnist, letters, balanced, byclass
#   --epochs    : Number of training epochs
#   --batch-size: Batch size (default: 64)
#   --lr        : Learning rate (default: 0.001)
#   --no-cuda   : Disable GPU
```

## 📓 Jupyter Notebook

For an interactive tutorial:

```bash
jupyter notebook mnist_walkthrough.ipynb
```

## 🛠️ Development

### Test Models

```bash
# Test extended model
python model_extended.py

# Test data loader
python data_loader.py

# Test predictions
python predict_extended.py
```

## 📄 License

MIT License

---

<p align="center">
  Made with ❤️ for learning deep learning by Ayush Singh
</p>
