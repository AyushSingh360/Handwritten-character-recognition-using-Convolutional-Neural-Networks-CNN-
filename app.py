"""
Interactive Gradio Demo for MNIST Digit Recognition

This script launches a web interface where users can draw digits
and get real-time predictions from the trained CNN model.
"""

import os
import numpy as np
import gradio as gr
from PIL import Image
import torch

from predict import MNISTPredictor


# Global predictor instance
predictor = None


def predict_digit(image):
    """
    Predict digit from drawn image.
    
    Args:
        image: Image from sketchpad (numpy array or dict)
        
    Returns:
        Dictionary of digit probabilities for Gradio Label component
    """
    global predictor
    
    if image is None:
        return {str(i): 0.0 for i in range(10)}
    
    # Handle different input formats from Gradio versions
    if isinstance(image, dict):
        img_array = image.get('composite', image.get('image'))
    else:
        img_array = image
        
    if img_array is None:
        return {str(i): 0.0 for i in range(10)}
    
    # Convert to numpy if needed
    if not isinstance(img_array, np.ndarray):
        img_array = np.array(img_array)
    
    # Convert RGBA/RGB to grayscale
    if len(img_array.shape) == 3:
        if img_array.shape[2] == 4:  # RGBA
            # Use alpha channel or brightness
            alpha = img_array[:, :, 3]
            if alpha.max() > 0:
                gray = alpha.astype(np.float32)
            else:
                gray = np.mean(img_array[:, :, :3], axis=2)
        else:  # RGB
            gray = np.mean(img_array, axis=2)
    else:
        gray = img_array.astype(np.float32)
    
    # Normalize to 0-255
    if gray.max() > 0:
        gray = (gray / gray.max() * 255).astype(np.uint8)
    else:
        gray = gray.astype(np.uint8)
    
    # Make prediction
    predicted, confidence, probs = predictor.predict(gray)
    
    # Return as dictionary for Label component
    return {str(i): float(probs[i]) for i in range(10)}


def main():
    """Launch the Gradio demo."""
    global predictor
    
    print("\n" + "="*60)
    print("   🚀 LAUNCHING MNIST DIGIT RECOGNITION DEMO")
    print("="*60)
    
    # Check if model exists
    model_path = './models/mnist_cnn.pth'
    best_model_path = './models/mnist_cnn_best.pth'
    
    if not os.path.exists(model_path) and not os.path.exists(best_model_path):
        print("\n⚠️  No trained model found!")
        print("   Please run 'python train.py' first to train the model.")
        print("="*60)
        return
    
    # Load predictor
    print("\n📦 Loading model...")
    try:
        predictor = MNISTPredictor()
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        return
    
    # Create the interface
    print("\n🌐 Starting web interface...")
    
    demo = gr.Interface(
        fn=predict_digit,
        inputs=gr.Image(
            label="✏️ Draw a digit (0-9)",
            type="numpy",
            image_mode="L",
            sources=["upload"],
            height=280,
            width=280,
        ),
        outputs=gr.Label(
            label="🎯 Prediction",
            num_top_classes=10
        ),
        title="🔢 Handwritten Digit Recognition",
        description="Upload an image of a handwritten digit (0-9) to classify it!",
        article="""
        ## 💡 Tips for Best Results
        - Use images with digits that are **large** and **centered**
        - White digit on black background works best
        - The model was trained on 28×28 grayscale images
        
        ---
        
        ## About This Model
        This demo uses a **Convolutional Neural Network (CNN)** trained on the MNIST dataset.
        - 3 Convolutional layers with BatchNorm and Dropout
        - ~300K trainable parameters
        - Achieves ~99% accuracy on MNIST test set
        
        Built with PyTorch & Gradio
        """,
        live=True
    )
    
    print("\n" + "="*60)
    print("   ✅ Demo is ready!")
    print("   Open your browser to the URL shown below")
    print("="*60 + "\n")
    
    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True
    )


if __name__ == "__main__":
    main()
