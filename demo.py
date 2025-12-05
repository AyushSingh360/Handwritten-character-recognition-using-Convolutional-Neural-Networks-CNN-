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


def create_demo(predictor: MNISTPredictor) -> gr.Blocks:
    """
    Create the Gradio demo interface.
    
    Args:
        predictor: MNISTPredictor instance
        
    Returns:
        Gradio Blocks interface
    """
    
    def predict_digit(image: dict) -> dict:
        """
        Predict digit from drawn image.
        
        Args:
            image: Dictionary with 'composite' key containing the drawn image
            
        Returns:
            Dictionary of digit probabilities for Gradio Label component
        """
        if image is None:
            return {str(i): 0.0 for i in range(10)}
        
        # Extract the composite image (what the user drew)
        img_array = image.get('composite')
        if img_array is None:
            return {str(i): 0.0 for i in range(10)}
        
        # Convert RGBA to grayscale, inverting colors
        # (Gradio draws white on black, MNIST is white on black)
        if len(img_array.shape) == 3:
            if img_array.shape[2] == 4:  # RGBA
                # Use alpha channel to determine drawing
                alpha = img_array[:, :, 3]
                # Where alpha is high (drawn), use white; else black
                gray = alpha.astype(np.float32)
            else:  # RGB
                gray = np.mean(img_array, axis=2)
        else:
            gray = img_array
        
        # Normalize to 0-255
        gray = (gray / gray.max() * 255).astype(np.uint8) if gray.max() > 0 else gray.astype(np.uint8)
        
        # Make prediction
        predicted, confidence, probs = predictor.predict(gray)
        
        # Return as dictionary for Label component
        return {str(i): float(probs[i]) for i in range(10)}
    
    # Custom CSS for styling
    custom_css = """
    .gradio-container {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .main-title {
        text-align: center;
        font-size: 2.5em;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5em;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1em;
        margin-bottom: 2em;
    }
    """
    
    # Create the interface
    with gr.Blocks(css=custom_css, title="MNIST Digit Recognition") as demo:
        gr.HTML("""
            <div class="main-title">🔢 Handwritten Digit Recognition</div>
            <div class="subtitle">Draw a digit (0-9) and watch the CNN classify it in real-time!</div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ✏️ Draw Here")
                canvas = gr.Sketchpad(
                    label="Draw a digit (0-9)",
                    type="numpy",
                    image_mode="RGBA",
                    brush=gr.Brush(
                        colors=["#FFFFFF"],
                        default_size=20,
                        color_mode="fixed"
                    ),
                    canvas_size=(280, 280),
                    layers=False
                )
                
                clear_btn = gr.Button("🗑️ Clear Canvas", variant="secondary")
                
            with gr.Column(scale=1):
                gr.Markdown("### 🎯 Prediction")
                output = gr.Label(
                    label="Digit Probabilities",
                    num_top_classes=10
                )
                
                gr.Markdown("""
                ### 💡 Tips for Best Results
                - Draw digits **large** and **centered**
                - Use **thick strokes** for better recognition
                - Draw digits similar to how you'd write by hand
                - The model works best with clear, simple digits
                """)
        
        # Event handlers
        canvas.change(
            fn=predict_digit,
            inputs=[canvas],
            outputs=[output]
        )
        
        clear_btn.click(
            fn=lambda: None,
            outputs=[canvas]
        )
        
        gr.Markdown("---")
        
        with gr.Accordion("ℹ️ About This Project", open=False):
            gr.Markdown("""
            ### Model Architecture
            This demo uses a **Convolutional Neural Network (CNN)** trained on the MNIST dataset.
            
            **Network Structure:**
            - 3 Convolutional layers with BatchNorm and Dropout
            - 3 Fully connected layers
            - ~300K trainable parameters
            - Achieves ~99% accuracy on MNIST test set
            
            **Training Details:**
            - Dataset: 60,000 training images, 10,000 test images
            - Image size: 28×28 grayscale
            - Optimizer: Adam with learning rate scheduling
            - Data augmentation: Random rotation, translation, scaling
            
            ### How It Works
            1. Your drawing is captured and converted to a 28×28 grayscale image
            2. The image is normalized using MNIST statistics
            3. The CNN processes the image through its layers
            4. Softmax activation produces probabilities for each digit (0-9)
            5. The digit with highest probability is the prediction
            """)
        
        gr.Markdown("""
        <div style="text-align: center; color: #888; font-size: 0.9em; margin-top: 2em;">
            Built with PyTorch & Gradio | MNIST Handwritten Digit Recognition
        </div>
        """)
    
    return demo


def main():
    """Launch the Gradio demo."""
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
    
    # Create and launch demo
    print("\n🌐 Starting web interface...")
    demo = create_demo(predictor)
    
    print("\n" + "="*60)
    print("   ✅ Demo is ready!")
    print("   Open your browser to the URL shown below")
    print("="*60 + "\n")
    
    demo.launch(
        share=False,  # Set to True to create a public link
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True
    )


if __name__ == "__main__":
    main()
