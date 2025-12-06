"""
Web Interface for Alphanumeric Character Recognition

Supports:
- MNIST: Digits (0-9)
- EMNIST Letters: Letters (A-Z)
- EMNIST Balanced: Alphanumeric (47 classes)
- EMNIST ByClass: Full Alphanumeric (62 classes)
"""

import os
import numpy as np
import gradio as gr
from PIL import Image

from predict_extended import AlphanumericPredictor
from model_extended import get_class_info

# Global predictor
predictor = None
current_model_type = None


def load_predictor(model_type: str):
    """Load or switch predictor based on model type."""
    global predictor, current_model_type
    
    if current_model_type != model_type:
        try:
            predictor = AlphanumericPredictor(dataset_type=model_type)
            current_model_type = model_type
            return True
        except FileNotFoundError:
            return False
    return True


def get_available_models():
    """Get list of available trained models."""
    available = []
    
    # Check for extended models
    for dtype in ['mnist', 'letters', 'balanced', 'byclass']:
        paths = [f'./models/{dtype}_model.pth', f'./models/{dtype}_model_best.pth']
        for p in paths:
            if os.path.exists(p):
                available.append(dtype)
                break
    
    # Check for backward compatible MNIST
    if 'mnist' not in available:
        if os.path.exists('./models/mnist_cnn.pth') or os.path.exists('./models/mnist_cnn_best.pth'):
            available.append('mnist')
    
    return available


def predict_character(image, model_type: str):
    """
    Predict character from image.
    
    Args:
        image: Input image (numpy array)
        model_type: Type of model to use
        
    Returns:
        Dictionary of character probabilities
    """
    global predictor
    
    if image is None:
        return {}
    
    # Load predictor if needed
    if not load_predictor(model_type):
        return {"Error": 1.0}
    
    # Handle different input formats
    if isinstance(image, dict):
        img_array = image.get('composite', image.get('image'))
    else:
        img_array = image
    
    if img_array is None:
        return {}
    
    # Convert to numpy if needed
    if not isinstance(img_array, np.ndarray):
        img_array = np.array(img_array)
    
    # Convert to grayscale
    if len(img_array.shape) == 3:
        if img_array.shape[2] == 4:
            alpha = img_array[:, :, 3]
            if alpha.max() > 0:
                gray = alpha.astype(np.float32)
            else:
                gray = np.mean(img_array[:, :, :3], axis=2)
        else:
            gray = np.mean(img_array, axis=2)
    else:
        gray = img_array.astype(np.float32)
    
    # Normalize
    if gray.max() > 0:
        gray = (gray / gray.max() * 255).astype(np.uint8)
    else:
        gray = gray.astype(np.uint8)
    
    # Get top predictions
    top_preds = predictor.get_top_predictions(gray, top_k=10)
    
    return {label: conf for label, conf in top_preds}


def create_interface():
    """Create the Gradio interface."""
    available_models = get_available_models()
    
    if not available_models:
        # No models available - show training instructions
        with gr.Blocks(title="Alphanumeric Recognition") as demo:
            gr.Markdown("""
            # 🔤 Alphanumeric Character Recognition
            
            ## ⚠️ No Trained Models Found
            
            Please train a model first using one of these commands:
            
            ```bash
            # Digits only (0-9)
            python train_extended.py --dataset mnist --epochs 10
            
            # Letters only (A-Z)
            python train_extended.py --dataset letters --epochs 10
            
            # Alphanumeric (47 classes)
            python train_extended.py --dataset balanced --epochs 15
            
            # Full Alphanumeric (62 classes)
            python train_extended.py --dataset byclass --epochs 15
            ```
            
            Then restart this app.
            """)
        return demo
    
    # Model descriptions
    model_descriptions = {
        'mnist': '🔢 Digits (0-9) - 10 classes',
        'letters': '🔤 Letters (A-Z) - 26 classes',
        'balanced': '🔣 Alphanumeric - 47 classes',
        'byclass': '📝 Full Alphanumeric - 62 classes'
    }
    
    # Default to first available model
    default_model = available_models[0]
    load_predictor(default_model)
    
    demo = gr.Interface(
        fn=predict_character,
        inputs=[
            gr.Image(
                label="📷 Upload an image of a character",
                type="numpy",
                image_mode="L",
                sources=["upload"],
                height=280,
                width=280,
            ),
            gr.Dropdown(
                choices=available_models,
                value=default_model,
                label="🎯 Model Type",
                info="Select which type of characters to recognize"
            )
        ],
        outputs=gr.Label(
            label="🎯 Prediction",
            num_top_classes=10
        ),
        title="🔤 Alphanumeric Character Recognition",
        description=f"""
        Upload an image of a handwritten character to classify it.
        
        **Available Models:** {', '.join([model_descriptions.get(m, m) for m in available_models])}
        """,
        article="""
        ## 💡 Tips for Best Results
        - Use images with characters that are **large** and **centered**
        - White character on black background works best
        - Characters should be clearly written
        
        ## About the Models
        - **MNIST (Digits)**: Trained on 60,000 handwritten digit images
        - **EMNIST Letters**: Trained on 124,800 letter images (A-Z)
        - **EMNIST Balanced**: Balanced set of 47 alphanumeric classes
        - **EMNIST ByClass**: Full 62-class alphanumeric (0-9, A-Z, a-z)
        
        Built with PyTorch & Gradio
        """,
        live=True
    )
    
    return demo


def main():
    """Launch the Gradio demo."""
    print("\n" + "="*60)
    print("   🚀 LAUNCHING ALPHANUMERIC RECOGNITION APP")
    print("="*60)
    
    available = get_available_models()
    
    if available:
        print(f"\n📦 Available models: {', '.join(available)}")
    else:
        print("\n⚠️  No trained models found!")
        print("   Please train a model first.")
    
    print("\n🌐 Starting web interface...")
    
    demo = create_interface()
    
    print("\n" + "="*60)
    print("   ✅ App is ready!")
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
