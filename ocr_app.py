"""
OCR Web Interface

Web interface for handwritten text recognition.
Upload an image of handwritten text and get the recognized words/sentences.
"""

import os
import numpy as np
import gradio as gr
from PIL import Image
import cv2

# Check for OCR module
try:
    from ocr import HandwritingOCR
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# Global OCR instance
ocr = None


def get_available_models():
    """Get list of available models for OCR."""
    available = []
    
    for dtype in ['letters', 'balanced', 'byclass']:
        paths = [f'./models/{dtype}_model.pth', f'./models/{dtype}_model_best.pth']
        for p in paths:
            if os.path.exists(p):
                available.append(dtype)
                break
    
    return available


def load_ocr(model_type: str):
    """Load or switch OCR model."""
    global ocr
    
    if ocr is None or ocr.model_type != model_type:
        ocr = HandwritingOCR(model_type=model_type)
    
    return ocr


def recognize_handwriting(image, model_type: str, detect_words: bool, confidence_threshold: float):
    """
    Recognize handwritten text from image.
    
    Args:
        image: Input image
        model_type: Type of model to use
        detect_words: Whether to group characters into words
        confidence_threshold: Minimum confidence for characters
        
    Returns:
        Tuple of (annotated_image, recognized_text, detailed_results)
    """
    
    if image is None:
        return None, "No image provided", ""
    
    try:
        # Load OCR
        ocr_instance = load_ocr(model_type)
        
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        # Recognize text
        result = ocr_instance.recognize_text(
            pil_image,
            detect_words=detect_words,
            confidence_threshold=confidence_threshold
        )
        
        # Handle case where no characters were found
        if result['num_characters'] == 0:
            msg = "⚠️ No text detected!\n\nPossible reasons:\n"
            msg += "- Image is too blurry or low contrast\n"
            msg += "- Handwriting is too faint\n"
            msg += "- Text is too small (try zooming in or cropping)\n"
            msg += "- Background is too cluttered"
            return image, msg, ""

        
        # Create visualization
        annotated = ocr_instance.visualize_detection(pil_image, result)
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        
        # Format results
        text = result.get('text', '')
        confidence = result.get('confidence', 0) * 100
        num_chars = result.get('num_characters', 0)
        num_words = result.get('num_words', 0) if detect_words else 'N/A'
        
        summary = f"📝 Recognized Text: {text}\n"
        summary += f"📊 Confidence: {confidence:.1f}%\n"
        summary += f"🔤 Characters: {num_chars}\n"
        summary += f"📖 Words: {num_words}"
        
        # Detailed character info
        details = "### Character Details:\n\n"
        details += "| Char | Confidence | Top 3 Predictions |\n"
        details += "|------|------------|-------------------|\n"
        
        for char in result.get('characters', [])[:20]:  # Limit to 20 chars
            c = char['char']
            conf = char['confidence'] * 100
            top3 = char.get('top_3', [])
            top3_str = ', '.join([f"{t[0]}:{t[1]*100:.0f}%" for t in top3[:3]])
            details += f"| {c} | {conf:.1f}% | {top3_str} |\n"
        
        if len(result.get('characters', [])) > 20:
            details += f"\n*...and {len(result['characters']) - 20} more characters*"
        
        return annotated_rgb, summary, details
        
    except Exception as e:
        return None, f"Error: {str(e)}", ""


def create_ocr_interface():
    """Create the OCR Gradio interface."""
    available_models = get_available_models()
    
    if not available_models:
        with gr.Blocks(title="Handwriting OCR") as demo:
            gr.Markdown("""
            # 📝 Handwritten Text Recognition (OCR)
            
            ## ⚠️ No Trained Models Found
            
            Please train a letters or alphanumeric model first:
            
            ```bash
            python train_extended.py --dataset letters --epochs 10
            ```
            
            Then restart this app.
            """)
        return demo
    
    # Model descriptions
    model_descriptions = {
        'letters': 'Letters (A-Z)',
        'balanced': 'Alphanumeric (47 classes)',
        'byclass': 'Full Alphanumeric (62 classes)'
    }
    
    default_model = available_models[0]
    
    with gr.Blocks(title="Handwriting OCR") as demo:
        gr.Markdown("""
        # 📝 Handwritten Text Recognition (OCR)
        
        Upload an image of handwritten text to recognize words and sentences.
        The system will:
        1. Detect individual characters
        2. Recognize each character using CNN
        3. Group characters into words
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="📷 Upload Handwritten Text Image",
                    type="numpy"
                )
                
                model_dropdown = gr.Dropdown(
                    choices=available_models,
                    value=default_model,
                    label="🤖 Model Type"
                )
                
                detect_words_checkbox = gr.Checkbox(
                    value=True,
                    label="🔤 Group characters into words"
                )
                
                confidence_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.3,
                    step=0.05,
                    label="📊 Minimum Confidence Threshold"
                )
                
                recognize_btn = gr.Button("🔍 Recognize Text", variant="primary")
                
            with gr.Column(scale=1):
                output_image = gr.Image(
                    label="🖼️ Detected Characters"
                )
                
                output_text = gr.Textbox(
                    label="📝 Recognition Results",
                    lines=5
                )
        
        with gr.Accordion("📋 Detailed Results", open=False):
            detailed_output = gr.Markdown()
        
        gr.Markdown("""
        ---
        
        ## 💡 Tips for Best Results
        
        - Use **clear, well-spaced handwriting**
        - **High contrast** images work better (dark text on light background)
        - Characters should be **separate**, not connected cursive
        - Try adjusting the **confidence threshold** if results are poor
        """)
        
        # Event handler
        recognize_btn.click(
            fn=recognize_handwriting,
            inputs=[image_input, model_dropdown, detect_words_checkbox, confidence_slider],
            outputs=[output_image, output_text, detailed_output]
        )
    
    return demo


def main():
    """Launch the OCR web interface."""
    print("\n" + "="*60)
    print("   📝 LAUNCHING HANDWRITING OCR")
    print("="*60)
    
    if not OCR_AVAILABLE:
        print("\n❌ OCR module not available!")
        print("   Make sure ocr.py exists and opencv-python is installed.")
        return
    
    available = get_available_models()
    if available:
        print(f"\n✓ Available models: {', '.join(available)}")
    else:
        print("\n⚠️ No trained models found!")
    
    print("\n🌐 Starting web interface...")
    
    demo = create_ocr_interface()
    
    print("\n" + "="*60)
    print("   ✅ OCR App is ready!")
    print("   Open your browser to the URL shown below")
    print("="*60 + "\n")
    
    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7861,  # Different port from main app
        show_error=True
    )


if __name__ == "__main__":
    main()
