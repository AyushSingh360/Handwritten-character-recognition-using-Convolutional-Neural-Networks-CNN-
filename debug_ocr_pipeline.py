
import numpy as np
import cv2
import os
from ocr import HandwritingOCR

def run_test():
    print("Generating synthetic image...")
    # White background
    img_np = np.ones((150, 500), dtype=np.uint8) * 255
    # Black text
    # Using HERSHEY_SIMPLEX which is sans-serif
    cv2.putText(img_np, "HELLO WORLD", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 5)
    
    cv2.imwrite("test_input.png", img_np)
    print("Saved test_input.png")

    try:
        print("Initializing OCR...")
        ocr = HandwritingOCR(model_type='letters')
        
        print("Recognizing text...")
        result = ocr.recognize_text("test_input.png")
        
        print("\n" + "="*40)
        print("DEBUG RESULTS")
        print("="*40)
        print(f"Recognized Text: '{result['text']}'")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Number of Characters: {result['num_characters']}")
        print(f"Number of Words: {result['num_words']}")
        print(f"Words detected: {[w['word'] for w in result['words']]}")
        
        # Verify bboxes
        if result['num_characters'] == 0:
            print("\n❌ FAILED TO DETECT ANY CHARACTERS")
            # Debug binarization
            gray = ocr.preprocess_image("test_input.png")
            binary = ocr.binarize_image(gray)
            cv2.imwrite("debug_binary.png", binary)
            print("Saved debug_binary.png - Check this to see if thresholding failed.")
        else:
            print(f"\n✅ Detected {result['num_characters']} characters.")
            ocr.visualize_detection("test_input.png", result, "test_output.png")
            print("Saved test_output.png")
            
    except Exception as e:
        print(f"\n❌ EXCEPTION: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()
