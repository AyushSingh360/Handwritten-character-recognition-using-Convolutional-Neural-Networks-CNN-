
import cv2
import numpy as np
from ocr import HandwritingOCR

def visualize_order():
    image_path = "test_debug.png"
    ocr = HandwritingOCR(model_type='letters')
    
    # Custom pipeline step by step
    gray = ocr.preprocess_image(image_path)
    binary = ocr.binarize_image(gray)
    bboxes = ocr.find_character_bboxes(binary)
    
    img = cv2.imread(image_path)
    
    for i, (x, y, w, h) in enumerate(bboxes):
        # Draw bbox
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
        # Draw index
        cv2.putText(img, str(i), (x, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        print(f"Index {i}: ({x}, {y}) - w={w}, h={h}")
                   
    cv2.imwrite("debug_order.png", img)
    print(f"Saved debug_order.png with {len(bboxes)} indexed boxes.")

if __name__ == "__main__":
    visualize_order()
