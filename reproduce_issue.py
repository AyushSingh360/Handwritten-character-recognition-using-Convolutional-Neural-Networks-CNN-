
import cv2
import numpy as np
import os
from ocr import HandwritingOCR

def reproduce():
    image_path = "test_debug.png"
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found.")
        return

    print(f"Processing {image_path}...")
    
    # Initialize OCR with letters model
    try:
        ocr = HandwritingOCR(model_type='letters')
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 1. Visualize Preprocessing
    gray = ocr.preprocess_image(image_path)
    cv2.imwrite("debug_1_gray.png", gray)
    
    binary = ocr.binarize_image(gray)
    cv2.imwrite("debug_2_binary.png", binary)
    
    # 2. Visualize Contours/BBoxes
    bboxes = ocr.find_character_bboxes(binary)
    print(f"Found {len(bboxes)} bounding boxes.")
    
    debug_bbox_img = cv2.imread(image_path)
    for (x, y, w, h) in bboxes:
        cv2.rectangle(debug_bbox_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imwrite("debug_3_bboxes.png", debug_bbox_img)
    
    # 3. Check Individual Character Processing
    if len(bboxes) > 0:
        # Check the second bbox (Index 1) as a sample 'B'
        if len(bboxes) > 1:
            char_index = 1
            char_img = ocr.extract_character(binary, bboxes[char_index])
            cv2.imwrite("debug_4_first_char_extracted.png", char_img)
            print("Saved debug images (Index 1).")
            
            # DEBUG RAW CROP
            x, y, w, h = bboxes[char_index]
            cropped = binary[y:y+h, x:x+w]
            cv2.imwrite("debug_cropped_raw.png", cropped)
            print(f"Cropped raw stats: Mean {np.mean(cropped):.2f}, Max {np.max(cropped)}")

        
    # 4. Full Recognition
    result = ocr.recognize_text(image_path)
    
    with open("debug_output.txt", "w", encoding="utf-8") as f:
        f.write(f"Found {len(bboxes)} bounding boxes.\n")
        f.write("\nRecognition Result:\n")
        f.write(f"Text: {result['text']}\n")
        f.write(f"Confidence: {result['confidence']:.4f}\n")
        
        f.write("\nCharacter details:\n")
        for c in result['characters']:
            f.write(f"{c['char']}: {c['confidence']:.4f}\n")
            
    print("\nRecognition Result:")
    print(f"Text: {result['text']}")
    print(f"Confidence: {result['confidence']:.4f}")
    
    # Save visualization
    ocr.visualize_detection(image_path, result, "debug_5_final_result.png")

if __name__ == "__main__":
    reproduce()
