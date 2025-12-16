
import cv2
import numpy as np
import torch
from ocr import HandwritingOCR
import matplotlib.pyplot as plt

def test_rotation():
    # Load the extracted character
    img_path = "debug_4_first_char_extracted.png"
    if not os.path.exists(img_path):
        print("Run reproduce_issue.py first")
        return
        
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # Initialize OCR (wrapper to access predictor)
    ocr = HandwritingOCR(model_type='letters')
    predictor = ocr.predictor
    
    transforms_list = [
        ("Original", lambda x: x),
        ("Rotate 90 CCW (Transpose)", lambda x: cv2.rotate(x, cv2.ROTATE_90_COUNTERCLOCKWISE)),
        ("Rotate 90 CW", lambda x: cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE)),
        ("Rotate 180", lambda x: cv2.rotate(x, cv2.ROTATE_180)),
        ("Flip Horizontal", lambda x: cv2.flip(x, 1)),
        ("Flip Vertical", lambda x: cv2.flip(x, 0)),
        ("Transpose + Flip", lambda x: cv2.flip(cv2.rotate(x, cv2.ROTATE_90_COUNTERCLOCKWISE), 0)) # Like EMNIST mapping
    ]
    
    print(f"\nTesting rotation on {img_path} (Should be 'A' or 'B' depending on bbox logic)\n")
    print(f"{'Transform':<25} | {'Pred':<5} | {'Conf':<10}")
    print("-" * 45)
    
    results = []
    
    for name, func in transforms_list:
        transformed = func(img)
        t_tensor = predictor.preprocess_image(transformed).to(predictor.device)
        
        # Predict
        with torch.no_grad():
            outputs = predictor.model(t_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, idx = probs.max(1)
            
        if idx.item() < len(predictor.class_info['labels']):
            label = predictor.class_info['labels'][idx.item()]
        else:
            label = "UNK"
        
        # print(f"{name:<25} | {label:<5} | {conf.item():.4f}")
        results.append(f"{name:<25} | {label:<5} | {conf.item():.4f}")
        
    with open("rotation_results.txt", "w") as f:
        f.write("\n".join(results))
    print("Saved results to rotation_results.txt")

if __name__ == "__main__":
    import os
    test_rotation()
