
import cv2
import numpy as np
import os

def test_thresh():
    image_path = "test_debug.png"
    if not os.path.exists(image_path):
        return
        
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return
        
    # Method 1: Current Adaptive
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    adaptive = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 10
    )
    cv2.imwrite("debug_test_adaptive.png", adaptive)
    
    # Method 2: Otsu
    # Gaussian blur needed? Usually yes.
    blurred_otsu = cv2.GaussianBlur(img, (5, 5), 0)
    _, otsu = cv2.threshold(
        blurred_otsu, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    cv2.imwrite("debug_test_otsu.png", otsu)
    
    print("Saved debug_test_adaptive.png and debug_test_otsu.png")
    
    # Check simple stats
    print(f"Adaptive Mean: {np.mean(adaptive):.2f}")
    print(f"Otsu Mean: {np.mean(otsu):.2f}")

if __name__ == "__main__":
    test_thresh()
