
import cv2
import numpy as np
import os

def inspect():
    files = [
        "debug_1_gray.png",
        "debug_2_binary.png",
        "debug_4_first_char_extracted.png"
    ]
    
    for f in files:
        if not os.path.exists(f):
            print(f"❌ {f} NOT FOUND")
            continue
            
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"❌ {f} failed to load")
            continue
            
        mean = np.mean(img)
        min_val = np.min(img)
        max_val = np.max(img)
        
        print(f"File: {f}")
        print(f"  Shape: {img.shape}")
        print(f"  Mean: {mean:.2f}")
        print(f"  Min: {min_val}, Max: {max_val}")
        
        # ASCII Art preview for small images
        if f == "debug_4_first_char_extracted.png":
            print("  Preview:")
            h, w = img.shape
            for y in range(h):
                line = ""
                for x in range(w):
                    val = img[y, x]
                    if val > 128:
                        line += "#"
                    elif val > 64:
                        line += "."
                    else:
                        line += " "
                print(f"    |{line}|")

if __name__ == "__main__":
    inspect()
