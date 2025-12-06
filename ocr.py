"""
Handwritten Text Recognition (OCR) Module

This module provides functionality to:
1. Segment individual characters from handwritten text images
2. Recognize each character using the trained CNN
3. Combine results into words/sentences
"""

import os
import numpy as np
from PIL import Image
import cv2
from typing import List, Tuple, Optional, Union
import torch

from predict_extended import AlphanumericPredictor


class HandwritingOCR:
    """
    OCR system for handwritten text recognition.
    
    Uses trained CNN models to recognize individual characters
    and combines them into words/sentences.
    """
    
    def __init__(
        self,
        model_type: str = 'letters',
        min_char_width: int = 10,
        min_char_height: int = 15,
        padding: int = 2
    ):
        """
        Initialize the OCR system.
        
        Args:
            model_type: Type of model to use ('letters', 'balanced', 'byclass')
            min_char_width: Minimum width for character detection
            min_char_height: Minimum height for character detection
            padding: Padding around detected characters
        """
        self.predictor = AlphanumericPredictor(dataset_type=model_type)
        self.model_type = model_type
        self.min_char_width = min_char_width
        self.min_char_height = min_char_height
        self.padding = padding
        
        print(f"✓ OCR initialized with {model_type} model")
    
    def preprocess_image(self, image: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Preprocess image for character segmentation.
        
        Args:
            image: Input image (path, numpy array, or PIL Image)
            
        Returns:
            Preprocessed grayscale image as numpy array
        """
        # Load image
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise FileNotFoundError(f"Could not load image: {image}")
        elif isinstance(image, Image.Image):
            img = np.array(image)
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        else:
            img = image.copy()
        
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        return gray
    
    def binarize_image(self, gray: np.ndarray) -> np.ndarray:
        """
        Convert grayscale to binary (black/white) image.
        
        Args:
            gray: Grayscale image
            
        Returns:
            Binary image
        """
        # Apply Gaussian Blur to reduce noise
        # This helps with paper texture and small artifacts
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        # Increased block size from 11 to 31 to handle thicker strokes better
        # Adjusted C constant from 2 to 10 to reduce background noise
        binary = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31, 10
        )
        
        # Optional: Apply morphological operations to clean up
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return binary
    
    def find_character_bboxes(
        self, 
        binary: np.ndarray,
        sort_by: str = 'left'
    ) -> List[Tuple[int, int, int, int]]:
        """
        Find bounding boxes for individual characters.
        
        Args:
            binary: Binary image
            sort_by: How to sort bboxes ('left', 'top')
            
        Returns:
            List of bounding boxes (x, y, w, h)
        """
        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        bboxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size
            if w >= self.min_char_width and h >= self.min_char_height:
                bboxes.append((x, y, w, h))
        
        # Sort bounding boxes
        if sort_by == 'left':
            bboxes.sort(key=lambda b: b[0])  # Sort by x coordinate
        elif sort_by == 'top':
            bboxes.sort(key=lambda b: b[1])  # Sort by y coordinate
        
        return bboxes
    
    def extract_character(
        self, 
        image: np.ndarray, 
        bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Extract and prepare a single character for recognition.
        
        Args:
            image: Source image (grayscale)
            bbox: Bounding box (x, y, w, h)
            
        Returns:
            28x28 character image ready for CNN
        """
        x, y, w, h = bbox
        
        # Add padding
        x1 = max(0, x - self.padding)
        y1 = max(0, y - self.padding)
        x2 = min(image.shape[1], x + w + self.padding)
        y2 = min(image.shape[0], y + h + self.padding)
        
        # Extract character
        char_img = image[y1:y2, x1:x2]
        
        # Ensure it's inverted (white on black for MNIST/EMNIST style)
        if np.mean(char_img) > 127:
            char_img = 255 - char_img
        
        # Resize to 28x28 while maintaining aspect ratio
        h, w = char_img.shape
        if h > w:
            new_h = 22
            new_w = int(w * 22 / h)
        else:
            new_w = 22
            new_h = int(h * 22 / w)
        
        if new_w < 1:
            new_w = 1
        if new_h < 1:
            new_h = 1
            
        char_img = cv2.resize(char_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Center in 28x28 canvas
        canvas = np.zeros((28, 28), dtype=np.uint8)
        y_offset = (28 - new_h) // 2
        x_offset = (28 - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = char_img
        
        return canvas
    
    def detect_word_boundaries(
        self, 
        bboxes: List[Tuple[int, int, int, int]],
        space_threshold: float = 1.5
    ) -> List[List[Tuple[int, int, int, int]]]:
        """
        Group character bboxes into words based on spacing.
        
        Args:
            bboxes: List of character bounding boxes
            space_threshold: Multiplier of average char width to detect space
            
        Returns:
            List of words, each containing character bboxes
        """
        if not bboxes:
            return []
        
        # Calculate average character width
        avg_width = np.mean([b[2] for b in bboxes])
        
        words = []
        current_word = [bboxes[0]]
        
        for i in range(1, len(bboxes)):
            prev_bbox = bboxes[i-1]
            curr_bbox = bboxes[i]
            
            # Calculate gap between characters
            gap = curr_bbox[0] - (prev_bbox[0] + prev_bbox[2])
            
            # If gap is larger than threshold, start new word
            if gap > avg_width * space_threshold:
                words.append(current_word)
                current_word = [curr_bbox]
            else:
                current_word.append(curr_bbox)
        
        if current_word:
            words.append(current_word)
        
        return words
    
    def recognize_text(
        self, 
        image: Union[str, np.ndarray, Image.Image],
        detect_words: bool = True,
        confidence_threshold: float = 0.5
    ) -> dict:
        """
        Recognize handwritten text from an image.
        
        Args:
            image: Input image
            detect_words: Whether to group characters into words
            confidence_threshold: Minimum confidence to include character
            
        Returns:
            Dictionary with recognized text and details
        """
        # Preprocess
        gray = self.preprocess_image(image)
        binary = self.binarize_image(gray)
        
        # Find characters
        bboxes = self.find_character_bboxes(binary)
        
        if not bboxes:
            return {
                'text': '',
                'words': [],
                'characters': [],
                'confidence': 0.0
            }
        
        # Recognize each character
        characters = []
        for bbox in bboxes:
            char_img = self.extract_character(gray, bbox)
            label, idx, conf, probs = self.predictor.predict(char_img)
            
            characters.append({
                'char': label,
                'confidence': float(conf),
                'bbox': bbox,
                'top_3': self.predictor.get_top_predictions(char_img, top_k=3)
            })
        
        # Group into words if requested
        if detect_words:
            word_bboxes = self.detect_word_boundaries(bboxes)
            words = []
            char_idx = 0
            
            for word_bbox_group in word_bboxes:
                word_chars = []
                for _ in word_bbox_group:
                    if char_idx < len(characters):
                        word_chars.append(characters[char_idx])
                        char_idx += 1
                
                word_text = ''.join([c['char'] for c in word_chars 
                                    if c['confidence'] >= confidence_threshold])
                word_conf = np.mean([c['confidence'] for c in word_chars]) if word_chars else 0
                
                words.append({
                    'word': word_text,
                    'confidence': float(word_conf),
                    'characters': word_chars
                })
            
            # Build full text
            full_text = ' '.join([w['word'] for w in words])
            avg_confidence = np.mean([c['confidence'] for c in characters])
            
            return {
                'text': full_text,
                'words': words,
                'characters': characters,
                'confidence': float(avg_confidence),
                'num_characters': len(characters),
                'num_words': len(words)
            }
        else:
            # Just return characters without word grouping
            text = ''.join([c['char'] for c in characters 
                          if c['confidence'] >= confidence_threshold])
            avg_confidence = np.mean([c['confidence'] for c in characters])
            
            return {
                'text': text,
                'characters': characters,
                'confidence': float(avg_confidence),
                'num_characters': len(characters)
            }
    
    def visualize_detection(
        self,
        image: Union[str, np.ndarray, Image.Image],
        result: dict,
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Visualize detected characters on the image.
        
        Args:
            image: Original image
            result: Recognition result from recognize_text()
            output_path: Optional path to save visualization
            
        Returns:
            Image with bounding boxes drawn
        """
        # Load image
        if isinstance(image, str):
            img = cv2.imread(image)
        elif isinstance(image, Image.Image):
            img = np.array(image)
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img = image.copy()
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Draw bounding boxes
        for char in result.get('characters', []):
            x, y, w, h = char['bbox']
            conf = char['confidence']
            label = char['char']
            
            # Color based on confidence
            if conf > 0.8:
                color = (0, 255, 0)  # Green
            elif conf > 0.5:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 0, 255)  # Red
            
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img, f"{label}", (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Add recognized text at bottom
        text = result.get('text', '')
        if text:
            cv2.putText(img, f"Text: {text}", (10, img.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if output_path:
            cv2.imwrite(output_path, img)
            print(f"✓ Saved visualization to {output_path}")
        
        return img


def main():
    """Demo the OCR functionality."""
    print("\n" + "="*60)
    print("   📝 HANDWRITTEN TEXT RECOGNITION (OCR)")
    print("="*60)
    
    # Check for available models
    model_types = ['letters', 'balanced', 'byclass']
    available = None
    
    for mt in model_types:
        paths = [f'./models/{mt}_model.pth', f'./models/{mt}_model_best.pth']
        for p in paths:
            if os.path.exists(p):
                available = mt
                break
        if available:
            break
    
    if not available:
        print("\n❌ No letter/alphanumeric model found!")
        print("   Train a model first:")
        print("   python train_extended.py --dataset letters --epochs 10")
        return
    
    print(f"\n✓ Using {available} model")
    
    # Initialize OCR
    ocr = HandwritingOCR(model_type=available)
    
    print("\n💡 OCR System Ready!")
    print("   Usage example:")
    print("   ")
    print("   from ocr import HandwritingOCR")
    print("   ocr = HandwritingOCR(model_type='letters')")
    print("   result = ocr.recognize_text('handwritten_image.png')")
    print("   print(result['text'])")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
