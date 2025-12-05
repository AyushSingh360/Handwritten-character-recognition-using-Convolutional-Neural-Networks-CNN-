"""
Batch Processing Module for MNIST Digit Recognition

This module provides functionality for processing multiple images
at once, including folder processing and saving predictions.
"""

import os
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from PIL import Image
import torch

from predict import MNISTPredictor


class BatchProcessor:
    """
    Process multiple digit images in batch.
    
    Supports:
    - Single image prediction
    - Folder batch processing
    - Results export (JSON, CSV)
    - Prediction logging
    """
    
    def __init__(
        self, 
        model_path: str = './models/mnist_cnn.pth',
        output_dir: str = './predictions'
    ):
        """
        Initialize the batch processor.
        
        Args:
            model_path: Path to trained model
            output_dir: Directory to save prediction results
        """
        self.predictor = MNISTPredictor(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create results subdirectories
        (self.output_dir / 'logs').mkdir(exist_ok=True)
        (self.output_dir / 'reports').mkdir(exist_ok=True)
        
        self.prediction_log: List[Dict] = []
        
    def predict_single(
        self, 
        image_path: str,
        save_result: bool = True
    ) -> Dict:
        """
        Predict a single image and optionally save the result.
        
        Args:
            image_path: Path to the image file
            save_result: Whether to log the prediction
            
        Returns:
            Dictionary with prediction results
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Make prediction
        predicted, confidence, probs = self.predictor.predict(str(image_path))
        
        result = {
            'image_path': str(image_path),
            'image_name': image_path.name,
            'predicted_digit': int(predicted),
            'confidence': float(confidence),
            'all_probabilities': {str(i): float(probs[i]) for i in range(10)},
            'timestamp': datetime.now().isoformat()
        }
        
        if save_result:
            self.prediction_log.append(result)
        
        return result
    
    def process_folder(
        self, 
        folder_path: str,
        extensions: Tuple[str, ...] = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    ) -> List[Dict]:
        """
        Process all images in a folder.
        
        Args:
            folder_path: Path to folder containing images
            extensions: File extensions to include
            
        Returns:
            List of prediction results
        """
        folder_path = Path(folder_path)
        
        if not folder_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {folder_path}")
        
        # Find all image files
        image_files = []
        for ext in extensions:
            image_files.extend(folder_path.glob(f'*{ext}'))
            image_files.extend(folder_path.glob(f'*{ext.upper()}'))
        
        if not image_files:
            print(f"No images found in {folder_path}")
            return []
        
        print(f"\n📂 Processing {len(image_files)} images from {folder_path}")
        print("-" * 50)
        
        results = []
        for i, img_path in enumerate(sorted(image_files), 1):
            try:
                result = self.predict_single(str(img_path))
                results.append(result)
                
                status = "✓" if result['confidence'] > 0.9 else "?"
                print(f"  {status} [{i}/{len(image_files)}] {img_path.name}: "
                      f"Digit {result['predicted_digit']} ({result['confidence']*100:.1f}%)")
                
            except Exception as e:
                print(f"  ✗ [{i}/{len(image_files)}] {img_path.name}: Error - {e}")
                results.append({
                    'image_path': str(img_path),
                    'image_name': img_path.name,
                    'error': str(e)
                })
        
        print("-" * 50)
        successful = sum(1 for r in results if 'error' not in r)
        print(f"✓ Processed {successful}/{len(image_files)} images successfully\n")
        
        return results
    
    def save_results_json(
        self, 
        filename: Optional[str] = None,
        results: Optional[List[Dict]] = None
    ) -> str:
        """
        Save prediction results to JSON file.
        
        Args:
            filename: Output filename (auto-generated if None)
            results: Results to save (uses prediction_log if None)
            
        Returns:
            Path to saved file
        """
        if results is None:
            results = self.prediction_log
        
        if not results:
            print("No results to save")
            return ""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"predictions_{timestamp}.json"
        
        output_path = self.output_dir / 'reports' / filename
        
        with open(output_path, 'w') as f:
            json.dump({
                'total_predictions': len(results),
                'generated_at': datetime.now().isoformat(),
                'predictions': results
            }, f, indent=2)
        
        print(f"✓ Saved JSON report to {output_path}")
        return str(output_path)
    
    def save_results_csv(
        self, 
        filename: Optional[str] = None,
        results: Optional[List[Dict]] = None
    ) -> str:
        """
        Save prediction results to CSV file.
        
        Args:
            filename: Output filename (auto-generated if None)
            results: Results to save (uses prediction_log if None)
            
        Returns:
            Path to saved file
        """
        if results is None:
            results = self.prediction_log
        
        if not results:
            print("No results to save")
            return ""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"predictions_{timestamp}.csv"
        
        output_path = self.output_dir / 'reports' / filename
        
        # Filter out error entries for CSV
        valid_results = [r for r in results if 'error' not in r]
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Image Name', 'Predicted Digit', 'Confidence (%)', 
                'Timestamp', 'Image Path'
            ])
            
            for r in valid_results:
                writer.writerow([
                    r['image_name'],
                    r['predicted_digit'],
                    f"{r['confidence'] * 100:.2f}",
                    r['timestamp'],
                    r['image_path']
                ])
        
        print(f"✓ Saved CSV report to {output_path}")
        return str(output_path)
    
    def generate_summary(self, results: Optional[List[Dict]] = None) -> Dict:
        """
        Generate a summary of prediction results.
        
        Args:
            results: Results to summarize (uses prediction_log if None)
            
        Returns:
            Summary statistics dictionary
        """
        if results is None:
            results = self.prediction_log
        
        valid_results = [r for r in results if 'error' not in r]
        
        if not valid_results:
            return {'error': 'No valid predictions'}
        
        # Digit distribution
        digit_counts = {str(i): 0 for i in range(10)}
        confidences = []
        
        for r in valid_results:
            digit_counts[str(r['predicted_digit'])] += 1
            confidences.append(r['confidence'])
        
        summary = {
            'total_images': len(results),
            'successful_predictions': len(valid_results),
            'failed_predictions': len(results) - len(valid_results),
            'digit_distribution': digit_counts,
            'average_confidence': float(np.mean(confidences)),
            'min_confidence': float(np.min(confidences)),
            'max_confidence': float(np.max(confidences)),
            'high_confidence_count': sum(1 for c in confidences if c > 0.95),
            'low_confidence_count': sum(1 for c in confidences if c < 0.5)
        }
        
        return summary
    
    def print_summary(self, results: Optional[List[Dict]] = None) -> None:
        """Print a formatted summary of results."""
        summary = self.generate_summary(results)
        
        if 'error' in summary:
            print(f"❌ {summary['error']}")
            return
        
        print("\n" + "="*50)
        print("📊 PREDICTION SUMMARY")
        print("="*50)
        print(f"  Total Images:       {summary['total_images']}")
        print(f"  Successful:         {summary['successful_predictions']}")
        print(f"  Failed:             {summary['failed_predictions']}")
        print(f"  Avg Confidence:     {summary['average_confidence']*100:.1f}%")
        print(f"  High Confidence:    {summary['high_confidence_count']} (>95%)")
        print(f"  Low Confidence:     {summary['low_confidence_count']} (<50%)")
        print("\n📈 Digit Distribution:")
        
        for digit, count in summary['digit_distribution'].items():
            if count > 0:
                bar = "█" * count
                print(f"    {digit}: {bar} ({count})")
        
        print("="*50 + "\n")
    
    def clear_log(self) -> None:
        """Clear the prediction log."""
        self.prediction_log = []
        print("✓ Prediction log cleared")


def main():
    """Demo the batch processing functionality."""
    print("\n" + "="*60)
    print("   📦 MNIST BATCH PROCESSING DEMO")
    print("="*60)
    
    # Initialize processor
    processor = BatchProcessor()
    
    print("\n💡 Batch Processor initialized!")
    print("   Use this module to:")
    print("   - Process multiple images at once")
    print("   - Save predictions to JSON/CSV")
    print("   - Generate summary reports")
    
    print("\n📝 Example usage:")
    print('   processor = BatchProcessor()')
    print('   results = processor.process_folder("./my_digits/")')
    print('   processor.save_results_csv()')
    print('   processor.print_summary()')
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
