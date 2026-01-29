"""
Color Histogram feature extractor.
Extracts color distribution information from images.
"""
from typing import Tuple
import numpy as np
import cv2


class ColorHistogramExtractor:
    """
    Color Histogram feature extractor.
    
    Extracts color distribution histograms from HSV color space.
    """
    
    def __init__(self, 
                 bins: int = 32,
                 color_space: str = 'hsv'):
        """
        Initialize Color Histogram extractor.
        
        Args:
            bins: Number of bins per channel
            color_space: Color space to use ('hsv' or 'bgr')
        """
        self.bins = bins
        self.color_space = color_space.lower()
        
        # Feature dimension: bins per channel * number of channels
        self.feature_dim = bins * 3
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract Color Histogram features from image.
        
        Args:
            image: Input image (BGR or HSV, float32, 0-1 range)
            
        Returns:
            Color Histogram feature vector (float32)
        """
        # Convert to uint8 (0-255) for histogram computation
        if image.max() <= 1.0:
            img_uint8 = (image * 255).astype(np.uint8)
        else:
            img_uint8 = image.astype(np.uint8)
        
        # Convert to HSV if needed
        if self.color_space == 'hsv':
            if len(img_uint8.shape) == 3:
                # Assume BGR if 3 channels
                img_color = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2HSV)
            else:
                # Grayscale: convert to 3-channel first
                img_color = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
                img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
        else:
            # Use BGR directly
            if len(img_uint8.shape) == 2:
                img_color = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
            else:
                img_color = img_uint8
        
        # Calculate histogram for each channel
        hist_features = []
        for i in range(3):
            hist = cv2.calcHist(
                [img_color], 
                [i], 
                None, 
                [self.bins], 
                [0, 256]
            )
            # Normalize histogram
            hist = hist / (hist.sum() + 1e-8)
            hist_features.append(hist.flatten())
        
        # Concatenate all channel histograms
        features = np.concatenate(hist_features).astype(np.float32)
        
        return features
    
    def get_feature_dim(self) -> int:
        """Get feature dimension."""
        return self.feature_dim
