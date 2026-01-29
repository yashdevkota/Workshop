"""
LBP (Local Binary Patterns) feature extractor.
Extracts texture information from images.
"""
from typing import Tuple
import numpy as np
from skimage import feature


class LBPExtractor:
    """
    LBP feature extractor.
    
    Extracts local binary patterns to capture texture information.
    """
    
    def __init__(self, 
                 num_points: int = 24,
                 radius: int = 3,
                 method: str = 'uniform'):
        """
        Initialize LBP extractor.
        
        Args:
            num_points: Number of circularly symmetric neighbor set points
            radius: Radius of circle (in pixels)
            method: Method to use ('default', 'ror', 'uniform', 'var')
        """
        self.num_points = num_points
        self.radius = radius
        self.method = method
        
        # Feature dimension: number of bins in histogram
        # uniform method: num_points + 2 (for non-uniform patterns)
        if method == 'uniform':
            self.feature_dim = num_points + 2
        else:
            # For other methods, use 2^num_points bins
            self.feature_dim = 2 ** num_points
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract LBP features from image.
        
        Args:
            image: Input image (grayscale or color, float32, 0-1 range)
            
        Returns:
            LBP feature vector (float32, normalized histogram)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image
        
        # Convert to uint8 (0-255) for LBP computation
        if gray.max() <= 1.0:
            gray_uint8 = (gray * 255).astype(np.uint8)
        else:
            gray_uint8 = gray.astype(np.uint8)
        
        # Compute LBP
        lbp = feature.local_binary_pattern(
            gray_uint8,
            self.num_points,
            self.radius,
            method=self.method
        )
        
        # Calculate histogram
        if self.method == 'uniform':
            # For uniform method, use num_points + 2 bins
            hist, _ = np.histogram(
                lbp.ravel(),
                bins=self.num_points + 2,
                range=(0, self.num_points + 2)
            )
        else:
            # For other methods, use 2^num_points bins
            hist, _ = np.histogram(
                lbp.ravel(),
                bins=2 ** self.num_points,
                range=(0, 2 ** self.num_points)
            )
        
        # Normalize histogram
        hist = hist.astype(np.float32)
        hist = hist / (hist.sum() + 1e-8)
        
        return hist
    
    def get_feature_dim(self) -> int:
        """Get feature dimension."""
        return self.feature_dim
