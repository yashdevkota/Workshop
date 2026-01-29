"""
Gradient feature extractor.
Extracts gradient magnitude and direction information.
"""
from typing import Tuple
import numpy as np
import cv2


class GradientExtractor:
    """
    Gradient feature extractor.
    
    Extracts gradient magnitude and direction histograms.
    """
    
    def __init__(self, 
                 num_bins: int = 8,
                 cell_size: Tuple[int, int] = (16, 16)):
        """
        Initialize Gradient extractor.
        
        Args:
            num_bins: Number of bins for gradient direction histogram
            cell_size: Size of cells for spatial binning
        """
        self.num_bins = num_bins
        self.cell_size = cell_size
        
        # Feature dimension: num_bins * number of cells
        # For 112x112 image with cell_size=(16,16): 7*7 = 49 cells
        # Total: 8 * 49 = 392 features
        self.feature_dim = num_bins * (112 // cell_size[0]) * (112 // cell_size[1])
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract Gradient features from image.
        
        Args:
            image: Input image (grayscale or color, float32, 0-1 range)
            
        Returns:
            Gradient feature vector (float32)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image
        
        # Convert to uint8 (0-255) for gradient computation
        if gray.max() <= 1.0:
            gray_uint8 = (gray * 255).astype(np.uint8)
        else:
            gray_uint8 = gray.astype(np.uint8)
        
        # Resize to expected size if needed
        if gray_uint8.shape[:2] != (112, 112):
            gray_uint8 = cv2.resize(gray_uint8, (112, 112), interpolation=cv2.INTER_LINEAR)
        
        # Calculate gradients using Sobel operator
        grad_x = cv2.Sobel(gray_uint8, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_uint8, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate magnitude and direction
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)
        
        # Convert direction to 0-2π range and then to 0-360 degrees
        direction = np.degrees(direction) % 360
        
        # Divide image into cells
        h, w = gray_uint8.shape
        cell_h, cell_w = self.cell_size
        num_cells_h = h // cell_h
        num_cells_w = w // cell_w
        
        # Initialize feature vector
        features = []
        
        # Calculate histogram for each cell
        for i in range(num_cells_h):
            for j in range(num_cells_w):
                # Extract cell region
                y_start = i * cell_h
                y_end = (i + 1) * cell_h
                x_start = j * cell_w
                x_end = (j + 1) * cell_w
                
                cell_magnitude = magnitude[y_start:y_end, x_start:x_end]
                cell_direction = direction[y_start:y_end, x_start:x_end]
                
                # Calculate weighted histogram (weighted by magnitude)
                hist, _ = np.histogram(
                    cell_direction.ravel(),
                    bins=self.num_bins,
                    range=(0, 360),
                    weights=cell_magnitude.ravel()
                )
                
                # Normalize
                hist = hist / (hist.sum() + 1e-8)
                features.append(hist)
        
        # Concatenate all cell histograms
        features = np.concatenate(features).astype(np.float32)
        
        return features
    
    def get_feature_dim(self) -> int:
        """Get feature dimension."""
        return self.feature_dim
