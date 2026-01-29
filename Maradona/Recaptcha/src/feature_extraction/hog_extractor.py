"""
HOG (Histogram of Oriented Gradients) feature extractor.
Extracts shape and edge information from images.
"""
from typing import Tuple
import numpy as np
import cv2


class HOGExtractor:
    """
    HOG feature extractor.
    
    Extracts gradient orientation histograms to capture shape information.
    Optimized for 112x112 images.
    """
    
    def __init__(self, 
                 cell_size: Tuple[int, int] = (8, 8),
                 block_size: Tuple[int, int] = (2, 2),
                 block_stride: Tuple[int, int] = (1, 1),
                 nbins: int = 9,
                 win_size: Tuple[int, int] = (112, 112)):
        """
        Initialize HOG extractor.
        
        Args:
            cell_size: Size of each cell in pixels (8x8 for 112x112 images)
            block_size: Number of cells per block (2x2 = 16x16 pixels)
            block_stride: Stride between blocks in cells (1x1 = 50% overlap)
            nbins: Number of orientation bins
            win_size: Window size (image size)
        """
        self.cell_size = cell_size
        self.block_size = block_size
        self.block_stride = block_stride
        self.nbins = nbins
        self.win_size = win_size
        
        # Calculate feature dimension
        # For 112x112 image with cell_size=(8,8), block_size=(2,2), block_stride=(1,1):
        # - Cells per dimension: 112/8 = 14
        # - Blocks per dimension: (14-2+1) = 13
        # - Total blocks: 13 * 13 = 169
        # - Features per block: 2*2*9 = 36
        # - Total features: 169 * 36 = 6084
        cells_per_dim = win_size[0] // cell_size[0]
        blocks_per_dim = cells_per_dim - block_size[0] + 1
        features_per_block = block_size[0] * block_size[1] * nbins
        self.feature_dim = blocks_per_dim * blocks_per_dim * features_per_block
        
        # Initialize HOG descriptor
        self.hog = cv2.HOGDescriptor(
            _winSize=win_size,
            _blockSize=(block_size[0] * cell_size[0], block_size[1] * cell_size[1]),
            _blockStride=(block_stride[0] * cell_size[0], block_stride[1] * cell_size[1]),
            _cellSize=cell_size,
            _nbins=nbins
        )
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract HOG features from image.
        
        Args:
            image: Input image (grayscale or color, float32, 0-1 range)
            
        Returns:
            HOG feature vector (float32)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Convert to uint8 (0-255) for HOG computation
        gray_uint8 = (gray * 255).astype(np.uint8)
        
        # Resize to expected window size if needed
        if gray_uint8.shape[:2] != self.win_size:
            gray_uint8 = cv2.resize(gray_uint8, self.win_size, interpolation=cv2.INTER_LINEAR)
        
        # Compute HOG features
        features = self.hog.compute(gray_uint8)
        
        # Flatten and normalize
        features = features.flatten().astype(np.float32)
        
        # L2 normalization for better performance
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        return features
    
    def get_feature_dim(self) -> int:
        """Get feature dimension."""
        return self.feature_dim
