"""
Texture (GLCM) feature extractor.
Extracts texture information using Gray-Level Co-occurrence Matrix.
"""
from typing import Tuple, List
import numpy as np
from skimage.feature import graycomatrix, graycoprops


class TextureExtractor:
    """
    Texture feature extractor using GLCM (Gray-Level Co-occurrence Matrix).
    
    Extracts texture properties like contrast, correlation, energy, homogeneity.
    """
    
    def __init__(self, 
                 distances: List[int] = [1, 2, 3],
                 angles: List[float] = [0, np.pi/4, np.pi/2, 3*np.pi/4],
                 levels: int = 256,
                 properties: List[str] = ['contrast', 'correlation', 'energy', 'homogeneity']):
        """
        Initialize Texture extractor.
        
        Args:
            distances: List of pixel pair distances
            angles: List of angles in radians
            levels: Number of gray-levels for quantization
            properties: List of GLCM properties to extract
        """
        self.distances = distances
        self.angles = angles
        self.levels = levels
        self.properties = properties
        
        # Feature dimension: num_distances * num_angles * num_properties
        self.feature_dim = len(distances) * len(angles) * len(properties)
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract Texture features from image.
        
        Args:
            image: Input image (grayscale or color, float32, 0-1 range)
            
        Returns:
            Texture feature vector (float32)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image
        
        # Convert to uint8 (0-255) and quantize
        if gray.max() <= 1.0:
            gray_uint8 = (gray * 255).astype(np.uint8)
        else:
            gray_uint8 = gray.astype(np.uint8)
        
        # Resize to expected size if needed
        if gray_uint8.shape[:2] != (112, 112):
            gray_uint8 = cv2.resize(gray_uint8, (112, 112), interpolation=cv2.INTER_LINEAR)
        
        # Quantize to reduce computation (optional, but recommended for speed)
        # Reduce to 32 levels for faster computation
        quantized = (gray_uint8 // (256 // 32)).astype(np.uint8)
        
        # Calculate GLCM
        glcm = graycomatrix(
            quantized,
            distances=self.distances,
            angles=self.angles,
            levels=32,  # Match quantization
            symmetric=True,
            normed=True
        )
        
        # Extract properties
        features = []
        for prop in self.properties:
            prop_values = graycoprops(glcm, prop)
            # Flatten and add to features
            features.append(prop_values.flatten())
        
        # Concatenate all properties
        features = np.concatenate(features).astype(np.float32)
        
        return features
    
    def get_feature_dim(self) -> int:
        """Get feature dimension."""
        return self.feature_dim
