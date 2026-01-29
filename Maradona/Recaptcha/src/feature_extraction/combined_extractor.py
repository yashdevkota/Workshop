"""
Combined feature extractor.
Combines all feature extractors into a single feature vector.
"""
from typing import Dict, Optional
import numpy as np

try:
    from .hog_extractor import HOGExtractor
    from .color_extractor import ColorHistogramExtractor
    from .lbp_extractor import LBPExtractor
    from .gradient_extractor import GradientExtractor
    from .texture_extractor import TextureExtractor
except ImportError:
    # Fallback for direct execution
    from src.feature_extraction.hog_extractor import HOGExtractor
    from src.feature_extraction.color_extractor import ColorHistogramExtractor
    from src.feature_extraction.lbp_extractor import LBPExtractor
    from src.feature_extraction.gradient_extractor import GradientExtractor
    from src.feature_extraction.texture_extractor import TextureExtractor


class CombinedExtractor:
    """
    Combined feature extractor.
    
    Combines HOG, Color Histogram, LBP, Gradient, and Texture features.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize Combined extractor.
        
        Args:
            config: Configuration dictionary with feature settings
        """
        self.config = config
        self.extractors = {}
        self.feature_dims = {}
        
        # Initialize extractors based on config
        features_config = config.get('features', {})
        
        if features_config.get('hog', {}).get('enabled', True):
            hog_config = features_config.get('hog', {})
            self.extractors['hog'] = HOGExtractor(
                cell_size=tuple(hog_config.get('cell_size', [8, 8])),
                block_size=tuple(hog_config.get('block_size', [2, 2])),
                block_stride=tuple(hog_config.get('block_stride', [1, 1])),
                nbins=hog_config.get('nbins', 9),
                win_size=tuple(hog_config.get('win_size', [112, 112]))
            )
            self.feature_dims['hog'] = self.extractors['hog'].get_feature_dim()
        
        if features_config.get('color_histogram', {}).get('enabled', True):
            color_config = features_config.get('color_histogram', {})
            self.extractors['color_hist'] = ColorHistogramExtractor(
                bins=color_config.get('bins', 32),
                color_space='hsv'
            )
            self.feature_dims['color_hist'] = self.extractors['color_hist'].get_feature_dim()
        
        if features_config.get('lbp', {}).get('enabled', True):
            lbp_config = features_config.get('lbp', {})
            self.extractors['lbp'] = LBPExtractor(
                num_points=lbp_config.get('num_points', 24),
                radius=lbp_config.get('radius', 3)
            )
            self.feature_dims['lbp'] = self.extractors['lbp'].get_feature_dim()
        
        if features_config.get('gradient', {}).get('enabled', True):
            gradient_config = features_config.get('gradient', {})
            self.extractors['gradient'] = GradientExtractor(
                num_bins=8,
                cell_size=(16, 16)
            )
            self.feature_dims['gradient'] = self.extractors['gradient'].get_feature_dim()
        
        if features_config.get('texture', {}).get('enabled', True):
            texture_config = features_config.get('texture', {})
            self.extractors['texture'] = TextureExtractor(
                distances=[1, 2, 3],
                angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                levels=256,
                properties=['contrast', 'correlation', 'energy', 'homogeneity']
            )
            self.feature_dims['texture'] = self.extractors['texture'].get_feature_dim()
        
        # Calculate total feature dimension
        self.feature_dim = sum(self.feature_dims.values())
    
    def extract(self, image: np.ndarray, grayscale_image: Optional[np.ndarray] = None, 
                hsv_image: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Extract combined features from image.
        
        Args:
            image: Input image (float32, 0-1 range) - 기본 RGB/BGR 이미지
            grayscale_image: Optional grayscale image (for HOG, LBP, Texture, Gradient)
            hsv_image: Optional HSV image (for Color Histogram)
            
        Returns:
            Combined feature vector (float32) - 각 특징이 개별적으로 정규화됨
        """
        import cv2
        
        features_list = []
        feature_names = []
        
        # 특징별 가중치 (형태 정보가 더 중요)
        feature_weights = {
            'hog': 1.5,
            'color_hist': 1.0,
            'lbp': 1.2,
            'gradient': 1.0,
            'texture': 1.1
        }
        
        # grayscale/hsv 이미지가 없으면 image에서 직접 변환 (fallback)
        if grayscale_image is None and len(image.shape) == 3:
            # RGB/BGR to Grayscale 변환
            img_uint8 = (image * 255).astype(np.uint8)
            if image.shape[2] == 3:
                gray_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2GRAY)
            else:
                gray_uint8 = img_uint8[:, :, 0]
            grayscale_image = (gray_uint8 / 255.0).astype(np.float32)
        elif grayscale_image is None and len(image.shape) == 2:
            # 이미 grayscale
            grayscale_image = image
        
        if hsv_image is None and len(image.shape) == 3:
            # RGB/BGR to HSV 변환
            img_uint8 = (image * 255).astype(np.uint8)
            if image.shape[2] == 3:
                hsv_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2HSV)
            else:
                hsv_uint8 = cv2.cvtColor(cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2HSV)
            hsv_image = (hsv_uint8 / 255.0).astype(np.float32)
        
        for name, extractor in self.extractors.items():
            try:
                # 최적 입력 이미지 선택
                if name in ['hog', 'lbp', 'texture', 'gradient']:
                    # Grayscale 이미지 사용
                    input_img = grayscale_image if grayscale_image is not None else image
                elif name == 'color_hist':
                    # HSV 이미지 사용
                    input_img = hsv_image if hsv_image is not None else image
                else:
                    input_img = image
                
                feat = extractor.extract(input_img)
                
                # 특징별 최적 정규화 전략 (Feature-wise Scaling)
                # 이 과정은 가중치 적용 및 결합 전에 각 특징의 스케일을 맞추는 핵심 단계입니다.
                if name == 'hog':
                    # HOG는 이미 L2 정규화됨 → 그대로 사용 (0-1 근처)
                    pass
                elif name in ['color_hist', 'lbp']:
                    # 확률 정규화된 히스토그램 → Min-Max로 0-1 범위 보장
                    feat_min = feat.min()
                    feat_max = feat.max()
                    if feat_max - feat_min > 1e-8:
                        feat = (feat - feat_min) / (feat_max - feat_min)
                    else:
                        feat = np.zeros_like(feat)
                elif name == 'gradient':
                    # 히스토그램 정규화됨 → L2 정규화 추가
                    norm = np.linalg.norm(feat)
                    if norm > 1e-8:
                        feat = feat / norm
                elif name == 'texture':
                    # GLCM 속성값 → StandardScaler 스타일 정규화
                    feat_mean = feat.mean()
                    feat_std = feat.std()
                    if feat_std > 1e-8:
                        feat = (feat - feat_mean) / feat_std
                    # 그 후 Min-Max로 0-1 범위로
                    feat_min = feat.min()
                    feat_max = feat.max()
                    if feat_max - feat_min > 1e-8:
                        feat = (feat - feat_min) / (feat_max - feat_min)
                
                # 특징 가중치 적용
                weight = feature_weights.get(name, 1.0)
                feat = feat * weight
                
                features_list.append(feat)
                feature_names.append(name)
            except Exception as e:
                print(f"Warning: Failed to extract {name} features: {e}")
                # Add zero vector as fallback
                features_list.append(np.zeros(self.feature_dims[name], dtype=np.float32))
                feature_names.append(name)
        
        # Concatenate all features
        if features_list:
            combined = np.concatenate(features_list).astype(np.float32)
        else:
            # Fallback: return zero vector
            combined = np.zeros(self.feature_dim, dtype=np.float32)
        
        return combined
    
    def get_feature_dim(self) -> int:
        """Get total feature dimension."""
        return self.feature_dim
    
    def get_feature_info(self) -> Dict:
        """
        Get information about each feature type.
        
        Returns:
            Dictionary with feature names and dimensions
        """
        return {
            'feature_dims': self.feature_dims,
            'total_dim': self.feature_dim,
            'extractors': list(self.extractors.keys())
        }
