"""
Utility functions for feature extraction.
Generator pattern for memory-efficient image loading.
"""
from pathlib import Path
from typing import Generator, Tuple, List, Optional
import numpy as np
import cv2


def load_image_npy(image_path: Path) -> Optional[np.ndarray]:
    """
    Load image from .npy file (normalized image).
    
    Args:
        image_path: Path to .npy file
        
    Returns:
        Image array (float32, 0-1 range) or None if failed
    """
    try:
        image = np.load(str(image_path))
        return image.astype(np.float32)
    except Exception as e:
        print(f"Warning: Failed to load {image_path}: {e}")
        return None


def load_image_file(image_path: Path) -> Optional[np.ndarray]:
    """
    Load image from PNG/JPG file and normalize to 0-1 range.
    
    Args:
        image_path: Path to image file (.png, .jpg, etc.)
        
    Returns:
        Image array (float32, 0-1 range) or None if failed
    """
    try:
        import cv2
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        # BGR to RGB (cv2는 BGR로 읽음)
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Normalize to 0-1 range
        image = image.astype(np.float32) / 255.0
        return image
    except Exception as e:
        print(f"Warning: Failed to load {image_path}: {e}")
        return None


def image_generator(image_paths: List[Path], 
                   grayscale_paths: Optional[List[Path]] = None,
                   hsv_paths: Optional[List[Path]] = None) -> Generator[Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Path], None, None]:
    """
    Generator that loads images one at a time for memory efficiency.
    
    Args:
        image_paths: List of image paths (.npy files) - 기본 RGB/BGR 이미지
        grayscale_paths: Optional list of grayscale image paths (can contain None)
        hsv_paths: Optional list of HSV image paths (can contain None)
        
    Yields:
        Tuple of (image, grayscale_image, hsv_image, image_path)
    """
    for idx, img_path in enumerate(image_paths):
        img = load_image_npy(img_path)
        if img is None:
            continue
        
        # Load grayscale if available and path is not None
        gray_img = None
        if grayscale_paths and idx < len(grayscale_paths) and grayscale_paths[idx] is not None:
            gray_path = grayscale_paths[idx]
            # .npy 파일이면 load_image_npy, 아니면 load_image_file
            if gray_path.suffix == '.npy':
                gray_img = load_image_npy(gray_path)
            else:
                gray_img = load_image_file(gray_path)
                # Grayscale 이미지는 2D일 수 있으므로 확인
                if gray_img is not None and len(gray_img.shape) == 2:
                    # 2D grayscale 이미지는 그대로 사용
                    pass
                elif gray_img is not None and len(gray_img.shape) == 3:
                    # 3D 이미지는 grayscale로 변환 (RGB -> Grayscale)
                    import cv2
                    gray_uint8 = (gray_img * 255).astype(np.uint8)
                    gray_uint8 = cv2.cvtColor(gray_uint8, cv2.COLOR_RGB2GRAY)
                    gray_img = (gray_uint8 / 255.0).astype(np.float32)
        
        # Load HSV if available and path is not None
        hsv_img = None
        if hsv_paths and idx < len(hsv_paths) and hsv_paths[idx] is not None:
            hsv_path = hsv_paths[idx]
            # .npy 파일이면 load_image_npy, 아니면 load_image_file
            if hsv_path.suffix == '.npy':
                hsv_img = load_image_npy(hsv_path)
            else:
                hsv_img = load_image_file(hsv_path)
                # HSV 이미지는 BGR로 저장되었을 수 있으므로 HSV로 변환
                if hsv_img is not None and len(hsv_img.shape) == 3:
                    import cv2
                    hsv_uint8 = (hsv_img * 255).astype(np.uint8)
                    # RGB를 BGR로 변환 후 HSV로
                    hsv_uint8 = cv2.cvtColor(hsv_uint8, cv2.COLOR_RGB2BGR)
                    hsv_uint8 = cv2.cvtColor(hsv_uint8, cv2.COLOR_BGR2HSV)
                    hsv_img = (hsv_uint8 / 255.0).astype(np.float32)
        
        yield img, gray_img, hsv_img, img_path
        # Memory is automatically freed after yield


def ensure_dir(path: Path) -> None:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)


def collect_image_paths(data_dir: Path, split: Optional[str] = None) -> Tuple[List[Path], List[str]]:
    """
    Collect all image paths and labels from processed directory.
    
    Args:
        data_dir: Directory containing processed images
        split: Optional split name (train/val/test)
        
    Returns:
        Tuple of (image_paths, labels)
    """
    image_paths = []
    labels = []
    
    # If split is specified, look in that subdirectory
    if split:
        search_dir = data_dir / split
        if not search_dir.exists():
            # Split directory doesn't exist, use data_dir directly
            search_dir = data_dir
    else:
        search_dir = data_dir
    
    if not search_dir.exists():
        return image_paths, labels
    
    # Walk through directory structure
    for img_path in search_dir.rglob("*.npy"):
        # Extract label from path (class name is parent directory)
        relative_path = img_path.relative_to(search_dir)
        # Find class name (usually the last directory before filename)
        # Handle nested paths like: google-recaptcha-v2/images/Bicycle/image.npy
        parts = relative_path.parts
        if len(parts) >= 2:
            # Look for class name (usually after 'images' directory)
            label = None
            for i, part in enumerate(parts):
                if part == 'images' and i + 1 < len(parts) - 1:  # -1 to exclude filename
                    label = parts[i + 1]
                    break
            if label is None:
                # Fallback: use the directory before filename
                label = parts[-2] if len(parts) > 1 else "unknown"
        else:
            label = "unknown"
        
        image_paths.append(img_path)
        labels.append(label)
    
    return image_paths, labels


def save_features_batch(features: np.ndarray, labels: np.ndarray, 
                       output_path: Path, split: str) -> None:
    """
    Save extracted features and labels to .npy files.
    
    Args:
        features: Feature array (N, feature_dim)
        labels: Label array (N,)
        output_path: Output directory
        split: Split name (train/val/test)
    """
    ensure_dir(output_path)
    
    features_file = output_path / f"{split}_features.npy"
    labels_file = output_path / f"{split}_labels.npy"
    
    np.save(str(features_file), features.astype(np.float32))
    np.save(str(labels_file), labels.astype(np.int32))
    
    print(f"✓ Saved {len(features)} features to {features_file}")


def load_cached_features(output_path: Path, split: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load cached features from .npy files.
    
    Args:
        output_path: Directory containing cached features
        split: Split name (train/val/test)
        
    Returns:
        Tuple of (features, labels)
    """
    features_file = output_path / f"{split}_features.npy"
    labels_file = output_path / f"{split}_labels.npy"
    
    features = np.load(str(features_file))
    labels = np.load(str(labels_file))
    
    return features, labels


def check_cache_exists(output_path: Path, split: str) -> bool:
    """
    Check if cached features exist.
    
    Args:
        output_path: Directory to check
        split: Split name (train/val/test)
        
    Returns:
        True if cache exists
    """
    features_file = output_path / f"{split}_features.npy"
    labels_file = output_path / f"{split}_labels.npy"
    
    return features_file.exists() and labels_file.exists()
