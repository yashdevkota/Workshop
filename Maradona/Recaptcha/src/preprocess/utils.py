"""
Utility functions for preprocessing module.
"""
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Generator
import numpy as np
import cv2
from PIL import Image
import pandas as pd


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    try:
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except ImportError:
        raise ImportError("pyyaml is required. Install with: pip install pyyaml")


def load_image(image_path: Path) -> Optional[np.ndarray]:
    """
    Load an image from file path.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Image as numpy array (BGR format) or None if loading fails
    """
    if not image_path.exists():
        return None
    
    img = cv2.imread(str(image_path))
    if img is None:
        # Try with PIL as fallback
        try:
            pil_img = Image.open(image_path)
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Warning: Failed to load {image_path}: {e}")
            return None
    
    return img


def image_generator(image_paths: List[Path]) -> Generator[Tuple[np.ndarray, Path], None, None]:
    """
    Generator that yields images one at a time for memory efficiency.
    
    Args:
        image_paths: List of image file paths
        
    Yields:
        Tuple of (image, image_path)
    """
    for img_path in image_paths:
        img = load_image(img_path)
        if img is not None:
            yield img, img_path
        # Image is automatically released from memory after yield


def save_image(image: np.ndarray, output_path: Path) -> bool:
    """
    Save an image to file.
    
    Args:
        image: Image as numpy array
        output_path: Path to save image
        
    Returns:
        True if successful, False otherwise
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), image)
        return True
    except Exception as e:
        print(f"Error saving image to {output_path}: {e}")
        return False


def get_image_stats(image: np.ndarray) -> Dict[str, float]:
    """
    Calculate image statistics.
    
    Args:
        image: Image as numpy array
        
    Returns:
        Dictionary with statistics (mean_brightness, std_dev, min, max, etc.)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    stats = {
        'mean_brightness': float(np.mean(gray)),
        'std_dev': float(np.std(gray)),
        'min': float(np.min(gray)),
        'max': float(np.max(gray)),
        'median': float(np.median(gray)),
        'height': int(image.shape[0]),
        'width': int(image.shape[1]),
        'channels': int(image.shape[2]) if len(image.shape) == 3 else 1
    }
    
    return stats


def create_metadata(image_paths: List[Path], labels: List[str], 
                   output_path: Path, stats: Optional[List[Dict]] = None) -> pd.DataFrame:
    """
    Create metadata DataFrame for processed images.
    
    Args:
        image_paths: List of image paths
        labels: List of labels
        output_path: Path to save metadata CSV
        stats: Optional list of image statistics
        
    Returns:
        DataFrame with metadata
    """
    data = {
        'image_path': [str(p) for p in image_paths],
        'label': labels
    }
    
    if stats:
        for key in stats[0].keys():
            data[key] = [s[key] for s in stats]
    
    df = pd.DataFrame(data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    return df


def save_label_mapping(label_mapping: Dict[str, int], output_path: Path):
    """Save label to integer mapping."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(label_mapping, f, indent=2)


def load_label_mapping(input_path: Path) -> Dict[str, int]:
    """Load label to integer mapping."""
    with open(input_path, 'r') as f:
        return json.load(f)


def ensure_dir(path: Path):
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)
