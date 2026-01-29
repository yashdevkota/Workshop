"""
Utility functions for feature selection.
"""
import sys
from pathlib import Path
from typing import Dict
import numpy as np
import pickle
import json
import yaml

# Add parent directory to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def load_config(config_path: Path) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def ensure_dir(path: Path) -> None:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)


def load_features(features_dir: Path, split: str = "train") -> tuple:
    """
    Load features and labels from .npy files.
    
    Args:
        features_dir: Directory containing feature files
        split: Split name (train/val/test)
        
    Returns:
        Tuple of (features, labels, label_mapping)
    """
    features_file = features_dir / f"{split}_features.npy"
    labels_file = features_dir / f"{split}_labels.npy"
    mapping_file = features_dir / f"{split}_label_mapping.json"
    
    if not features_file.exists():
        raise FileNotFoundError(f"Features file not found: {features_file}")
    if not labels_file.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_file}")
    
    features = np.load(str(features_file))
    labels = np.load(str(labels_file))
    
    label_mapping = {}
    if mapping_file.exists():
        with open(mapping_file, 'r') as f:
            label_mapping = json.load(f)
    
    return features, labels, label_mapping


def save_selector(selector, output_path: Path) -> None:
    """Save feature selector to pickle file."""
    ensure_dir(output_path.parent)
    with open(output_path, 'wb') as f:
        pickle.dump(selector, f)


def load_selector(selector_path: Path):
    """Load feature selector from pickle file."""
    with open(selector_path, 'rb') as f:
        return pickle.load(f)
