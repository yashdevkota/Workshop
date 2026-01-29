"""
Utility functions for model training.
"""
import sys
from pathlib import Path
import numpy as np
import pickle
import json

# Add parent directory to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.preprocess.utils import load_config


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


def save_model(model, scaler, selector, output_path: Path) -> None:
    """Save model, scaler, and selector to pickle files."""
    ensure_dir(output_path.parent)
    
    with open(output_path / "model.pkl", 'wb') as f:
        pickle.dump(model, f)
    
    if scaler is not None:
        with open(output_path / "scaler.pkl", 'wb') as f:
            pickle.dump(scaler, f)
    
    if selector is not None:
        with open(output_path / "selector.pkl", 'wb') as f:
            pickle.dump(selector, f)


def load_model(model_path: Path):
    """Load model, scaler, and selector from pickle files."""
    with open(model_path / "model.pkl", 'rb') as f:
        model = pickle.load(f)
    
    scaler = None
    selector = None
    
    if (model_path / "scaler.pkl").exists():
        with open(model_path / "scaler.pkl", 'rb') as f:
            scaler = pickle.load(f)
    
    if (model_path / "selector.pkl").exists():
        with open(model_path / "selector.pkl", 'rb') as f:
            selector = pickle.load(f)
    
    return model, scaler, selector
