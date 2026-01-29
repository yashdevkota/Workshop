"""
Normalization module.
Normalizes pixel values to a specified range.
"""
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from .utils import load_image, save_image, ensure_dir


def normalize_image(image: np.ndarray, method: str = "minmax", 
                   range_min: float = 0.0, range_max: float = 1.0) -> np.ndarray:
    """
    Normalize image pixel values.
    
    Args:
        image: Input image
        method: Normalization method (minmax, standard)
        range_min: Minimum value for minmax normalization
        range_max: Maximum value for minmax normalization
        
    Returns:
        Normalized image (as float32)
    """
    img_float = image.astype(np.float32)
    
    if method == "minmax":
        # Min-Max normalization: (x - min) / (max - min) * (range_max - range_min) + range_min
        img_min = img_float.min()
        img_max = img_float.max()
        if img_max - img_min > 0:
            normalized = (img_float - img_min) / (img_max - img_min) * (range_max - range_min) + range_min
        else:
            normalized = img_float * 0 + range_min
    elif method == "standard":
        # Standard normalization: (x - mean) / std
        mean = img_float.mean()
        std = img_float.std()
        if std > 0:
            normalized = (img_float - mean) / std
        else:
            normalized = img_float * 0
    else:
        normalized = img_float
    
    return normalized


def normalize_dataset(input_dir: Path, output_dir: Path,
                     method: str = "minmax",
                     range_min: float = 0.0,
                     range_max: float = 1.0,
                     save_as_float: bool = True,
                     num_samples: int = 10, vis_dir: Path = None) -> dict:
    """
    Normalize all images in dataset.
    
    Args:
        input_dir: Directory containing images
        output_dir: Directory to save normalized images
        method: Normalization method
        range_min: Minimum value for minmax normalization
        range_max: Maximum value for minmax normalization
        save_as_float: Whether to save as float32 (True) or convert back to uint8 (False)
        num_samples: Number of samples to visualize
        
    Returns:
        Dictionary with processing statistics
    """
    print(f"\n{'='*60}")
    print(f"Normalizing images ({method}, range=[{range_min}, {range_max}])...")
    print(f"{'='*60}")
    
    # Find all images
    import os
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    image_paths = []
    
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if Path(f).suffix.lower() in image_extensions:
                image_paths.append(Path(root) / f)
    
    print(f"Found {len(image_paths)} images")
    
    # Process images
    processed_count = 0
    failed_count = 0
    sample_images = []
    sample_paths = []
    
    # Select random samples for visualization
    if len(image_paths) > num_samples:
        sample_indices = np.random.choice(len(image_paths), num_samples, replace=False)
    else:
        sample_indices = range(len(image_paths))
    
    for idx, img_path in enumerate(tqdm(image_paths, desc="Normalizing images")):
        img = load_image(img_path)
        if img is None:
            failed_count += 1
            continue
        
        # Normalize
        normalized = normalize_image(img, method, range_min, range_max)
        
        # Save
        relative_path = img_path.relative_to(input_dir)
        output_path = output_dir / relative_path
        
        if save_as_float:
            # Save as float32 (for numpy arrays)
            output_path = output_path.with_suffix('.npy')
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(output_path), normalized)
            processed_count += 1
        else:
            # Convert back to uint8 for image format
            if method == "minmax":
                normalized_uint8 = ((normalized - range_min) / (range_max - range_min) * 255).astype(np.uint8)
            else:
                # For standard normalization, scale to 0-255
                normalized_uint8 = np.clip((normalized + 3) * 255 / 6, 0, 255).astype(np.uint8)
            
            if save_image(normalized_uint8, output_path):
                processed_count += 1
            else:
                failed_count += 1
                continue
        
        # Collect samples for visualization
        if idx in sample_indices:
            sample_images.append((img, normalized))
            sample_paths.append(relative_path)
    
    # Visualize results
    visualize_normalization_results(sample_images, sample_paths, output_dir, method, vis_dir)
    
    stats = {
        'total_images': len(image_paths),
        'processed': processed_count,
        'failed': failed_count,
        'method': method,
        'range': [range_min, range_max],
        'save_as_float': save_as_float
    }
    
    print(f"\n✓ Normalization complete:")
    print(f"  - Processed: {processed_count}/{len(image_paths)}")
    print(f"  - Failed: {failed_count}")
    
    return stats


def visualize_normalization_results(sample_images: List[Tuple], sample_paths: List[Path],
                                   output_dir: Path, method: str, vis_dir: Path = None):
    """Visualize normalization results."""
    if vis_dir is None:
        vis_dir = output_dir / "visualizations"
    ensure_dir(vis_dir)
    
    num_samples = len(sample_images)
    if num_samples == 0:
        return
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples * 3))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f'Normalization: Before → After ({method})', 
                 fontsize=14, fontweight='bold')
    
    for i, ((original, normalized), path) in enumerate(zip(sample_images, sample_paths)):
        # Original
        if len(original.shape) == 3:
            axes[i, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        else:
            axes[i, 0].imshow(original, cmap='gray')
        axes[i, 0].set_title(f'Original\nRange: [{original.min()}, {original.max()}]', 
                            fontsize=10)
        axes[i, 0].axis('off')
        
        # Normalized (convert to uint8 for display)
        if method == "minmax":
            normalized_display = ((normalized - normalized.min()) / 
                                (normalized.max() - normalized.min() + 1e-8) * 255).astype(np.uint8)
        else:
            normalized_display = np.clip((normalized + 3) * 255 / 6, 0, 255).astype(np.uint8)
        
        if len(normalized.shape) == 3:
            axes[i, 1].imshow(cv2.cvtColor(normalized_display, cv2.COLOR_BGR2RGB))
        else:
            axes[i, 1].imshow(normalized_display, cmap='gray')
        axes[i, 1].set_title(f'Normalized\nRange: [{normalized.min():.3f}, {normalized.max():.3f}]', 
                            fontsize=10)
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(vis_dir / "normalization_comparison.png", 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved normalization visualization to {vis_dir / 'normalization_comparison.png'}")
