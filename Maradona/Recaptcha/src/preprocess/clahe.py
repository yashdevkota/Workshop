"""
CLAHE (Contrast Limited Adaptive Histogram Equalization) module.
Applies adaptive histogram equalization to improve contrast.
"""
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from .utils import load_image, save_image, get_image_stats, ensure_dir


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, 
                tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Apply CLAHE to image.
    
    Args:
        image: Input image (grayscale or BGR)
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization
        
    Returns:
        Enhanced image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    # Apply CLAHE
    enhanced_gray = clahe.apply(gray)
    
    # If original was color, apply to each channel
    if len(image.shape) == 3:
        # Apply CLAHE to each channel separately
        enhanced = np.zeros_like(image)
        for i in range(3):
            clahe_channel = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            enhanced[:, :, i] = clahe_channel.apply(image[:, :, i])
        return enhanced
    else:
        return enhanced_gray


def get_adaptive_clahe_params(image: np.ndarray, 
                              default_clip_limit: float = 2.0,
                              default_tile_size: Tuple[int, int] = (8, 8),
                              brightness_threshold: float = 100) -> Tuple[float, Tuple[int, int]]:
    """
    Get adaptive CLAHE parameters based on image characteristics.
    
    Args:
        image: Input image
        default_clip_limit: Default clip limit
        default_tile_size: Default tile grid size
        brightness_threshold: Brightness threshold for adjustment
        
    Returns:
        Tuple of (clip_limit, tile_grid_size)
    """
    stats = get_image_stats(image)
    mean_brightness = stats['mean_brightness']
    height, width = stats['height'], stats['width']
    
    # Adjust clip_limit for dark images
    clip_limit = default_clip_limit
    if mean_brightness < brightness_threshold:
        clip_limit = min(3.0, default_clip_limit * 1.5)
    
    # Adjust tile size for small images
    tile_size = default_tile_size
    if min(height, width) < 100:
        tile_size = (4, 4)
    elif min(height, width) < 200:
        tile_size = (6, 6)
    
    return clip_limit, tile_size


def apply_clahe_dataset(input_dir: Path, output_dir: Path,
                       clip_limit: float = 2.0,
                       tile_grid_size: Tuple[int, int] = (8, 8),
                       adaptive: bool = True,
                       brightness_threshold: float = 100,
                       num_samples: int = 10, vis_dir: Path = None) -> dict:
    """
    Apply CLAHE to all images in dataset.
    
    Args:
        input_dir: Directory containing images
        output_dir: Directory to save processed images
        clip_limit: Default clip limit
        tile_grid_size: Default tile grid size
        adaptive: Whether to use adaptive parameters
        brightness_threshold: Brightness threshold for adaptive adjustment
        num_samples: Number of samples to visualize
        
    Returns:
        Dictionary with processing statistics
    """
    print(f"\n{'='*60}")
    print(f"Applying CLAHE (adaptive={adaptive})...")
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
    applied_params = []
    
    # Select random samples for visualization
    if len(image_paths) > num_samples:
        sample_indices = np.random.choice(len(image_paths), num_samples, replace=False)
    else:
        sample_indices = range(len(image_paths))
    
    for idx, img_path in enumerate(tqdm(image_paths, desc="Applying CLAHE")):
        img = load_image(img_path)
        if img is None:
            failed_count += 1
            continue
        
        # Get parameters
        if adaptive:
            cl, ts = get_adaptive_clahe_params(img, clip_limit, tile_grid_size, 
                                               brightness_threshold)
        else:
            cl, ts = clip_limit, tile_grid_size
        
        # Apply CLAHE
        enhanced = apply_clahe(img, cl, ts)
        
        # Save
        relative_path = img_path.relative_to(input_dir)
        output_path = output_dir / relative_path
        
        if save_image(enhanced, output_path):
            processed_count += 1
            
            # Collect samples for visualization
            if idx in sample_indices:
                sample_images.append((img, enhanced))
                sample_paths.append(relative_path)
                applied_params.append((cl, ts))
        else:
            failed_count += 1
    
    # Visualize results
    visualize_clahe_results(sample_images, sample_paths, applied_params, output_dir, vis_dir)
    
    stats = {
        'total_images': len(image_paths),
        'processed': processed_count,
        'failed': failed_count,
        'adaptive': adaptive,
        'default_clip_limit': clip_limit,
        'default_tile_size': tile_grid_size
    }
    
    print(f"\n✓ CLAHE complete:")
    print(f"  - Processed: {processed_count}/{len(image_paths)}")
    print(f"  - Failed: {failed_count}")
    
    return stats


def visualize_clahe_results(sample_images: List[Tuple], sample_paths: List[Path],
                            applied_params: List[Tuple], output_dir: Path, vis_dir: Path = None):
    """Visualize CLAHE results."""
    if vis_dir is None:
        vis_dir = output_dir / "visualizations"
    else:
        vis_dir = vis_dir / "clahe"
    ensure_dir(vis_dir)
    
    num_samples = len(sample_images)
    if num_samples == 0:
        return
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples * 3))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('CLAHE: Before → After', fontsize=14, fontweight='bold')
    
    for i, ((original, enhanced), path, (cl, ts)) in enumerate(zip(sample_images, sample_paths, applied_params)):
        # Original
        if len(original.shape) == 3:
            axes[i, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        else:
            axes[i, 0].imshow(original, cmap='gray')
        stats_orig = get_image_stats(original)
        axes[i, 0].set_title(f'Original\nBrightness: {stats_orig["mean_brightness"]:.1f}', 
                            fontsize=10)
        axes[i, 0].axis('off')
        
        # Enhanced
        if len(enhanced.shape) == 3:
            axes[i, 1].imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
        else:
            axes[i, 1].imshow(enhanced, cmap='gray')
        stats_enh = get_image_stats(enhanced)
        title = f'CLAHE Enhanced\nBrightness: {stats_enh["mean_brightness"]:.1f}\n'
        title += f'Params: clip={cl:.1f}, tile={ts[0]}x{ts[1]}'
        axes[i, 1].set_title(title, fontsize=10)
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(vis_dir / "clahe_comparison.png", 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved CLAHE visualization to {vis_dir / 'clahe_comparison.png'}")
