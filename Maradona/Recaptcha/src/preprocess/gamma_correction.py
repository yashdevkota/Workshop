"""
Gamma correction module.
Applies gamma correction to adjust image brightness non-linearly.
"""
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from .utils import load_image, save_image, get_image_stats, ensure_dir


def apply_gamma_correction(image: np.ndarray, gamma: float = 0.8) -> np.ndarray:
    """
    Apply gamma correction to image.
    
    Args:
        image: Input image
        gamma: Gamma value (< 1 brightens, > 1 darkens)
        
    Returns:
        Gamma-corrected image
    """
    # Build lookup table
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 
                      for i in np.arange(0, 256)]).astype("uint8")
    
    # Apply lookup table
    if len(image.shape) == 3:
        corrected = cv2.LUT(image, table)
    else:
        corrected = cv2.LUT(image, table)
    
    return corrected


def should_apply_gamma(image: np.ndarray, threshold: float = 100.0) -> bool:
    """
    Determine if gamma correction should be applied based on brightness.
    
    Args:
        image: Input image
        threshold: Brightness threshold
        
    Returns:
        True if gamma correction should be applied
    """
    stats = get_image_stats(image)
    return stats['mean_brightness'] < threshold


def apply_gamma_correction_dataset(input_dir: Path, output_dir: Path,
                                  gamma: float = 0.8,
                                  threshold: float = 100.0,
                                  num_samples: int = 10, vis_dir: Path = None) -> dict:
    """
    Apply gamma correction to all images in dataset.
    
    Args:
        input_dir: Directory containing images
        output_dir: Directory to save processed images
        gamma: Gamma value (< 1 brightens dark images)
        threshold: Brightness threshold for conditional application
        num_samples: Number of samples to visualize
        
    Returns:
        Dictionary with processing statistics
    """
    print(f"\n{'='*60}")
    print(f"Applying gamma correction (gamma={gamma}, threshold={threshold})...")
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
    applied_count = 0
    skipped_count = 0
    failed_count = 0
    sample_images = []
    sample_paths = []
    
    # Select random samples for visualization
    if len(image_paths) > num_samples:
        sample_indices = np.random.choice(len(image_paths), num_samples, replace=False)
    else:
        sample_indices = range(len(image_paths))
    
    for idx, img_path in enumerate(tqdm(image_paths, desc="Applying gamma correction")):
        img = load_image(img_path)
        if img is None:
            failed_count += 1
            continue
        
        # Check if gamma correction is needed
        should_apply = should_apply_gamma(img, threshold)
        
        if should_apply:
            corrected = apply_gamma_correction(img, gamma)
            applied_count += 1
        else:
            corrected = img
            skipped_count += 1
        
        # Save
        relative_path = img_path.relative_to(input_dir)
        output_path = output_dir / relative_path
        
        if save_image(corrected, output_path):
            processed_count += 1
            
            # Collect samples for visualization
            if idx in sample_indices:
                sample_images.append((img, corrected, should_apply))
                sample_paths.append(relative_path)
        else:
            failed_count += 1
    
    # Visualize results
    visualize_gamma_correction_results(sample_images, sample_paths, output_dir, gamma, vis_dir)
    
    stats = {
        'total_images': len(image_paths),
        'processed': processed_count,
        'gamma_applied': applied_count,
        'gamma_skipped': skipped_count,
        'failed': failed_count,
        'gamma': gamma,
        'threshold': threshold
    }
    
    print(f"\n✓ Gamma correction complete:")
    print(f"  - Processed: {processed_count}/{len(image_paths)}")
    print(f"  - Gamma correction applied: {applied_count}")
    print(f"  - Skipped (bright enough): {skipped_count}")
    
    return stats


def visualize_gamma_correction_results(sample_images: List[Tuple], sample_paths: List[Path],
                                     output_dir: Path, gamma: float, vis_dir: Path = None):
    """Visualize gamma correction results."""
    if vis_dir is None:
        vis_dir = output_dir / "visualizations"
    ensure_dir(vis_dir)
    
    num_samples = len(sample_images)
    if num_samples == 0:
        return
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples * 3))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f'Gamma Correction: Before → After (γ={gamma})', 
                 fontsize=14, fontweight='bold')
    
    for i, ((original, corrected, applied), path) in enumerate(zip(sample_images, sample_paths)):
        # Original
        if len(original.shape) == 3:
            axes[i, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        else:
            axes[i, 0].imshow(original, cmap='gray')
        stats_orig = get_image_stats(original)
        title = f'Original\nBrightness: {stats_orig["mean_brightness"]:.1f}'
        if applied:
            title += ' → Applied'
        axes[i, 0].set_title(title, fontsize=10)
        axes[i, 0].axis('off')
        
        # Corrected
        if len(corrected.shape) == 3:
            axes[i, 1].imshow(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))
        else:
            axes[i, 1].imshow(corrected, cmap='gray')
        stats_corr = get_image_stats(corrected)
        axes[i, 1].set_title(f'Gamma Corrected\nBrightness: {stats_corr["mean_brightness"]:.1f}', 
                            fontsize=10)
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(vis_dir / "gamma_correction_comparison.png", 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved gamma correction visualization to {vis_dir / 'gamma_correction_comparison.png'}")
