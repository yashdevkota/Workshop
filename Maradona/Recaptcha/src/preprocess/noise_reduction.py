"""
Noise reduction module.
Applies Gaussian blur or bilateral filter to reduce noise.
"""
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from .utils import load_image, save_image, get_image_stats, ensure_dir


def apply_gaussian_blur(image: np.ndarray, kernel_size: Tuple[int, int] = (3, 3), sigma: float = None) -> np.ndarray:
    """
    Apply Gaussian blur to reduce noise.
    
    Args:
        image: Input image
        kernel_size: Gaussian kernel size (must be odd)
        sigma: Gaussian kernel standard deviation. If None, calculated from kernel size.
        
    Returns:
        Denoised image
    """
    # Ensure kernel size is odd
    ksize = (kernel_size[0] | 1, kernel_size[1] | 1)
    # Calculate sigma from kernel size if not provided
    # Rule of thumb: sigma = 0.3 * ((ksize-1)*0.5 - 1) + 0.8
    if sigma is None:
        sigma = 0.3 * ((ksize[0] - 1) * 0.5 - 1) + 0.8
    return cv2.GaussianBlur(image, ksize, sigma)


def apply_bilateral_filter(image: np.ndarray, d: int = 9, 
                          sigma_color: float = 75, 
                          sigma_space: float = 75) -> np.ndarray:
    """
    Apply bilateral filter to reduce noise while preserving edges.
    
    Args:
        image: Input image
        d: Diameter of pixel neighborhood
        sigma_color: Filter sigma in the color space
        sigma_space: Filter sigma in the coordinate space
        
    Returns:
        Denoised image
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def calculate_noise_level(image: np.ndarray) -> float:
    """
    Calculate actual noise level using Laplacian variance method.
    Higher variance = more noise.
    
    Args:
        image: Input image
        
    Returns:
        Noise level (Laplacian variance)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Calculate Laplacian (second derivative)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    # Variance of Laplacian is a good noise indicator
    noise_level = laplacian.var()
    return noise_level


def should_apply_noise_reduction(image: np.ndarray, threshold: float = 100.0) -> bool:
    """
    Determine if noise reduction should be applied based on noise level.
    Uses Laplacian variance method for more accurate noise detection.
    
    Args:
        image: Input image
        threshold: Noise threshold (Laplacian variance)
        
    Returns:
        True if noise reduction should be applied
    """
    noise_level = calculate_noise_level(image)
    return noise_level > threshold


def reduce_noise_dataset(input_dir: Path, output_dir: Path,
                        method: str = "gaussian",
                        threshold: float = 15.0,
                        gaussian_kernel: Tuple[int, int] = (3, 3),
                        gaussian_sigma: float = 1.0,
                        bilateral_params: Optional[dict] = None,
                        num_samples: int = 10, vis_dir: Path = None) -> dict:
    """
    Apply noise reduction to all images in dataset.
    
    Args:
        input_dir: Directory containing images
        output_dir: Directory to save processed images
        method: Noise reduction method (gaussian, bilateral)
        threshold: Noise threshold for conditional application
        gaussian_kernel: Kernel size for Gaussian blur
        bilateral_params: Parameters for bilateral filter
        num_samples: Number of samples to visualize
        
    Returns:
        Dictionary with processing statistics
    """
    print(f"\n{'='*60}")
    print(f"Applying noise reduction ({method})...")
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
    
    for idx, img_path in enumerate(tqdm(image_paths, desc="Reducing noise")):
        img = load_image(img_path)
        if img is None:
            failed_count += 1
            continue
        
        # Check if noise reduction is needed
        should_apply = should_apply_noise_reduction(img, threshold)
        
        if should_apply:
            if method == "gaussian":
                # Use configurable sigma for better noise reduction
                denoised = apply_gaussian_blur(img, gaussian_kernel, sigma=gaussian_sigma)
            elif method == "bilateral":
                params = bilateral_params or {'d': 9, 'sigma_color': 75, 'sigma_space': 75}
                denoised = apply_bilateral_filter(img, **params)
            else:
                denoised = img
            applied_count += 1
        else:
            denoised = img
            skipped_count += 1
        
        # Save
        relative_path = img_path.relative_to(input_dir)
        output_path = output_dir / relative_path
        
        if save_image(denoised, output_path):
            processed_count += 1
            
            # Collect samples for visualization
            if idx in sample_indices:
                sample_images.append((img, denoised, should_apply))
                sample_paths.append(relative_path)
        else:
            failed_count += 1
    
    # Visualize results
    visualize_noise_reduction_results(sample_images, sample_paths, output_dir, method, vis_dir)
    
    stats = {
        'total_images': len(image_paths),
        'processed': processed_count,
        'noise_reduction_applied': applied_count,
        'noise_reduction_skipped': skipped_count,
        'failed': failed_count,
        'method': method,
        'threshold': threshold
    }
    
    print(f"\n✓ Noise reduction complete:")
    print(f"  - Processed: {processed_count}/{len(image_paths)}")
    print(f"  - Noise reduction applied: {applied_count}")
    print(f"  - Skipped (low noise): {skipped_count}")
    
    return stats


def visualize_noise_reduction_results(sample_images: List[Tuple], sample_paths: List[Path],
                                     output_dir: Path, method: str, vis_dir: Path = None):
    """Visualize noise reduction results."""
    if vis_dir is None:
        vis_dir = output_dir / "visualizations"
    else:
        vis_dir = vis_dir / "noise_reduction"
    ensure_dir(vis_dir)
    
    num_samples = len(sample_images)
    if num_samples == 0:
        return
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples * 3))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f'Noise Reduction: Before → After ({method})', 
                 fontsize=14, fontweight='bold')
    
    for i, ((original, denoised, applied), path) in enumerate(zip(sample_images, sample_paths)):
        # Original
        if len(original.shape) == 3:
            axes[i, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        else:
            axes[i, 0].imshow(original, cmap='gray')
        noise_orig = calculate_noise_level(original)
        title = f'Original\nNoise: {noise_orig:.1f}'
        if applied:
            title += ' → Applied'
        axes[i, 0].set_title(title, fontsize=10)
        axes[i, 0].axis('off')
        
        # Denoised
        if len(denoised.shape) == 3:
            axes[i, 1].imshow(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB))
        else:
            axes[i, 1].imshow(denoised, cmap='gray')
        noise_denoised = calculate_noise_level(denoised)
        axes[i, 1].set_title(f'Denoised\nNoise: {noise_denoised:.1f}', 
                            fontsize=10)
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(vis_dir / "noise_reduction_comparison.png", 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved noise reduction visualization to {vis_dir / 'noise_reduction_comparison.png'}")
