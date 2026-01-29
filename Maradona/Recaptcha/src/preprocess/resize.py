"""
Image resizing module.
Resizes all images to a fixed target size.
"""
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from .utils import load_image, save_image, get_image_stats, ensure_dir


def resize_image(image: np.ndarray, target_size: Tuple[int, int], 
                method: str = "bilinear") -> np.ndarray:
    """
    Resize image to target size.
    
    Args:
        image: Input image as numpy array
        target_size: Target (width, height)
        method: Resize method (bilinear, nearest, cubic)
        
    Returns:
        Resized image
    """
    if method == "bilinear":
        interpolation = cv2.INTER_LINEAR
    elif method == "nearest":
        interpolation = cv2.INTER_NEAREST
    elif method == "cubic":
        interpolation = cv2.INTER_CUBIC
    else:
        interpolation = cv2.INTER_LINEAR
    
    resized = cv2.resize(image, target_size, interpolation=interpolation)
    return resized


def resize_dataset(input_dir: Path, output_dir: Path, 
                  target_size: Tuple[int, int], method: str = "bilinear",
                  num_samples: int = 10, vis_dir: Path = None) -> dict:
    """
    Resize all images in dataset to target size.
    
    Args:
        input_dir: Directory containing raw images
        output_dir: Directory to save resized images
        target_size: Target (width, height)
        method: Resize method
        num_samples: Number of samples to visualize
        
    Returns:
        Dictionary with processing statistics
    """
    print(f"\n{'='*60}")
    print(f"Resizing images to {target_size[0]}x{target_size[1]}...")
    print(f"{'='*60}")
    
    # Find all images
    import os
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    image_paths = []
    
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if Path(f).suffix.lower() in image_extensions:
                image_paths.append(Path(root) / f)
    
    print(f"Found {len(image_paths)} images to resize")
    
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
    
    # Use generator pattern for memory efficiency
    from .utils import image_generator
    
    for idx, (img, img_path) in enumerate(tqdm(image_generator(image_paths), 
                                                total=len(image_paths), 
                                                desc="Resizing images")):
        if img is None:
            failed_count += 1
            continue
        
        # Resize
        resized = resize_image(img, target_size, method)
        
        # Calculate relative path and save
        relative_path = img_path.relative_to(input_dir)
        output_path = output_dir / relative_path
        
        if save_image(resized, output_path):
            processed_count += 1
            
            # Collect samples for visualization
            if idx in sample_indices:
                sample_images.append((img, resized))
                sample_paths.append(relative_path)
        else:
            failed_count += 1
    
    # Visualize results
    visualize_resize_results(sample_images, sample_paths, output_dir, target_size, vis_dir)
    
    stats = {
        'total_images': len(image_paths),
        'processed': processed_count,
        'failed': failed_count,
        'target_size': target_size,
        'method': method
    }
    
    print(f"\n✓ Resizing complete:")
    print(f"  - Processed: {processed_count}/{len(image_paths)}")
    print(f"  - Failed: {failed_count}")
    
    return stats


def visualize_resize_results(sample_images: List[Tuple], sample_paths: List[Path],
                            output_dir: Path, target_size: Tuple[int, int], vis_dir: Path = None):
    """Visualize resize results with before/after comparison."""
    if vis_dir is None:
        vis_dir = output_dir / "visualizations"
    else:
        vis_dir = vis_dir / "resize"
    ensure_dir(vis_dir)
    
    num_samples = len(sample_images)
    if num_samples == 0:
        return
    
    # Create comparison grid
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples * 3))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f'Image Resizing: Before → After ({target_size[0]}x{target_size[1]})', 
                 fontsize=14, fontweight='bold')
    
    for i, ((original, resized), path) in enumerate(zip(sample_images, sample_paths)):
        # Original
        if len(original.shape) == 3:
            axes[i, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        else:
            axes[i, 0].imshow(original, cmap='gray')
        axes[i, 0].set_title(f'Original\n{original.shape[1]}x{original.shape[0]}', 
                            fontsize=10)
        axes[i, 0].axis('off')
        
        # Resized
        if len(resized.shape) == 3:
            axes[i, 1].imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        else:
            axes[i, 1].imshow(resized, cmap='gray')
        axes[i, 1].set_title(f'Resized\n{resized.shape[1]}x{resized.shape[0]}', 
                            fontsize=10)
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(vis_dir / "resize_comparison.png", 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved resize visualization to {vis_dir / 'resize_comparison.png'}")
