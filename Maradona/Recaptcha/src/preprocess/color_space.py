"""
Color space conversion module.
Converts images to HSV, Lab, and Grayscale color spaces.
"""
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from .utils import load_image, save_image, ensure_dir


def convert_to_hsv(image: np.ndarray) -> np.ndarray:
    """Convert BGR image to HSV."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def convert_to_lab(image: np.ndarray) -> np.ndarray:
    """Convert BGR image to Lab."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert BGR image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def convert_color_spaces_dataset(input_dir: Path, output_dir: Path,
                                convert_hsv: bool = True,
                                convert_lab: bool = False,
                                convert_grayscale: bool = True,
                                num_samples: int = 10, vis_dir: Path = None) -> dict:
    """
    Convert all images to specified color spaces.
    
    Args:
        input_dir: Directory containing images
        output_dir: Base directory to save converted images
        convert_hsv: Whether to convert to HSV
        convert_lab: Whether to convert to Lab
        convert_grayscale: Whether to convert to grayscale
        num_samples: Number of samples to visualize
        
    Returns:
        Dictionary with processing statistics
    """
    print(f"\n{'='*60}")
    print("Converting color spaces...")
    print(f"  - HSV: {convert_hsv}")
    print(f"  - Lab: {convert_lab}")
    print(f"  - Grayscale: {convert_grayscale}")
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
    
    for idx, img_path in enumerate(tqdm(image_paths, desc="Converting color spaces")):
        img = load_image(img_path)
        if img is None:
            failed_count += 1
            continue
        
        # Skip if already grayscale
        if len(img.shape) == 2:
            continue
        
        relative_path = img_path.relative_to(input_dir)
        saved_any = False
        
        # Convert and save
        if convert_hsv:
            hsv = convert_to_hsv(img)
            output_path = output_dir / "hsv" / relative_path
            if save_image(hsv, output_path):
                saved_any = True
        
        if convert_lab:
            lab = convert_to_lab(img)
            output_path = output_dir / "lab" / relative_path
            if save_image(lab, output_path):
                saved_any = True
        
        if convert_grayscale:
            gray = convert_to_grayscale(img)
            output_path = output_dir / "grayscale" / relative_path
            if save_image(gray, output_path):
                saved_any = True
        
        if saved_any:
            processed_count += 1
            
            # Collect samples for visualization
            if idx in sample_indices:
                conversions = {}
                if convert_hsv:
                    conversions['hsv'] = convert_to_hsv(img)
                if convert_lab:
                    conversions['lab'] = convert_to_lab(img)
                if convert_grayscale:
                    conversions['grayscale'] = convert_to_grayscale(img)
                sample_images.append((img, conversions))
                sample_paths.append(relative_path)
        else:
            failed_count += 1
    
    # Visualize results
    visualize_color_space_conversions(sample_images, sample_paths, output_dir,
                                      convert_hsv, convert_lab, convert_grayscale, vis_dir)
    
    stats = {
        'total_images': len(image_paths),
        'processed': processed_count,
        'failed': failed_count,
        'hsv': convert_hsv,
        'lab': convert_lab,
        'grayscale': convert_grayscale
    }
    
    print(f"\n✓ Color space conversion complete:")
    print(f"  - Processed: {processed_count}/{len(image_paths)}")
    print(f"  - Failed: {failed_count}")
    
    return stats


def visualize_color_space_conversions(sample_images: List[Tuple], sample_paths: List[Path],
                                     output_dir: Path, convert_hsv: bool, 
                                     convert_lab: bool, convert_grayscale: bool, vis_dir: Path = None):
    """Visualize color space conversions."""
    if vis_dir is None:
        vis_dir = output_dir / "visualizations"
    ensure_dir(vis_dir)
    
    num_samples = len(sample_images)
    if num_samples == 0:
        return
    
    # Determine number of columns (original + conversions)
    num_cols = 1 + sum([convert_hsv, convert_lab, convert_grayscale])
    
    fig, axes = plt.subplots(num_samples, num_cols, figsize=(num_cols * 3, num_samples * 3))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Color Space Conversions', fontsize=14, fontweight='bold')
    
    for i, ((original, conversions), path) in enumerate(zip(sample_images, sample_paths)):
        col = 0
        
        # Original
        axes[i, col].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[i, col].set_title('Original (BGR)', fontsize=10)
        axes[i, col].axis('off')
        col += 1
        
        # HSV
        if convert_hsv and 'hsv' in conversions:
            hsv = conversions['hsv']
            # Convert HSV to RGB for display
            hsv_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            axes[i, col].imshow(hsv_rgb)
            axes[i, col].set_title('HSV', fontsize=10)
            axes[i, col].axis('off')
            col += 1
        
        # Lab
        if convert_lab and 'lab' in conversions:
            lab = conversions['lab']
            # Convert Lab to RGB for display
            lab_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            axes[i, col].imshow(lab_rgb)
            axes[i, col].set_title('Lab', fontsize=10)
            axes[i, col].axis('off')
            col += 1
        
        # Grayscale
        if convert_grayscale and 'grayscale' in conversions:
            gray = conversions['grayscale']
            axes[i, col].imshow(gray, cmap='gray')
            axes[i, col].set_title('Grayscale', fontsize=10)
            axes[i, col].axis('off')
            col += 1
    
    plt.tight_layout()
    plt.savefig(vis_dir / "color_space_conversions.png", 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved color space visualization to {vis_dir / 'color_space_conversions.png'}")
