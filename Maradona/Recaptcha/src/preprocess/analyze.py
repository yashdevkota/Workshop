"""
Dataset structure analysis module.
Analyzes image sizes, brightness distribution, and noise levels.
"""
import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from .utils import load_image, get_image_stats, ensure_dir


def analyze_dataset_structure(data_dir: Path, output_dir: Path, 
                             num_samples: int = 0, vis_dir: Path = None) -> Dict:
    """
    Analyze dataset structure: image sizes, brightness, noise levels.
    
    Args:
        data_dir: Root directory of raw dataset
        output_dir: Directory to save analysis results
        num_samples: Maximum number of images to analyze (for speed)
        
    Returns:
        Dictionary with analysis results
    """
    print(f"\n{'='*60}")
    print("Analyzing dataset structure...")
    print(f"{'='*60}")
    
    # Find all images
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    image_paths = []
    
    for root, _, files in os.walk(data_dir):
        for f in files:
            if Path(f).suffix.lower() in image_extensions:
                image_paths.append(Path(root) / f)
    
    total_images = len(image_paths)
    
    # Limit samples for analysis (0 means analyze all)
    if num_samples > 0 and total_images > num_samples:
        indices = np.random.choice(total_images, num_samples, replace=False)
        sampled_paths = [image_paths[i] for i in indices]
        print(f"Total images found: {total_images}")
        print(f"Analyzing {num_samples} random samples (use num_samples=0 to analyze all)")
        image_paths = sampled_paths
    else:
        print(f"Analyzing all {total_images} images")
    
    # Collect statistics
    sizes = []
    brightness_values = []
    noise_levels = []
    all_stats = []
    failed_count = 0
    
    for img_path in tqdm(image_paths, desc="Analyzing images"):
        img = load_image(img_path)
        if img is None:
            failed_count += 1
            continue
        
        stats = get_image_stats(img)
        all_stats.append(stats)
        
        sizes.append((stats['width'], stats['height']))
        brightness_values.append(stats['mean_brightness'])
        noise_levels.append(stats['std_dev'])
    
    if failed_count > 0:
        print(f"Warning: {failed_count} images failed to load out of {len(image_paths)} samples")
    
    # Calculate summary statistics
    analysis = {
        'total_images_found': total_images,
        'total_images_analyzed': len(all_stats),
        'failed_to_load': failed_count,
        'image_sizes': {
            'widths': [s[0] for s in sizes],
            'heights': [s[1] for s in sizes],
            'mean_width': float(np.mean([s[0] for s in sizes])),
            'mean_height': float(np.mean([s[1] for s in sizes])),
            'min_width': int(np.min([s[0] for s in sizes])),
            'max_width': int(np.max([s[0] for s in sizes])),
            'min_height': int(np.min([s[1] for s in sizes])),
            'max_height': int(np.max([s[1] for s in sizes]))
        },
        'brightness': {
            'values': brightness_values,
            'mean': float(np.mean(brightness_values)),
            'std': float(np.std(brightness_values)),
            'min': float(np.min(brightness_values)),
            'max': float(np.max(brightness_values)),
            'median': float(np.median(brightness_values))
        },
        'noise_levels': {
            'values': noise_levels,
            'mean': float(np.mean(noise_levels)),
            'std': float(np.std(noise_levels)),
            'min': float(np.min(noise_levels)),
            'max': float(np.max(noise_levels)),
            'median': float(np.median(noise_levels))
        },
        'recommendations': _generate_recommendations(all_stats)
    }
    
    # Visualize results
    visualize_analysis(analysis, output_dir, vis_dir)
    
    # Save analysis results
    ensure_dir(output_dir)
    import json
    with open(output_dir / "dataset_analysis.json", 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    
    print(f"\n✓ Analysis complete. Results saved to {output_dir}")
    print(f"  - Total images found: {analysis['total_images_found']}")
    print(f"  - Successfully analyzed: {analysis['total_images_analyzed']}")
    if analysis['failed_to_load'] > 0:
        print(f"  - Failed to load: {analysis['failed_to_load']}")
    print(f"  - Mean image size: {analysis['image_sizes']['mean_width']:.0f}x{analysis['image_sizes']['mean_height']:.0f}")
    print(f"  - Mean brightness: {analysis['brightness']['mean']:.1f}")
    print(f"  - Mean noise level: {analysis['noise_levels']['mean']:.1f}")
    
    return analysis


def _generate_recommendations(stats: List[Dict]) -> Dict:
    """Generate preprocessing recommendations based on statistics."""
    mean_brightness = np.mean([s['mean_brightness'] for s in stats])
    mean_noise = np.mean([s['std_dev'] for s in stats])
    
    recommendations = {
        'noise_reduction_needed': mean_noise > 15.0,
        'clahe_needed': mean_brightness < 100 or np.std([s['mean_brightness'] for s in stats]) > 30,
        'gamma_correction_needed': mean_brightness < 100,
        'suggested_noise_threshold': float(mean_noise),
        'suggested_brightness_threshold': float(mean_brightness)
    }
    
    return recommendations


def visualize_analysis(analysis: Dict, output_dir: Path, vis_dir: Path = None):
    """Create visualizations for dataset analysis."""
    if vis_dir is None:
        vis_dir = output_dir / "visualizations"
    else:
        vis_dir = vis_dir / "analysis"
    ensure_dir(vis_dir)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Dataset Structure Analysis', fontsize=16, fontweight='bold')
    
    # 1. Image size distribution
    ax = axes[0, 0]
    widths = analysis['image_sizes']['widths']
    heights = analysis['image_sizes']['heights']
    
    # Check if sizes are discrete (limited unique values)
    unique_sizes = set(zip(widths, heights))
    if len(unique_sizes) <= 10:
        # Use count plot for discrete sizes
        from collections import Counter
        size_counts = Counter(zip(widths, heights))
        size_labels = [f"{w}×{h}" for w, h in sorted(size_counts.keys())]
        counts = [size_counts[k] for k in sorted(size_counts.keys())]
        
        bars = ax.bar(size_labels, counts, color='steelblue', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Image Size (Width × Height)')
        ax.set_ylabel('Count')
        ax.set_title('Image Size Distribution (Count)')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add count labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=9)
    else:
        # Use scatter plot for continuous sizes
        ax.scatter(widths, heights, alpha=0.5, s=10)
        ax.set_xlabel('Width (pixels)')
        ax.set_ylabel('Height (pixels)')
        ax.set_title('Image Size Distribution')
        ax.grid(True, alpha=0.3)
    
    # 2. Brightness distribution
    ax = axes[0, 1]
    ax.hist(analysis['brightness']['values'], bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(analysis['brightness']['mean'], color='red', linestyle='--', 
               label=f"Mean: {analysis['brightness']['mean']:.1f}")
    ax.axvline(analysis['brightness']['median'], color='green', linestyle='--', 
               label=f"Median: {analysis['brightness']['median']:.1f}")
    ax.set_xlabel('Mean Brightness')
    ax.set_ylabel('Frequency')
    ax.set_title('Brightness Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Noise level distribution
    ax = axes[1, 0]
    ax.hist(analysis['noise_levels']['values'], bins=50, edgecolor='black', alpha=0.7, color='orange')
    ax.axvline(analysis['noise_levels']['mean'], color='red', linestyle='--', 
               label=f"Mean: {analysis['noise_levels']['mean']:.1f}")
    ax.axvline(analysis['noise_levels']['median'], color='green', linestyle='--', 
               label=f"Median: {analysis['noise_levels']['median']:.1f}")
    ax.set_xlabel('Noise Level (Std Dev)')
    ax.set_ylabel('Frequency')
    ax.set_title('Noise Level Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Recommendations summary
    ax = axes[1, 1]
    ax.axis('off')
    rec_text = "Preprocessing Recommendations:\n\n"
    rec = analysis['recommendations']
    rec_text += f"• Noise Reduction: {'✓ Needed' if rec['noise_reduction_needed'] else '✗ Not needed'}\n"
    rec_text += f"• CLAHE: {'✓ Needed' if rec['clahe_needed'] else '✗ Not needed'}\n"
    rec_text += f"• Gamma Correction: {'✓ Needed' if rec['gamma_correction_needed'] else '✗ Not needed'}\n"
    rec_text += f"\nSuggested Thresholds:\n"
    rec_text += f"• Noise: {rec['suggested_noise_threshold']:.1f}\n"
    rec_text += f"• Brightness: {rec['suggested_brightness_threshold']:.1f}"
    
    ax.text(0.1, 0.5, rec_text, fontsize=12, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(vis_dir / "dataset_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved analysis visualization to {vis_dir / 'dataset_analysis.png'}")
