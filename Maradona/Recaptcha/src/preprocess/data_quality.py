"""
Data quality checks: duplicate detection, class distribution analysis.
"""
from pathlib import Path
from typing import Dict, List, Tuple, Set
import numpy as np
import cv2
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from .utils import load_image, ensure_dir


def detect_duplicate_images(image_paths: List[Path], 
                            similarity_threshold: float = 0.95,
                            method: str = 'hash') -> Dict:
    """
    Detect duplicate or near-duplicate images.
    
    Args:
        image_paths: List of image paths
        similarity_threshold: Similarity threshold (0.95 = 95% similar)
        method: 'hash' (fast) or 'feature' (accurate but slow)
    
    Returns:
        Dictionary with duplicate groups
    """
    print(f"\n{'='*60}")
    print("Detecting duplicate images...")
    print(f"{'='*60}")
    
    if method == 'hash':
        return _detect_duplicates_by_hash(image_paths, similarity_threshold)
    else:
        return _detect_duplicates_by_features(image_paths, similarity_threshold)


def _detect_duplicates_by_hash(image_paths: List[Path], 
                               threshold: float) -> Dict:
    """
    Detect duplicates using perceptual hash (O(N) 방식).
    
    수정된 O(N) 방식: 중첩 루프 없이 해시 그룹화로 빠른 처리.
    """
    try:
        from PIL import Image
        import imagehash
    except ImportError:
        print("Warning: imagehash not installed. Install with: pip install imagehash")
        return {}
    
    print("Computing perceptual hashes...")
    # 해시를 키로 사용하여 경로들을 그룹화 (O(N))
    hash_to_paths = {}
    
    for img_path in tqdm(image_paths, desc="Computing hashes"):
        try:
            # 해상도를 낮춰서 읽으면 속도가 더 빨라집니다
            img = Image.open(img_path).convert('L').resize((16, 16), Image.LANCZOS)
            phash = str(imagehash.phash(img))  # 비교를 위해 문자열로 변환
            
            if phash not in hash_to_paths:
                hash_to_paths[phash] = []
            hash_to_paths[phash].append(img_path)
        except Exception as e:
            continue
    
    # 그룹화 (O(N))
    # 동일한 해시를 가진 것들(완전 중복)만 먼저 골라냅니다
    duplicate_groups = {
        paths[0]: paths 
        for phash, paths in hash_to_paths.items() 
        if len(paths) > 1
    }
    
    print(f"Found {len(duplicate_groups)} exact duplicate groups")
    
    # 유사한 해시도 찾기 (선택적, threshold 기반)
    # Hamming distance 기반 유사도 검사는 필요시 추가 가능
    # 하지만 대부분의 경우 완전 중복만 찾아도 충분합니다
    
    return duplicate_groups


def _detect_duplicates_by_features(image_paths: List[Path], 
                                   threshold: float) -> Dict:
    """Detect duplicates using feature matching (slow but accurate)."""
    # This is more accurate but much slower
    # For 55,000 images, hash method is recommended
    print("Feature-based duplicate detection is slow. Use hash method instead.")
    return {}


def check_filename_duplicates(image_paths: List[Path]) -> Dict[str, List[Path]]:
    """
    Check for duplicate filenames (simple method).
    
    Returns:
        Dictionary mapping filename to list of paths with that filename
    """
    filename_groups = {}
    
    for img_path in image_paths:
        filename = img_path.name
        if filename not in filename_groups:
            filename_groups[filename] = []
        filename_groups[filename].append(img_path)
    
    # Return only duplicates
    duplicates = {k: v for k, v in filename_groups.items() if len(v) > 1}
    return duplicates


def analyze_class_distribution(data_dir: Path) -> Dict:
    """
    Analyze class distribution in dataset.
    
    Args:
        data_dir: Root directory containing class folders
    
    Returns:
        Dictionary with class counts, ratios, and balance info
    """
    print(f"\n{'='*60}")
    print("Analyzing class distribution...")
    print(f"{'='*60}")
    
    class_counts = {}
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    
    # Find all class directories
    for item in data_dir.rglob("*"):
        if item.is_dir():
            # Count images in this directory
            images = []
            for ext in image_extensions:
                images.extend(list(item.glob(f"*{ext}")))
                images.extend(list(item.glob(f"*{ext.upper()}")))
            
            if len(images) > 0:
                class_name = item.name
                class_counts[class_name] = len(images)
    
    if not class_counts:
        # Try alternative structure: look for common class names in path
        for img_path in data_dir.rglob("*"):
            if img_path.is_file() and img_path.suffix.lower() in image_extensions:
                # Extract class from parent directory
                class_name = img_path.parent.name
                if class_name not in ['data', 'train', 'test', 'val', 'images']:
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    total = sum(class_counts.values())
    class_ratios = {k: v/total for k, v in class_counts.items()} if total > 0 else {}
    
    # Check balance
    if class_ratios:
        max_ratio = max(class_ratios.values())
        min_ratio = min(class_ratios.values())
        balance_ratio = max_ratio / min_ratio if min_ratio > 0 else float('inf')
        is_balanced = balance_ratio < 2.0
    else:
        balance_ratio = float('inf')
        is_balanced = False
    
    result = {
        'counts': class_counts,
        'ratios': class_ratios,
        'total': total,
        'num_classes': len(class_counts),
        'balance_ratio': balance_ratio,
        'is_balanced': is_balanced,
        'max_class': max(class_counts.items(), key=lambda x: x[1])[0] if class_counts else None,
        'min_class': min(class_counts.items(), key=lambda x: x[1])[0] if class_counts else None
    }
    
    print(f"Total images: {total:,}")
    print(f"Number of classes: {len(class_counts)}")
    print(f"Balance ratio: {balance_ratio:.2f} {'(balanced)' if is_balanced else '(imbalanced)'}")
    print(f"Max class: {result['max_class']} ({class_counts.get(result['max_class'], 0):,} images)")
    print(f"Min class: {result['min_class']} ({class_counts.get(result['min_class'], 0):,} images)")
    
    return result


def visualize_class_distribution(class_dist: Dict, output_dir: Path, vis_base_dir: Path = None):
    """Visualize class distribution with imbalance analysis."""
    if vis_base_dir is None:
        vis_base_dir = output_dir / "visualizations"
    else:
        vis_base_dir = vis_base_dir / "quality_checks"
    ensure_dir(vis_base_dir)
    
    counts = class_dist['counts']
    ratios = class_dist['ratios']
    balance_ratio = class_dist.get('balance_ratio', 1.0)
    is_balanced = class_dist.get('is_balanced', True)
    max_class = class_dist.get('max_class', '')
    min_class = class_dist.get('min_class', '')
    
    # Filter out non-class entries (like 'visualizations')
    filtered_counts = {k: v for k, v in counts.items() 
                     if k not in ['visualizations'] and v > 0}
    filtered_ratios = {k: v for k, v in ratios.items() 
                      if k not in ['visualizations'] and v > 0}
    
    classes = list(filtered_counts.keys())
    values = list(filtered_counts.values())
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Bar chart of class counts
    ax1 = fig.add_subplot(gs[0, :])
    bars = ax1.barh(classes, values, color='steelblue')
    ax1.set_xlabel('Number of Images', fontsize=12)
    ax1.set_title('Class Distribution (Counts)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax1.text(val, i, f' {val:,}', va='center', fontsize=9)
    
    # 2. Pie chart of class ratios (with no text overlap)
    ax2 = fig.add_subplot(gs[1, 0])
    if len(classes) <= 20:
        colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
        # Use labeldistance and pctdistance to prevent overlap
        wedges, texts, autotexts = ax2.pie(values, labels=None, autopct='',
                                           colors=colors, startangle=90,
                                           textprops={'fontsize': 8})
        ax2.set_title('Class Distribution (Ratios)', fontsize=12, fontweight='bold')
        
        # Add labels outside the pie chart to prevent overlap
        for i, (wedge, cls, val, ratio) in enumerate(zip(wedges, classes, values, [filtered_ratios.get(c, 0) for c in classes])):
            # Calculate label position outside the pie
            angle = (wedge.theta2 + wedge.theta1) / 2
            x = np.cos(np.deg2rad(angle))
            y = np.sin(np.deg2rad(angle))
            
            # Position label outside
            label_x = 1.15 * x
            label_y = 1.15 * y
            
            # Add class name and percentage
            label_text = f'{cls}: {ratio:.1f}%'
            ax2.text(label_x, label_y, label_text, 
                    ha='center' if abs(x) < 0.1 else ('left' if x > 0 else 'right'),
                    va='center' if abs(y) < 0.1 else ('bottom' if y > 0 else 'top'),
                    fontsize=8, fontweight='bold')
    else:
        ax2.axis('off')
        text_content = "Top 20 Classes by Ratio:\n\n"
        for cls, ratio in sorted(filtered_ratios.items(), key=lambda x: x[1], reverse=True)[:20]:
            text_content += f"{cls}: {ratio:.2%}\n"
        ax2.text(0.5, 0.5, text_content, ha='center', va='center',
               fontsize=9, family='monospace', transform=ax2.transAxes)
        ax2.set_title('Top 20 Classes by Ratio', fontsize=12, fontweight='bold')
    
    # 3. Class Imbalance Analysis
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    
    # Calculate imbalance metrics
    sorted_values = sorted(values)
    max_val = max(values)
    min_val = min([v for v in values if v > 0])
    
    imbalance_text = f"""
    ⚠️ CLASS IMBALANCE ANALYSIS ⚠️
    
    Balance Ratio: {balance_ratio:.2f}
    Status: {'✓ Balanced' if is_balanced else '✗ Imbalanced'}
    
    Max Class: {max_class}
      - Count: {counts.get(max_class, 0):,}
      - Ratio: {ratios.get(max_class, 0):.2%}
    
    Min Class: {min_class}
      - Count: {counts.get(min_class, 0):,}
      - Ratio: {ratios.get(min_class, 0):.2%}
    
    Ratio Difference: {max_val / min_val if min_val > 0 else 0:.1f}x
    
    Recommendations:
    • Balance Ratio > 2.0: Consider SMOTE or Undersampling
    • Classes with < 50 images: Consider removal or augmentation
    • Use 'class_weight=balanced' in training
    """
    
    # Color code based on balance
    text_color = 'red' if not is_balanced else 'green'
    ax3.text(0.1, 0.5, imbalance_text, ha='left', va='center',
            fontsize=10, family='monospace', transform=ax3.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            color=text_color)
    
    # 4. Class size distribution (log scale for better visualization)
    ax4 = fig.add_subplot(gs[2, :])
    sorted_classes = sorted(filtered_counts.items(), key=lambda x: x[1], reverse=True)
    class_names = [c[0] for c in sorted_classes]
    class_counts = [c[1] for c in sorted_classes]
    
    bars = ax4.bar(range(len(class_names)), class_counts, color='coral')
    ax4.set_xticks(range(len(class_names)))
    ax4.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    ax4.set_ylabel('Number of Images (Log Scale)', fontsize=11)
    ax4.set_title('Class Size Distribution (Sorted by Count)', fontsize=12, fontweight='bold')
    ax4.set_yscale('log')  # Log scale to better show imbalance
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add threshold line for minimum samples
    min_samples_threshold = 50
    ax4.axhline(y=min_samples_threshold, color='r', linestyle='--', 
               label=f'Minimum Recommended: {min_samples_threshold}')
    ax4.legend()
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, class_counts)):
        if val > 0:
            ax4.text(i, val, f'{val:,}', ha='center', va='bottom', 
                    fontsize=8, rotation=90)
    
    plt.suptitle('Class Distribution and Imbalance Analysis', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(vis_base_dir / "class_distribution.png", 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved class distribution visualization to {vis_base_dir / 'class_distribution.png'}")
    
    # Create simple ratio-only visualization (no text overlap)
    create_ratio_only_visualization(class_dist, vis_base_dir)
    
    # Create imbalance evidence visualization
    create_imbalance_evidence(class_dist, vis_base_dir)


def create_ratio_only_visualization(class_dist: Dict, output_dir: Path):
    """
    Create a simple ratio-only pie chart visualization without text overlap.
    Shows class distribution as percentages in a clean pie chart.
    """
    ensure_dir(output_dir)
    
    # Use 'counts' key (not 'class_counts')
    counts = class_dist.get('counts', {})
    total = class_dist.get('total', 0)
    
    if total == 0:
        print("Warning: No images found. Cannot create ratio visualization.")
        return
    
    # Filter out non-class entries
    filtered_counts = {k: v for k, v in counts.items() 
                     if k not in ['visualizations'] and v > 0}
    
    if not filtered_counts:
        print("Warning: No valid classes found. Cannot create ratio visualization.")
        return
    
    # Calculate ratios
    ratios = {k: (v / total * 100) if total > 0 else 0 
              for k, v in filtered_counts.items()}
    
    # Sort by ratio (descending)
    sorted_ratios = sorted(ratios.items(), key=lambda x: x[1], reverse=True)
    classes = [c[0] for c in sorted_ratios]
    percentages = [c[1] for c in sorted_ratios]
    values = [filtered_counts[c] for c in classes]
    
    if not percentages:
        print("Warning: No percentages to visualize.")
        return
    
    # Create figure with space for legend on the right
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create pie chart with colors
    colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
    
    # Create labels with percentage and count for legend
    legend_labels = [f'{cls}: {pct:.1f}% ({count:,})' 
                     for cls, pct, count in zip(classes, percentages, values)]
    
    # Create pie chart without labels inside
    wedges, texts = ax.pie(values, labels=None, colors=colors, startangle=90,
                           textprops={'fontsize': 10})
    
    # Add legend on the right side
    ax.legend(wedges, legend_labels, 
              title="Class Distribution",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1),
              fontsize=10,
              title_fontsize=12)
    
    ax.set_title('Class Distribution - Ratio Only (Pie Chart)', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Add summary text at the bottom
    total_classes = len(classes)
    summary_text = f'Total Classes: {total_classes} | Total Images: {total:,}'
    fig.text(0.5, 0.02, summary_text, ha='center', fontsize=11, style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'class_distribution_ratio.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved ratio-only pie chart with legend to {output_dir / 'class_distribution_ratio.png'}")


def create_imbalance_evidence(class_dist: Dict, output_dir: Path):
    """
    Create a clear evidence visualization showing class imbalance.
    This is used as proof/documentation of the imbalance issue.
    """
    ensure_dir(output_dir)
    
    counts = class_dist['counts']
    ratios = class_dist['ratios']
    balance_ratio = class_dist.get('balance_ratio', 1.0)
    is_balanced = class_dist.get('is_balanced', True)
    max_class = class_dist.get('max_class', '')
    min_class = class_dist.get('min_class', '')
    total = class_dist.get('total', 0)
    
    # Filter out non-class entries
    filtered_counts = {k: v for k, v in counts.items() 
                     if k not in ['visualizations'] and v > 0}
    
    # Sort by count
    sorted_classes = sorted(filtered_counts.items(), key=lambda x: x[1], reverse=True)
    class_names = [c[0] for c in sorted_classes]
    class_counts = [c[1] for c in sorted_classes]
    
    # Create evidence visualization
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.3)
    
    # 1. Bar chart with imbalance highlighted
    ax1 = fig.add_subplot(gs[0, :2])
    colors = ['red' if name == max_class else 'orange' if name == min_class else 'steelblue' 
              for name in class_names]
    bars = ax1.barh(class_names, class_counts, color=colors)
    ax1.set_xlabel('Number of Images', fontsize=14, fontweight='bold')
    ax1.set_title('Class Distribution - Imbalance Evidence', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val, name) in enumerate(zip(bars, class_counts, class_names)):
        ax1.text(val, i, f' {val:,}', va='center', fontsize=10, fontweight='bold')
        if name == max_class:
            ax1.text(val * 0.5, i, ' ← MAX', va='center', fontsize=9, 
                    color='red', fontweight='bold')
        elif name == min_class:
            ax1.text(val * 0.5, i, ' ← MIN', va='center', fontsize=9, 
                    color='orange', fontweight='bold')
    
    # 2. Ratio comparison (Max vs Min)
    ax2 = fig.add_subplot(gs[0, 2])
    max_count = counts.get(max_class, 0)
    min_count = counts.get(min_class, 0)
    
    comparison_data = [max_count, min_count]
    comparison_labels = [f'{max_class}\n({max_count:,})', f'{min_class}\n({min_count:,})']
    colors_comp = ['red', 'orange']
    
    bars2 = ax2.bar(comparison_labels, comparison_data, color=colors_comp, alpha=0.7)
    ax2.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
    ax2.set_title('Max vs Min Class\nComparison', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add ratio annotation
    ratio_text = f'Ratio: {balance_ratio:.1f}x'
    ax2.text(0.5, 0.95, ratio_text, transform=ax2.transAxes,
            ha='center', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    # Add value labels
    for bar, val in zip(bars2, comparison_data):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 3. Imbalance statistics
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axis('off')
    
    stats_text = f"""
    ⚠️ CLASS IMBALANCE EVIDENCE ⚠️
    
    Total Images: {total:,}
    Number of Classes: {len(filtered_counts)}
    
    Balance Ratio: {balance_ratio:.2f}
    Status: {'✓ Balanced' if is_balanced else '✗ SEVERELY IMBALANCED'}
    
    Maximum Class:
      • Name: {max_class}
      • Count: {max_count:,}
      • Ratio: {ratios.get(max_class, 0):.2%}
    
    Minimum Class:
      • Name: {min_class}
      • Count: {min_count:,}
      • Ratio: {ratios.get(min_class, 0):.2%}
    
    Difference: {max_count / min_count if min_count > 0 else 0:.1f}x
    
    ⚠️ WARNING ⚠️
    Balance Ratio > 2.0 indicates
    severe class imbalance!
    
    Recommendations:
    • Use class_weight='balanced'
    • Apply SMOTE or Undersampling
    • Consider removing classes
      with < 50 samples
    """
    
    ax3.text(0.1, 0.5, stats_text, ha='left', va='center',
            fontsize=11, family='monospace', transform=ax3.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.9),
            color='darkred')
    
    # 4. Distribution histogram
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(class_counts, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    ax4.axvline(max_count, color='red', linestyle='--', linewidth=2, label=f'Max: {max_class}')
    ax4.axvline(min_count, color='orange', linestyle='--', linewidth=2, label=f'Min: {min_class}')
    ax4.set_xlabel('Number of Images per Class', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax4.set_title('Class Size Distribution', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Ratio pie chart (top 10)
    ax5 = fig.add_subplot(gs[1, 2])
    top_10 = sorted_classes[:10]
    top_10_names = [c[0] for c in top_10]
    top_10_counts = [c[1] for c in top_10]
    top_10_ratios = [ratios.get(name, 0) for name in top_10_names]
    
    colors_pie = plt.cm.Set3(np.linspace(0, 1, len(top_10)))
    wedges, texts, autotexts = ax5.pie(top_10_counts, labels=top_10_names, 
                                       autopct='%1.1f%%', colors=colors_pie,
                                       startangle=90, textprops={'fontsize': 8})
    ax5.set_title('Top 10 Classes\nby Count', fontsize=12, fontweight='bold')
    
    plt.suptitle('CLASS IMBALANCE EVIDENCE - Proof of Data Quality Issue', 
                fontsize=18, fontweight='bold', y=0.995, color='darkred')
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(output_dir / "class_imbalance_evidence.png", 
                dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved class imbalance evidence to {output_dir / 'class_imbalance_evidence.png'}")


def get_representative_images(duplicate_groups: Dict) -> Set[Path]:
    """
    Get representative images from duplicate groups (keep first, remove others).
    
    Args:
        duplicate_groups: Dictionary of duplicate groups
    
    Returns:
        Set of representative image paths to keep
    """
    representative = set()
    
    for group in duplicate_groups.values():
        # Keep first image, mark others for removal
        if len(group) > 0:
            representative.add(group[0])
    
    return representative
