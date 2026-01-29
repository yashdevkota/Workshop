"""
Download reCAPTCHA datasets from Kaggle
Supports both kagglehub API and manual download validation
"""
import os
import json
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import shutil

try:
    import kagglehub
    KAGGLEHUB_AVAILABLE = True
except ImportError:
    KAGGLEHUB_AVAILABLE = False


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file"""
    try:
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except ImportError:
        print("Warning: PyYAML not installed. Using default config.")
        return {}
    except FileNotFoundError:
        print(f"Warning: Config file not found: {config_path}")
        return {}


def download_via_kagglehub(dataset_name: str, output_dir: Path) -> bool:
    """
    Download dataset using kagglehub
    
    Args:
        dataset_name: Dataset name (e.g., "mikhailma/test-dataset")
        output_dir: Output directory
    
    Returns:
        True if successful, False otherwise
    """
    if not KAGGLEHUB_AVAILABLE:
        print("✗ kagglehub not available. Install with: pip install kagglehub")
        return False
    
    try:
        print(f"Downloading {dataset_name} using kagglehub...")
        dataset_path = kagglehub.dataset_download(dataset_name)
        
        # Move to output directory
        if dataset_path and Path(dataset_path).exists():
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy contents to output directory
            source = Path(dataset_path)
            for item in source.iterdir():
                dest = output_dir / item.name
                if item.is_dir():
                    if dest.exists():
                        shutil.rmtree(dest)
                    shutil.copytree(item, dest)
                else:
                    shutil.copy2(item, dest)
            
            print(f"✓ Downloaded to {output_dir}")
            return True
        else:
            print(f"✗ Download failed: path not found")
            return False
            
    except Exception as e:
        print(f"✗ Error downloading {dataset_name}: {str(e)}")
        return False


def validate_manual_download(data_dir: Path) -> bool:
    """
    Validate manually downloaded data
    
    Args:
        data_dir: Directory containing downloaded data
    
    Returns:
        True if valid, False otherwise
    """
    if not data_dir.exists():
        return False
    
    # Check if directory has content
    items = list(data_dir.iterdir())
    if len(items) == 0:
        return False
    
    # Check for common dataset structures
    has_images = False
    for item in items:
        if item.is_dir():
            # Check for image files in subdirectories
            for subitem in item.rglob("*"):
                if subitem.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                    has_images = True
                    break
        elif item.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
            has_images = True
    
    return has_images


def explore_dataset_structure(data_dir: Path) -> dict:
    """
    Explore dataset structure and collect statistics
    
    Args:
        data_dir: Dataset directory
    
    Returns:
        Dictionary with dataset statistics
    """
    stats = {
        'total_files': 0,
        'image_files': 0,
        'directories': [],
        'class_distribution': {},
        'file_extensions': Counter()
    }
    
    print(f"\nExploring dataset structure: {data_dir.name}")
    print("=" * 60)
    
    # Find all files
    all_files = list(data_dir.rglob("*"))
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    
    for file_path in all_files:
        if file_path.is_file():
            stats['total_files'] += 1
            ext = file_path.suffix.lower()
            stats['file_extensions'][ext] += 1
            
            if ext in image_extensions:
                stats['image_files'] += 1
                
                # Try to extract class from path
                parts = file_path.parts
                for part in parts:
                    if part not in ['data', 'train', 'test', 'val'] and not part.startswith('.'):
                        if part not in stats['class_distribution']:
                            stats['class_distribution'][part] = 0
                        stats['class_distribution'][part] += 1
                        break
    
    # Find directories
    for item in data_dir.rglob("*"):
        if item.is_dir() and item.name not in ['__pycache__', '.git']:
            if item.name not in stats['directories']:
                stats['directories'].append(item.name)
    
    return stats


def visualize_dataset_info(datasets_info: dict, output_dir: Path):
    """
    Create visualizations for dataset information
    
    Args:
        datasets_info: Dictionary with dataset statistics
        output_dir: Output directory for visualizations
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Dataset Exploration Report', fontsize=16, fontweight='bold')
    
    # 1. Class distribution (bar chart)
    ax = axes[0, 0]
    all_classes = {}
    for dataset_name, stats in datasets_info.items():
        for class_name, count in stats.get('class_distribution', {}).items():
            if class_name not in all_classes:
                all_classes[class_name] = 0
            all_classes[class_name] += count
    
    if all_classes:
        classes = list(all_classes.keys())[:20]  # Top 20 classes
        counts = [all_classes[c] for c in classes]
        ax.barh(classes, counts, color='steelblue')
        ax.set_xlabel('Number of Images')
        ax.set_title('Class Distribution (Top 20)')
        ax.grid(True, alpha=0.3, axis='x')
    
    # 2. Dataset sizes comparison
    ax = axes[0, 1]
    dataset_names = list(datasets_info.keys())
    image_counts = [datasets_info[name].get('image_files', 0) for name in dataset_names]
    ax.bar(dataset_names, image_counts, color='coral')
    ax.set_ylabel('Number of Images')
    ax.set_title('Dataset Sizes')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. File extensions
    ax = axes[1, 0]
    all_extensions = Counter()
    for stats in datasets_info.values():
        all_extensions.update(stats.get('file_extensions', {}))
    
    if all_extensions:
        exts = list(all_extensions.keys())[:10]
        counts = [all_extensions[ext] for ext in exts]
        ax.pie(counts, labels=exts, autopct='%1.1f%%', startangle=90)
        ax.set_title('File Extensions Distribution')
    
    # 4. Summary statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    total_images = sum(stats.get('image_files', 0) for stats in datasets_info.values())
    total_files = sum(stats.get('total_files', 0) for stats in datasets_info.values())
    total_classes = len(set(
        class_name 
        for stats in datasets_info.values() 
        for class_name in stats.get('class_distribution', {}).keys()
    ))
    
    summary_text = f"""
    Dataset Summary
    
    Total Datasets: {len(datasets_info)}
    Total Images: {total_images:,}
    Total Files: {total_files:,}
    Unique Classes: {total_classes}
    
    Datasets:
    """
    for name, stats in datasets_info.items():
        summary_text += f"\n  • {name}:"
        summary_text += f"\n    - Images: {stats.get('image_files', 0):,}"
        summary_text += f"\n    - Files: {stats.get('total_files', 0):,}"
        summary_text += f"\n    - Classes: {len(stats.get('class_distribution', {}))}"
    
    ax.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
           verticalalignment='center', transform=ax.transAxes)
    
    plt.tight_layout()
    
    # Save
    output_path = output_dir / 'dataset_exploration.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved visualization to {output_path}")


def create_sample_image_grid(data_dir: Path, output_dir: Path, num_samples: int = 12):
    """
    Create a grid of sample images from each class
    
    Args:
        data_dir: Dataset directory
        output_dir: Output directory
        num_samples: Number of sample images per class
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp']:
        image_files.extend(data_dir.rglob(ext))
    
    if len(image_files) == 0:
        print("⚠ No images found for sample grid")
        return
    
    # Sample images
    import random
    sample_images = random.sample(image_files, min(num_samples, len(image_files)))
    
    # Create grid
    cols = 4
    rows = (len(sample_images) + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    fig.suptitle(f'Sample Images from {data_dir.name}', fontsize=14, fontweight='bold')
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, img_path in enumerate(sample_images):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        try:
            from PIL import Image
            img = Image.open(img_path)
            ax.imshow(img)
            ax.set_title(f"{img_path.parent.name}\n{img_path.name}", fontsize=8)
            ax.axis('off')
        except Exception as e:
            ax.text(0.5, 0.5, f"Error\n{str(e)[:30]}", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
    
    # Hide empty subplots
    for idx in range(len(sample_images), rows * cols):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.axis('off')
    
    plt.tight_layout()
    
    # Save
    output_path = output_dir / f'{data_dir.name}_samples.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved sample grid to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Download or validate reCAPTCHA datasets')
    parser.add_argument('--mode', type=str, default='auto',
                       choices=['auto', 'manual'],
                       help='Download mode: auto (kagglehub) or manual (validate existing)')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--output', type=str, default='data/raw',
                       help='Output directory for datasets')
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(__file__).parent.parent / args.config
    config = load_config(config_path)
    
    # Get datasets from config or use defaults
    if 'data' in config and 'datasets' in config['data']:
        datasets = config['data']['datasets']
    else:
        datasets = [
            {'name': 'sanjeetsinghnaik/google-recaptcha', 'output': 'google-recaptcha'},
            {'name': 'mikhailma/test-dataset', 'output': 'test-dataset'},
            {'name': 'cry2003/google-recaptcha-v2-images', 'output': 'google-recaptcha-v2'}
        ]
    
    project_root = Path(__file__).parent.parent
    output_base = project_root / args.output
    output_base.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("reCAPTCHA Dataset Download/Validation")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Output directory: {output_base}")
    print("=" * 60)
    
    datasets_info = {}
    results = []
    
    for dataset in datasets:
        dataset_name = dataset['name']
        output_name = dataset['output']
        output_dir = output_base / output_name
        
        print(f"\n{'='*60}")
        print(f"Processing: {dataset_name}")
        print(f"{'='*60}")
        
        if args.mode == 'auto':
            # Try to download via kagglehub
            success = download_via_kagglehub(dataset_name, output_dir)
            if not success:
                print(f"⚠ Download failed. Checking if data already exists...")
                if validate_manual_download(output_dir):
                    print(f"✓ Found existing data in {output_dir}")
                    success = True
        else:
            # Manual mode: validate existing data
            if validate_manual_download(output_dir):
                print(f"✓ Valid data found in {output_dir}")
                success = True
            else:
                print(f"✗ No valid data found in {output_dir}")
                print(f"  Please download manually and place in: {output_dir}")
                success = False
        
        if success:
            # Explore dataset structure
            stats = explore_dataset_structure(output_dir)
            datasets_info[output_name] = stats
            
            # Create sample image grid
            vis_dir = output_base / 'visualizations'
            create_sample_image_grid(output_dir, vis_dir, num_samples=12)
        
        results.append((dataset_name, success))
    
    # Create comprehensive visualization
    if datasets_info:
        vis_dir = output_base / 'visualizations'
        visualize_dataset_info(datasets_info, vis_dir)
        
        # Save statistics
        stats_file = output_base / 'dataset_info.json'
        with open(stats_file, 'w') as f:
            json.dump(datasets_info, f, indent=2, default=str)
        print(f"\n✓ Saved dataset statistics to {stats_file}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("Download/Validation Summary")
    print(f"{'='*60}")
    for name, success in results:
        status = "✓ Success" if success else "✗ Failed"
        print(f"{status}: {name}")
    
    successful = sum(1 for _, success in results if success)
    print(f"\nTotal: {successful}/{len(results)} datasets ready")
    print("=" * 60)


if __name__ == "__main__":
    main()
