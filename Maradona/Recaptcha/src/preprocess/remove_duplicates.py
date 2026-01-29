"""
Remove duplicate images based on quality check results.
"""
from pathlib import Path
from typing import Dict, List, Set
import json
import shutil
from tqdm import tqdm

from .utils import ensure_dir


def remove_duplicate_images(duplicate_groups_path: Path,
                           output_dir: Path,
                           action: str = 'move') -> Dict:
    """
    Remove duplicate images based on duplicate groups.
    
    Args:
        duplicate_groups_path: Path to duplicate_groups.json
        output_dir: Directory to save removed images (if action='move')
        action: 'move' (move to output_dir) or 'delete' (permanently delete)
    
    Returns:
        Dictionary with removal statistics
    """
    print(f"\n{'='*60}")
    print("Removing duplicate images...")
    print(f"Action: {action}")
    print(f"{'='*60}")
    
    if not duplicate_groups_path.exists():
        print(f"Error: {duplicate_groups_path} not found")
        return {}
    
    # Load duplicate groups
    with open(duplicate_groups_path, 'r') as f:
        duplicate_groups = json.load(f)
    
    # Convert string keys back to Path objects
    groups = {}
    for key, paths in duplicate_groups.items():
        groups[Path(key)] = [Path(p) for p in paths]
    
    print(f"Found {len(groups)} duplicate groups")
    
    # Collect all duplicate images (except first one in each group)
    duplicates_to_remove = []
    representative_images = set()
    
    for representative, group in groups.items():
        representative_images.add(representative)
        # Add all others as duplicates
        for img_path in group[1:]:  # Skip first (representative)
            duplicates_to_remove.append(img_path)
    
    print(f"Total duplicate images to remove: {len(duplicates_to_remove):,}")
    print(f"Representative images to keep: {len(representative_images):,}")
    
    # Remove duplicates
    removed_count = 0
    failed_count = 0
    
    if action == 'move':
        ensure_dir(output_dir / "removed_duplicates")
    
    for dup_path in tqdm(duplicates_to_remove, desc="Removing duplicates"):
        if not dup_path.exists():
            failed_count += 1
            continue
        
        try:
            if action == 'move':
                # Move to removed_duplicates directory
                relative_path = dup_path.relative_to(dup_path.parents[2])  # Go up to data/raw
                dest_path = output_dir / "removed_duplicates" / relative_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(dup_path), str(dest_path))
            elif action == 'delete':
                # Permanently delete
                dup_path.unlink()
            
            removed_count += 1
        except Exception as e:
            print(f"Error processing {dup_path}: {e}")
            failed_count += 1
    
    stats = {
        'total_duplicate_groups': len(groups),
        'total_duplicates_removed': removed_count,
        'representative_images_kept': len(representative_images),
        'failed': failed_count,
        'action': action
    }
    
    print(f"\n✓ Duplicate removal complete:")
    print(f"  - Removed: {removed_count:,}")
    print(f"  - Kept: {len(representative_images):,}")
    print(f"  - Failed: {failed_count}")
    
    # Save statistics
    stats_path = output_dir / "duplicate_removal_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    
    return stats


def filter_classes(data_dir: Path,
                  classes_to_exclude: List[str],
                  output_dir: Path,
                  action: str = 'move') -> Dict:
    """
    Filter out specific classes (e.g., 'visualizations', 'Mountain').
    
    Args:
        data_dir: Root data directory
        classes_to_exclude: List of class names to exclude
        output_dir: Directory to save excluded images
        action: 'move' or 'delete'
    
    Returns:
        Dictionary with filtering statistics
    """
    print(f"\n{'='*60}")
    print(f"Filtering out classes: {classes_to_exclude}")
    print(f"{'='*60}")
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    images_to_remove = []
    
    # Find all images in excluded classes
    for class_name in classes_to_exclude:
        # Skip if it's the visualizations directory itself (not a class folder)
        if class_name == 'visualizations' and (data_dir / 'visualizations').exists():
            # Check if it's actually a visualizations folder (contains .png files, not class subdirs)
            vis_dir = data_dir / 'visualizations'
            has_png_files = any(vis_dir.glob("*.png")) or any(vis_dir.glob("*.jpg"))
            if has_png_files:
                print(f"⚠ Skipping 'visualizations' folder - it's a visualization directory, not a class")
                continue
        
        class_dir = data_dir / class_name
        if not class_dir.exists():
            # Try to find in subdirectories
            for root_dir in data_dir.rglob("*"):
                if root_dir.is_dir() and root_dir.name == class_name:
                    # Double check: skip if it's the visualizations directory
                    if class_name == 'visualizations':
                        has_png_files = any(root_dir.glob("*.png")) or any(root_dir.glob("*.jpg"))
                        if has_png_files:
                            print(f"⚠ Skipping 'visualizations' folder at {root_dir} - it's a visualization directory")
                            continue
                    class_dir = root_dir
                    break
        
        if class_dir.exists() and class_dir.is_dir():
            for ext in image_extensions:
                images_to_remove.extend(list(class_dir.glob(f"*{ext}")))
                images_to_remove.extend(list(class_dir.glob(f"*{ext.upper()}")))
    
    print(f"Found {len(images_to_remove)} images in excluded classes")
    
    if action == 'move':
        ensure_dir(output_dir / "removed_classes")
    
    removed_count = 0
    failed_count = 0
    
    for img_path in tqdm(images_to_remove, desc="Removing class images"):
        if not img_path.exists():
            failed_count += 1
            continue
        
        try:
            if action == 'move':
                relative_path = img_path.relative_to(data_dir)
                dest_path = output_dir / "removed_classes" / relative_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(img_path), str(dest_path))
            elif action == 'delete':
                img_path.unlink()
            
            removed_count += 1
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            failed_count += 1
    
    stats = {
        'excluded_classes': classes_to_exclude,
        'images_removed': removed_count,
        'failed': failed_count,
        'action': action
    }
    
    print(f"\n✓ Class filtering complete:")
    print(f"  - Removed: {removed_count:,}")
    print(f"  - Failed: {failed_count}")
    
    # Save statistics
    stats_path = output_dir / "class_filtering_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    
    return stats
