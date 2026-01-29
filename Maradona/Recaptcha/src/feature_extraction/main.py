"""
Main feature extraction pipeline.
Extracts hand-crafted features from preprocessed images.
"""
import sys
import argparse
from pathlib import Path
from typing import Optional, Dict, Tuple
import numpy as np
import json
from tqdm import tqdm

# Add parent directory to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.preprocess.utils import load_config
from src.feature_extraction.utils import (
    image_generator,
    collect_image_paths,
    save_features_batch,
    load_cached_features,
    check_cache_exists,
    ensure_dir,
    load_image_npy
)
from src.feature_extraction.combined_extractor import CombinedExtractor
from src.feature_extraction import visualization as feat_vis


def extract_features_for_split(
    data_dir: Path,
    output_dir: Path,
    split: str,
    extractor: CombinedExtractor,
    config: Dict,
    use_cache: bool = True,
    batch_size: int = 1000
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Extract features for a specific split (train/val/test).
    
    Args:
        data_dir: Directory containing processed images
        output_dir: Directory to save extracted features
        split: Split name (train/val/test)
        extractor: Feature extractor instance
        use_cache: Whether to use cached features if available
        batch_size: Batch size for saving features
        
    Returns:
        Tuple of (features, labels, stats)
    """
    # Check cache
    if use_cache and check_cache_exists(output_dir, split):
        print(f"✓ Loading cached features for {split} split...")
        features, labels = load_cached_features(output_dir, split)
        stats = {
            'split': split,
            'num_samples': len(features),
            'feature_dim': features.shape[1],
            'cached': True
        }
        return features, labels, stats
    
    # Collect image paths and labels
    print(f"\nCollecting image paths for {split} split...")
    image_paths, labels = collect_image_paths(data_dir, split)
    
    if len(image_paths) == 0:
        print(f"⚠️  No images found for {split} split")
        return np.array([]), np.array([]), {'split': split, 'num_samples': 0}
    
    print(f"Found {len(image_paths)} images")
    
    # Load grayscale and HSV images from color_spaces_v2
    project_root = Path(__file__).parent.parent.parent
    color_spaces_dir = project_root / config['data']['processed_dir'] / "color_spaces_v2"
    grayscale_paths = None
    hsv_paths = None
    
    if color_spaces_dir.exists():
        print("Looking for optimized color space images from color_spaces_v2...")
        grayscale_dir = color_spaces_dir / "grayscale"
        hsv_dir = color_spaces_dir / "hsv"
        
        if grayscale_dir.exists():
            grayscale_paths = []
            matched_count = 0
            for img_path in image_paths:
                # normalized_v2/dataset/images/class/image.npy -> color_spaces_v2/grayscale/dataset/images/class/image.png
                try:
                    # .npy를 제거하고 경로에서 normalized_v2를 color_spaces_v2/grayscale로 교체
                    path_str = str(img_path)
                    if 'normalized_v2' in path_str:
                        # .npy 확장자 제거 후 .png로 변경
                        gray_path_str = path_str.replace('normalized_v2', 'color_spaces_v2/grayscale')
                        gray_path_str = gray_path_str.replace('.npy', '.png')
                        gray_path = Path(gray_path_str)
                        if not gray_path.exists():
                            # .png가 없으면 .jpg 시도
                            gray_path_str = gray_path_str.replace('.png', '.jpg')
                            gray_path = Path(gray_path_str)
                        if gray_path.exists():
                            grayscale_paths.append(gray_path)
                            matched_count += 1
                        else:
                            grayscale_paths.append(None)
                    else:
                        grayscale_paths.append(None)
                except Exception:
                    grayscale_paths.append(None)
            print(f"  ✓ Found {matched_count}/{len(image_paths)} grayscale images")
        
        if hsv_dir.exists():
            hsv_paths = []
            matched_count = 0
            for img_path in image_paths:
                try:
                    path_str = str(img_path)
                    if 'normalized_v2' in path_str:
                        hsv_path_str = path_str.replace('normalized_v2', 'color_spaces_v2/hsv')
                        hsv_path_str = hsv_path_str.replace('.npy', '.png')
                        hsv_path = Path(hsv_path_str)
                        if not hsv_path.exists():
                            hsv_path_str = hsv_path_str.replace('.png', '.jpg')
                            hsv_path = Path(hsv_path_str)
                        if hsv_path.exists():
                            hsv_paths.append(hsv_path)
                            matched_count += 1
                        else:
                            hsv_paths.append(None)
                    else:
                        hsv_paths.append(None)
                except Exception:
                    hsv_paths.append(None)
            print(f"  ✓ Found {matched_count}/{len(image_paths)} HSV images")
    
    # Create label mapping
    unique_labels = sorted(set(labels))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    label_indices = np.array([label_to_idx[label] for label in labels], dtype=np.int32)
    
    # Extract features using generator pattern
    print(f"\nExtracting features for {split} split...")
    features_list = []
    processed_count = 0
    failed_count = 0
    
    # Prepare grayscale/hsv paths for generator
    gray_paths_list = grayscale_paths if grayscale_paths else None
    hsv_paths_list = hsv_paths if hsv_paths else None
    
    for img, gray_img, hsv_img, img_path in tqdm(
        image_generator(image_paths, gray_paths_list, hsv_paths_list), 
        total=len(image_paths), desc=f"Extracting {split}"):
        try:
            # 입력 이미지 품질 검증
            if img is None:
                failed_count += 1
                features_list.append(np.zeros(extractor.get_feature_dim(), dtype=np.float32))
                continue
            
            # 빈 이미지 필터링 (모든 픽셀이 동일한 값)
            if len(img.shape) == 3:
                img_flat = img.reshape(-1, img.shape[-1])
            else:
                img_flat = img.flatten()
            
            if img_flat.std() < 1e-6:  # 거의 모든 픽셀이 동일
                print(f"Warning: Low variance image detected: {img_path}")
                failed_count += 1
                features_list.append(np.zeros(extractor.get_feature_dim(), dtype=np.float32))
                continue
            
            features = extractor.extract(img, gray_img, hsv_img)
            features_list.append(features)
            processed_count += 1
        except Exception as e:
            print(f"Warning: Failed to extract features from {img_path}: {e}")
            failed_count += 1
            # Add zero vector as fallback
            features_list.append(np.zeros(extractor.get_feature_dim(), dtype=np.float32))
    
    # Convert to numpy array
    if features_list:
        features_array = np.array(features_list, dtype=np.float32)
        labels_array = label_indices[:len(features_array)]
    else:
        print(f"⚠️  No features extracted for {split} split")
        return np.array([]), np.array([]), {'split': split, 'num_samples': 0}
    
    # Save features
    ensure_dir(output_dir)
    save_features_batch(features_array, labels_array, output_dir, split)
    
    # Save label mapping
    label_mapping = {idx: label for label, idx in label_to_idx.items()}
    with open(output_dir / f"{split}_label_mapping.json", 'w') as f:
        json.dump(label_mapping, f, indent=2)
    
    # Statistics
    stats = {
        'split': split,
        'num_samples': len(features_array),
        'feature_dim': features_array.shape[1],
        'processed': processed_count,
        'failed': failed_count,
        'num_classes': len(unique_labels),
        'cached': False
    }
    
    return features_array, labels_array, stats


def run_feature_extraction_pipeline(
    config_path: Path,
    feature_type: str = "combined",
    use_cache: bool = True,
    splits: Optional[list] = None
) -> Dict:
    """
    Run the complete feature extraction pipeline.
    
    Args:
        config_path: Path to config.yaml
        feature_type: Type of features to extract (hog, color_hist, lbp, combined, etc.)
        use_cache: Whether to use cached features
        splits: List of splits to process (None = all splits)
        
    Returns:
        Dictionary with extraction results
    """
    # Load configuration
    config = load_config(config_path)
    
    # Setup paths
    project_root = Path(__file__).parent.parent.parent
    processed_dir = project_root / config['data']['processed_dir'] / "normalized_v2"
    # v2 특징 저장 (개선된 버전)
    features_dir = project_root / config['data']['features_dir'] / "combined_v2"
    
    print("\n" + "="*60)
    print("Feature Extraction Pipeline")
    print("="*60)
    print(f"Processed images: {processed_dir}")
    print(f"Features output: {features_dir}")
    print(f"Feature type: {feature_type}")
    print("="*60)
    
    # Initialize extractor
    if feature_type == "combined":
        extractor = CombinedExtractor(config)
    else:
        # For individual feature types, create a minimal config
        # This would require refactoring - for now, use combined
        print(f"⚠️  Individual feature type extraction not yet implemented. Using combined.")
        extractor = CombinedExtractor(config)
    
    feature_info = extractor.get_feature_info()
    print(f"\nFeature extractors: {feature_info['extractors']}")
    print(f"Total feature dimension: {feature_info['total_dim']}")
    print(f"Feature dimensions: {feature_info['feature_dims']}")
    
    # Determine splits to process
    # Check if split directories exist, otherwise process all data as 'train'
    if splits is None:
        # Check if split directories exist
        has_splits = any((processed_dir / split).exists() for split in ['train', 'val', 'test'])
        if has_splits:
            splits = ['train', 'val', 'test']
        else:
            # No split directories, process all data as 'train'
            splits = ['train']
            print("⚠️  No train/val/test splits found. Processing all data as 'train' split.")
    
    # Process each split
    results = {}
    all_stats = []
    
    for split in splits:
        print(f"\n{'='*60}")
        print(f"Processing {split} split...")
        print(f"{'='*60}")
        
        split_output_dir = features_dir / feature_type / split
        features, labels, stats = extract_features_for_split(
            processed_dir,
            split_output_dir,
            split,
            extractor,
            config,
            use_cache=use_cache
        )
        
        results[split] = {
            'features': features,
            'labels': labels,
            'stats': stats
        }
        all_stats.append(stats)
        
        if len(features) > 0:
            print(f"\n✓ {split} split complete:")
            print(f"  - Samples: {stats['num_samples']}")
            print(f"  - Feature dim: {stats['feature_dim']}")
            print(f"  - Classes: {stats['num_classes']}")
    
    # Save overall statistics
    overall_stats = {
        'feature_type': feature_type,
        'feature_info': feature_info,
        'splits': all_stats
    }
    
    stats_path = features_dir / feature_type / "extraction_stats.json"
    ensure_dir(stats_path.parent)
    with open(stats_path, 'w') as f:
        json.dump(overall_stats, f, indent=2, default=str)
    
    # Create visualizations for each split
    print("\n" + "="*60)
    print("Creating Feature Extraction Visualizations")
    print("="*60)
    
    vis_output_dir = project_root / "data" / "visualization" / "feature_extraction" / feature_type
    ensure_dir(vis_output_dir)
    
    for split in splits:
        if len(results[split]['features']) > 0:
            print(f"\nVisualizing {split} split...")
            split_features_dir = features_dir / feature_type / split
            feat_vis.create_feature_extraction_report(
                split_features_dir,
                vis_output_dir,
                split=split
            )
    
    print("\n" + "="*60)
    print("Feature Extraction Complete!")
    print("="*60)
    print(f"Results saved to: {features_dir / feature_type}")
    print(f"Statistics saved to: {stats_path}")
    print(f"Visualizations saved to: {vis_output_dir}")
    print("="*60)
    
    return results


def main():
    """Main entry point for feature extraction."""
    parser = argparse.ArgumentParser(description='Extract features from preprocessed images')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config.yaml')
    parser.add_argument('--feature_type', type=str, default='combined',
                       choices=['hog', 'color_hist', 'lbp', 'gradient', 'texture', 'combined'],
                       help='Type of features to extract')
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable cache (re-extract all features)')
    parser.add_argument('--splits', nargs='+', default=None,
                       choices=['train', 'val', 'test'],
                       help='Specific splits to process (default: all)')
    
    args = parser.parse_args()
    
    config_path = Path(__file__).parent.parent.parent / args.config
    
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    run_feature_extraction_pipeline(
        config_path,
        feature_type=args.feature_type,
        use_cache=not args.no_cache,
        splits=args.splits
    )


if __name__ == "__main__":
    main()
