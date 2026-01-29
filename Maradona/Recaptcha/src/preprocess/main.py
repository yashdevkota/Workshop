"""
Main preprocessing pipeline.
Orchestrates all preprocessing steps sequentially.
"""
import sys
import argparse
from pathlib import Path
from typing import Optional
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.preprocess.utils import load_config, ensure_dir
from src.preprocess.data_quality import (
    detect_duplicate_images, 
    check_filename_duplicates,
    analyze_class_distribution,
    visualize_class_distribution,
    get_representative_images
)
from src.preprocess.remove_duplicates import (
    remove_duplicate_images,
    filter_classes
)
from src.preprocess.analyze import analyze_dataset_structure
from src.preprocess.resize import resize_dataset
from src.preprocess.noise_reduction import reduce_noise_dataset
from src.preprocess.clahe import apply_clahe_dataset
from src.preprocess.color_space import convert_color_spaces_dataset
from src.preprocess.gamma_correction import apply_gamma_correction_dataset
from src.preprocess.normalize import normalize_dataset
import shutil

def merge_classes_step(base_dir: Path) -> dict:
    """
    Merge classes that are conceptually identical.
    Focus on merging 'TLight' into 'Traffic Light'.
    """
    merged_count = 0
    target_class = "Traffic Light"
    source_classes = ["TLight"]
    
    stats = {
        'target_class': target_class,
        'sources': source_classes,
        'merged_images': 0,
        'removed_dirs': []
    }
    
    # 1. raw data에서 통합
    for source_name in source_classes:
        # Find all directories matching source_name (case-insensitive)
        for path in base_dir.rglob("*"):
            if path.is_dir() and path.name.lower() == source_name.lower():
                target_dir = path.parent / target_class
                ensure_dir(target_dir)
                
                # Move all images
                image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
                for img_path in path.iterdir():
                    if img_path.suffix.lower() in image_extensions:
                        # Handle filename collisions
                        new_path = target_dir / img_path.name
                        if new_path.exists():
                            new_path = target_dir / f"merged_{img_path.name}"
                        
                        shutil.move(str(img_path), str(new_path))
                        merged_count += 1
                
                # Remove source dir if empty
                if not any(path.iterdir()):
                    path.rmdir()
                    stats['removed_dirs'].append(str(path))
    
    stats['merged_images'] = merged_count
    print(f"✓ Merged {merged_count} images from {source_classes} into {target_class}")
    return stats


def run_preprocessing_pipeline(config_path: Path, 
                               step: Optional[str] = None,
                               skip_analysis: bool = False) -> dict:
    """
    Run the complete preprocessing pipeline.
    
    Args:
        config_path: Path to config.yaml
        step: Specific step to run (None = all steps)
        skip_analysis: Whether to skip dataset analysis
        
    Returns:
        Dictionary with pipeline results
    """
    # Load configuration
    config = load_config(config_path)
    preprocess_config = config['preprocessing']
    vis_config = config['visualization']
    
    # Setup paths
    project_root = Path(__file__).parent.parent.parent
    raw_data_dir = project_root / config['data']['raw_dir']
    processed_dir = project_root / config['data']['processed_dir']
    visualization_dir = project_root / "data" / "visualization"
    ensure_dir(visualization_dir)
    
    print("\n" + "="*60)
    print("reCAPTCHA Preprocessing Pipeline")
    print("="*60)
    print(f"Raw data directory: {raw_data_dir}")
    print(f"Processed data directory: {processed_dir}")
    print("="*60)
    
    results = {}
    # 단계별 실행 시 이전 단계의 출력을 입력으로 사용
    if step == "clahe":
        # CLAHE를 단독 실행할 때는 resize 결과를 입력으로 사용
        resize_output = processed_dir / "resized"
        if resize_output.exists():
            # 이미지 파일이 있는지 확인
            image_files = list(resize_output.rglob("*.png")) + list(resize_output.rglob("*.jpg")) + \
                         list(resize_output.rglob("*.jpeg"))
            if image_files:
                current_dir = resize_output
                print(f"✓ Using existing resized images from: {current_dir}")
            else:
                current_dir = raw_data_dir
                print(f"⚠️  Resized images not found, using raw data: {current_dir}")
        else:
            current_dir = raw_data_dir
            print(f"⚠️  Resized directory not found, using raw data: {current_dir}")
    elif step == "noise_reduction":
        # Noise Reduction만 실행할 때는 CLAHE v2 결과를 우선 확인
        clahe_v2_output = processed_dir / "clahe_v2"
        if clahe_v2_output.exists():
            image_files = list(clahe_v2_output.rglob("*.png")) + list(clahe_v2_output.rglob("*.jpg")) + \
                         list(clahe_v2_output.rglob("*.jpeg"))
            if image_files:
                current_dir = clahe_v2_output
                print(f"✓ Using existing CLAHE v2 images from: {current_dir}")
            else:
                # CLAHE v2가 없으면 기존 CLAHE 확인
                clahe_output = processed_dir / "clahe"
                if clahe_output.exists():
                    image_files = list(clahe_output.rglob("*.png")) + list(clahe_output.rglob("*.jpg")) + \
                                 list(clahe_output.rglob("*.jpeg"))
                    if image_files:
                        current_dir = clahe_output
                        print(f"⚠️  Using existing CLAHE (v1) images from: {current_dir}")
                    else:
                        current_dir = raw_data_dir
                        print(f"⚠️  No CLAHE images found, using raw data: {current_dir}")
                else:
                    current_dir = raw_data_dir
                    print(f"⚠️  No CLAHE directory found, using raw data: {current_dir}")
        else:
            # CLAHE v2가 없으면 기존 CLAHE 확인
            clahe_output = processed_dir / "clahe"
            if clahe_output.exists():
                image_files = list(clahe_output.rglob("*.png")) + list(clahe_output.rglob("*.jpg")) + \
                             list(clahe_output.rglob("*.jpeg"))
                if image_files:
                    current_dir = clahe_output
                    print(f"⚠️  Using existing CLAHE (v1) images from: {current_dir}")
                else:
                    current_dir = raw_data_dir
                    print(f"⚠️  No CLAHE images found, using raw data: {current_dir}")
            else:
                current_dir = raw_data_dir
                print(f"⚠️  No CLAHE directory found, using raw data: {current_dir}")
    else:
        current_dir = raw_data_dir
    
    # Step 0: Data Quality Checks (🚨 Critical!)
    if (step is None or step == "data_quality") and config.get('data_quality', {}).get('duplicate_detection', {}).get('enabled', True):
        print("\n[Step 0] Data Quality Checks")
        quality_output = processed_dir / "quality_checks"
        ensure_dir(quality_output)
        
        # Find all images
        import os
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
        all_image_paths = []
        for root, dirs, files in os.walk(raw_data_dir):
            for f in files:
                if Path(f).suffix.lower() in image_extensions:
                    all_image_paths.append(Path(root) / f)
        
        print(f"Found {len(all_image_paths)} images for quality checks")
        
        # 1. Class Distribution Analysis
        print("\n--- Analyzing Class Distribution ---")
        class_dist = analyze_class_distribution(raw_data_dir)
        visualize_class_distribution(class_dist, quality_output, visualization_dir)
        results['class_distribution'] = class_dist
        
        # Save class distribution
        with open(quality_output / "class_distribution.json", 'w') as f:
            json.dump(class_dist, f, indent=2, default=str)
        
        # 2. Duplicate Detection
        quality_config = config.get('data_quality', {})
        dup_config = quality_config.get('duplicate_detection', {})
        
        if dup_config.get('enabled', True):
            print("\n--- Detecting Duplicate Images ---")
            
            # Filename-based duplicate check (fast)
            filename_duplicates = check_filename_duplicates(all_image_paths)
            if filename_duplicates:
                print(f"Found {len(filename_duplicates)} files with duplicate filenames")
                with open(quality_output / "filename_duplicates.json", 'w') as f:
                    json.dump({k: [str(p) for p in v] for k, v in filename_duplicates.items()}, 
                             f, indent=2)
            
            # Perceptual hash-based duplicate check
            duplicate_groups = detect_duplicate_images(
                all_image_paths,
                similarity_threshold=dup_config.get('similarity_threshold', 0.95),
                method=dup_config.get('method', 'hash')
            )
            
            if duplicate_groups:
                print(f"Found {len(duplicate_groups)} duplicate groups")
                # Save duplicate groups
                with open(quality_output / "duplicate_groups.json", 'w') as f:
                    json.dump({str(k): [str(p) for p in v] 
                              for k, v in duplicate_groups.items()}, f, indent=2)
                
                # Get representative images (keep first, remove others)
                if dup_config.get('remove_duplicates', True):
                    representative_images = get_representative_images(duplicate_groups)
                    print(f"Keeping {len(representative_images)} representative images")
                    results['duplicates'] = {
                        'total_groups': len(duplicate_groups),
                        'total_duplicates': sum(len(v) - 1 for v in duplicate_groups.values()),
                        'representative_images': len(representative_images)
                    }
                else:
                    results['duplicates'] = {
                        'total_groups': len(duplicate_groups),
                        'total_duplicates': sum(len(v) - 1 for v in duplicate_groups.values())
                    }
            else:
                print("✓ No duplicate images found")
                results['duplicates'] = {'total_groups': 0, 'total_duplicates': 0}
        
        # Save quality check results
        with open(quality_output / "quality_checks.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✓ Data quality checks complete. Results saved to {quality_output}")
        
        # Apply duplicate removal and class filtering if enabled
        quality_config = config.get('data_quality', {})
        dup_config = quality_config.get('duplicate_detection', {})
        
        if dup_config.get('remove_duplicates', True) and duplicate_groups:
            print("\n" + "="*60)
            print("Applying duplicate removal...")
            print("="*60)
            
            duplicate_groups_path = quality_output / "duplicate_groups.json"
            removal_stats = remove_duplicate_images(
                duplicate_groups_path,
                quality_output,
                action='move'  # Move to removed_duplicates folder (safer than delete)
            )
            results['duplicate_removal'] = removal_stats
        
        # Filter out problematic classes
        classes_to_exclude = ['visualizations']  # Add 'Mountain' if needed
        if classes_to_exclude:
            print("\n" + "="*60)
            print(f"Filtering out classes: {classes_to_exclude}")
            print("="*60)
            
            filtering_stats = filter_classes(
                raw_data_dir,
                classes_to_exclude,
                quality_output,
                action='move'
            )
            results['class_filtering'] = filtering_stats
        
        # Save updated results
        with open(quality_output / "quality_checks.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    # [NEW STEP] Merge TLight and Traffic Light classes
    if step is None or step == "merge_classes":
        print("\n[Step 0.5] Merging Classes (TLight -> Traffic Light)")
        merge_stats = merge_classes_step(raw_data_dir)
        results['class_merging'] = merge_stats
        
        # Merge stats saved to quality checks
        quality_output = processed_dir / "quality_checks"
        ensure_dir(quality_output)
        with open(quality_output / "class_merging.json", 'w') as f:
            json.dump(merge_stats, f, indent=2)
    
    # Step 1: Dataset Structure Analysis
    if not skip_analysis and (step is None or step == "analyze"):
        print("\n[Step 0] Dataset Structure Analysis")
        analysis_output = processed_dir / "analysis"
        analysis_result = analyze_dataset_structure(
            raw_data_dir, 
            analysis_output,
            num_samples=0,  # 0 = analyze all images
            vis_dir=visualization_dir
        )
        results['analysis'] = analysis_result
        
        # Update recommendations in config if needed
        recommendations = analysis_result.get('recommendations', {})
        if recommendations.get('suggested_noise_threshold'):
            preprocess_config['noise_reduction']['threshold'] = recommendations['suggested_noise_threshold']
        if recommendations.get('suggested_brightness_threshold'):
            preprocess_config['gamma_correction']['threshold'] = recommendations['suggested_brightness_threshold']
    
    # Step 1: Resize
    if step is None or step == "resize":
        print("\n[Step 1] Image Resizing")
        resize_output = processed_dir / "resized"
        resize_result = resize_dataset(
            current_dir,
            resize_output,
            target_size=tuple(preprocess_config['target_size']),
            method=preprocess_config['resize_method'],
            num_samples=vis_config['num_samples'],
            vis_dir=visualization_dir
        )
        results['resize'] = resize_result
        current_dir = resize_output
    
    # Step 2: CLAHE (먼저 적용하여 엣지 강조) - v2 파이프라인
    if (step is None or step == "clahe") and preprocess_config['clahe']['enabled']:
        print("\n[Step 2] CLAHE (v2: CLAHE → Gaussian 파이프라인)")
        clahe_output = processed_dir / "clahe_v2"  # v2로 저장
        clahe_vis_dir = visualization_dir / "clahe_v2"  # v2 시각화 폴더
        ensure_dir(clahe_vis_dir)
        clahe_result = apply_clahe_dataset(
            current_dir,
            clahe_output,
            clip_limit=preprocess_config['clahe']['clip_limit'],
            tile_grid_size=tuple(preprocess_config['clahe']['tile_grid_size']),
            adaptive=True,
            brightness_threshold=preprocess_config['clahe']['brightness_threshold'],
            num_samples=vis_config['num_samples'],
            vis_dir=clahe_vis_dir  # v2 시각화 폴더 사용
        )
        results['clahe'] = clahe_result
        current_dir = clahe_output
    
    # Step 3: Noise Reduction (CLAHE 후 Gaussian으로 노이즈 제거) - v2 파이프라인
    # 실험 결과: CLAHE → Gaussian이 최적 성능 (노이즈 제거 + HOG 보존 균형)
    if (step is None or step == "noise_reduction") and preprocess_config['noise_reduction']['enabled']:
        print("\n[Step 3] Noise Reduction (Gaussian Blur) - v2: CLAHE → Gaussian 파이프라인")
        # Noise Reduction 단독 실행 시 CLAHE v2 결과를 입력으로 사용
        if step == "noise_reduction":
            clahe_output = processed_dir / "clahe_v2"
            if clahe_output.exists():
                image_files = list(clahe_output.rglob("*.png")) + list(clahe_output.rglob("*.jpg")) + \
                             list(clahe_output.rglob("*.jpeg"))
                if image_files:
                    current_dir = clahe_output
                    print(f"✓ Using existing CLAHE v2 images from: {current_dir}")
        
        noise_output = processed_dir / "denoised_v2"  # v2로 저장
        noise_vis_dir = visualization_dir / "noise_reduction_v2"  # v2 시각화 폴더
        ensure_dir(noise_vis_dir)
        noise_result = reduce_noise_dataset(
            current_dir,
            noise_output,
            method=preprocess_config['noise_reduction']['method'],  # Use config value
            threshold=preprocess_config['noise_reduction']['threshold'],
            gaussian_kernel=tuple(preprocess_config['noise_reduction']['gaussian_kernel']),
            gaussian_sigma=preprocess_config['noise_reduction'].get('gaussian_sigma', 1.0),
            bilateral_params={
                'd': preprocess_config['noise_reduction'].get('bilateral_d', 5),
                'sigma_color': preprocess_config['noise_reduction'].get('bilateral_sigma_color', 50),
                'sigma_space': preprocess_config['noise_reduction'].get('bilateral_sigma_space', 50)
            },
            num_samples=vis_config['num_samples'],
            vis_dir=noise_vis_dir  # v2 시각화 폴더 사용
        )
        results['noise_reduction'] = noise_result
        current_dir = noise_output
    
    # Step 4: Color Space Conversion - v2 파이프라인
    if step is None or step == "color_space":
        print("\n[Step 4] Color Space Conversion (v2: CLAHE → Gaussian 파이프라인)")
        # Color Space 단독 실행 시 denoised_v2 결과를 입력으로 사용
        if step == "color_space":
            denoised_v2_output = processed_dir / "denoised_v2"
            if denoised_v2_output.exists():
                image_files = list(denoised_v2_output.rglob("*.png")) + list(denoised_v2_output.rglob("*.jpg")) + \
                             list(denoised_v2_output.rglob("*.jpeg"))
                if image_files:
                    current_dir = denoised_v2_output
                    print(f"✓ Using existing denoised_v2 images from: {current_dir}")
        
        color_output = processed_dir / "color_spaces_v2"  # v2로 저장
        color_vis_dir = visualization_dir / "color_space_v2"  # v2 시각화 폴더
        ensure_dir(color_vis_dir)
        color_result = convert_color_spaces_dataset(
            current_dir,
            color_output,
            convert_hsv=preprocess_config['color_space']['hsv'],
            convert_lab=preprocess_config['color_space']['lab'],
            convert_grayscale=preprocess_config['color_space']['grayscale'],
            num_samples=vis_config['num_samples'],
            vis_dir=color_vis_dir  # v2 시각화 폴더 사용
        )
        results['color_space'] = color_result
        # Use grayscale for next steps if available
        if preprocess_config['color_space']['grayscale']:
            current_dir = color_output / "grayscale"
    
    # Step 5: Gamma Correction - v2 파이프라인
    if (step is None or step == "gamma") and preprocess_config['gamma_correction']['enabled']:
        print("\n[Step 5] Gamma Correction (v2: CLAHE → Gaussian 파이프라인)")
        # Gamma Correction 단독 실행 시 color_spaces_v2/grayscale 결과를 입력으로 사용
        if step == "gamma":
            color_spaces_v2_output = processed_dir / "color_spaces_v2" / "grayscale"
            if color_spaces_v2_output.exists():
                image_files = list(color_spaces_v2_output.rglob("*.png")) + list(color_spaces_v2_output.rglob("*.jpg")) + \
                             list(color_spaces_v2_output.rglob("*.jpeg"))
                if image_files:
                    current_dir = color_spaces_v2_output
                    print(f"✓ Using existing color_spaces_v2/grayscale images from: {current_dir}")
        
        gamma_output = processed_dir / "gamma_corrected_v2"  # v2로 저장
        gamma_vis_dir = visualization_dir / "gamma_correction_v2"  # v2 시각화 폴더
        ensure_dir(gamma_vis_dir)
        gamma_result = apply_gamma_correction_dataset(
            current_dir,
            gamma_output,
            gamma=preprocess_config['gamma_correction']['gamma'],
            threshold=preprocess_config['gamma_correction']['threshold'],
            num_samples=vis_config['num_samples'],
            vis_dir=gamma_vis_dir  # v2 시각화 폴더 사용
        )
        results['gamma_correction'] = gamma_result
        current_dir = gamma_output
    
    # Step 6: Normalization - v2 파이프라인
    if step is None or step == "normalize":
        print("\n[Step 6] Normalization (v2: CLAHE → Gaussian 파이프라인)")
        # Normalization 단독 실행 시 gamma_corrected_v2 결과를 입력으로 사용
        if step == "normalize":
            gamma_corrected_v2_output = processed_dir / "gamma_corrected_v2"
            if gamma_corrected_v2_output.exists():
                image_files = list(gamma_corrected_v2_output.rglob("*.png")) + list(gamma_corrected_v2_output.rglob("*.jpg")) + \
                             list(gamma_corrected_v2_output.rglob("*.jpeg"))
                if image_files:
                    current_dir = gamma_corrected_v2_output
                    print(f"✓ Using existing gamma_corrected_v2 images from: {current_dir}")
        
        normalize_output = processed_dir / "normalized_v2"  # v2로 저장
        normalize_vis_dir = visualization_dir / "normalization_v2"  # v2 시각화 폴더
        ensure_dir(normalize_vis_dir)
        normalize_result = normalize_dataset(
            current_dir,
            normalize_output,
            method=preprocess_config['normalization']['method'],
            range_min=preprocess_config['normalization']['range'][0],
            range_max=preprocess_config['normalization']['range'][1],
            save_as_float=True,  # Save as .npy for feature extraction
            num_samples=vis_config['num_samples'],
            vis_dir=normalize_vis_dir  # v2 시각화 폴더 사용
        )
        results['normalization'] = normalize_result
        current_dir = normalize_output
    
    # Save pipeline results
    results_path = processed_dir / "preprocessing_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n" + "="*60)
    print("Preprocessing Pipeline Complete!")
    print("="*60)
    print(f"Results saved to: {results_path}")
    print(f"Final processed images: {current_dir}")
    print("="*60)
    
    return results


def main():
    """Main entry point for preprocessing pipeline."""
    parser = argparse.ArgumentParser(description='Run preprocessing pipeline')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config.yaml')
    parser.add_argument('--step', type=str, default=None,
                        choices=['data_quality', 'analyze', 'resize', 'noise_reduction', 'clahe', 
                               'color_space', 'gamma', 'normalize', 'merge_classes'],
                       help='Run specific step only')
    parser.add_argument('--remove-duplicates', action='store_true',
                       help='Remove duplicate images after detection')
    parser.add_argument('--exclude-classes', nargs='+', default=['visualizations'],
                       help='Classes to exclude (default: visualizations)')
    parser.add_argument('--skip-analysis', action='store_true',
                       help='Skip dataset analysis step')
    
    args = parser.parse_args()
    
    config_path = Path(__file__).parent.parent.parent / args.config
    
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    run_preprocessing_pipeline(config_path, args.step, args.skip_analysis)


if __name__ == "__main__":
    main()
