"""
Blur 방법 및 적용 시점 비교 실험
- Gaussian vs Bilateral
- CLAHE 전 vs CLAHE 후
총 4가지 조합 비교
"""
import sys
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.preprocess.utils import load_image, save_image, ensure_dir
from src.preprocess.noise_reduction import (
    apply_gaussian_blur, 
    apply_bilateral_filter,
    calculate_noise_level
)
from src.preprocess.clahe import apply_clahe, get_adaptive_clahe_params


def extract_hog_features(image: np.ndarray, cell_size=(8, 8), block_size=(2, 2), nbins=9):
    """HOG 특징 추출"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # HOG 계산
    hog = cv2.HOGDescriptor(
        _winSize=(gray.shape[1], gray.shape[0]),
        _blockSize=(block_size[0] * cell_size[0], block_size[1] * cell_size[1]),
        _blockStride=(cell_size[0], cell_size[1]),
        _cellSize=cell_size,
        _nbins=nbins
    )
    features = hog.compute(gray)
    return features.flatten().astype(np.float32)


def compare_blur_methods(input_dir: Path, output_dir: Path, num_samples: int = 50):
    """
    Blur 방법 및 적용 시점 비교 실험
    
    비교 항목:
    1. Gaussian Blur → CLAHE
    2. CLAHE → Gaussian Blur
    3. Bilateral Filter → CLAHE
    4. CLAHE → Bilateral Filter
    """
    print("=" * 60)
    print("Blur 방법 및 적용 시점 비교 실험")
    print("=" * 60)
    
    ensure_dir(output_dir)
    
    # 실험 결과 저장용
    results = {
        'gaussian_before_clahe': [],
        'gaussian_after_clahe': [],
        'bilateral_before_clahe': [],
        'bilateral_after_clahe': []
    }
    
    # 이미지 찾기
    import os
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    image_paths = []
    
    for root, _, files in os.walk(input_dir):
        for f in files:
            if Path(f).suffix.lower() in image_extensions:
                image_paths.append(Path(root) / f)
    
    # 샘플 선택
    if len(image_paths) > num_samples:
        indices = np.random.choice(len(image_paths), num_samples, replace=False)
        sample_paths = [image_paths[i] for i in indices]
    else:
        sample_paths = image_paths
    
    print(f"Testing {len(sample_paths)} images...")
    
    # 각 이미지에 대해 4가지 방법 적용
    for img_path in tqdm(sample_paths, desc="Processing"):
        original = load_image(img_path)
        if original is None:
            continue
        
        # 원본 HOG 특징
        hog_original = extract_hog_features(original)
        
        # 1. Gaussian Blur → CLAHE
        gaussian_blurred = apply_gaussian_blur(original, (3, 3), sigma=1.0)
        clip_limit, tile_size = get_adaptive_clahe_params(gaussian_blurred)
        result1 = apply_clahe(gaussian_blurred, clip_limit, tile_size)
        hog1 = extract_hog_features(result1)
        
        # 2. CLAHE → Gaussian Blur
        clip_limit, tile_size = get_adaptive_clahe_params(original)
        clahe_result = apply_clahe(original, clip_limit, tile_size)
        result2 = apply_gaussian_blur(clahe_result, (3, 3), sigma=1.0)
        hog2 = extract_hog_features(result2)
        
        # 3. Bilateral Filter → CLAHE
        bilateral_result = apply_bilateral_filter(original, d=5, sigma_color=50, sigma_space=50)
        clip_limit, tile_size = get_adaptive_clahe_params(bilateral_result)
        result3 = apply_clahe(bilateral_result, clip_limit, tile_size)
        hog3 = extract_hog_features(result3)
        
        # 4. CLAHE → Bilateral Filter
        clip_limit, tile_size = get_adaptive_clahe_params(original)
        clahe_result2 = apply_clahe(original, clip_limit, tile_size)
        result4 = apply_bilateral_filter(clahe_result2, d=5, sigma_color=50, sigma_space=50)
        hog4 = extract_hog_features(result4)
        
        # 노이즈 레벨 측정
        noise_original = calculate_noise_level(original)
        noise1 = calculate_noise_level(result1)
        noise2 = calculate_noise_level(result2)
        noise3 = calculate_noise_level(result3)
        noise4 = calculate_noise_level(result4)
        
        # HOG 특징 유사도 (원본과 비교)
        sim1 = cosine_similarity([hog_original], [hog1])[0][0]
        sim2 = cosine_similarity([hog_original], [hog2])[0][0]
        sim3 = cosine_similarity([hog_original], [hog3])[0][0]
        sim4 = cosine_similarity([hog_original], [hog4])[0][0]
        
        # 결과 저장
        results['gaussian_before_clahe'].append({
            'noise_reduction': noise_original - noise1,
            'hog_similarity': float(sim1),
            'final_noise': float(noise1)
        })
        
        results['gaussian_after_clahe'].append({
            'noise_reduction': noise_original - noise2,
            'hog_similarity': float(sim2),
            'final_noise': float(noise2)
        })
        
        results['bilateral_before_clahe'].append({
            'noise_reduction': noise_original - noise3,
            'hog_similarity': float(sim3),
            'final_noise': float(noise3)
        })
        
        results['bilateral_after_clahe'].append({
            'noise_reduction': noise_original - noise4,
            'hog_similarity': float(sim4),
            'final_noise': float(noise4)
        })
    
    # 통계 계산
    stats = {}
    for method, data in results.items():
        if data:
            stats[method] = {
                'mean_noise_reduction': float(np.mean([d['noise_reduction'] for d in data])),
                'std_noise_reduction': float(np.std([d['noise_reduction'] for d in data])),
                'mean_hog_similarity': float(np.mean([d['hog_similarity'] for d in data])),
                'std_hog_similarity': float(np.std([d['hog_similarity'] for d in data])),
                'mean_final_noise': float(np.mean([d['final_noise'] for d in data])),
                'std_final_noise': float(np.std([d['final_noise'] for d in data]))
            }
    
    # 결과 저장
    with open(output_dir / 'blur_comparison_results.json', 'w') as f:
        json.dump({
            'results': results,
            'statistics': stats,
            'num_samples': len(sample_paths)
        }, f, indent=2, default=str)
    
    # 시각화
    visualize_blur_comparison(stats, output_dir)
    
    print("\n✓ Experiment complete!")
    print(f"Results saved to: {output_dir}")
    
    return stats


def visualize_blur_comparison(stats: dict, output_dir: Path):
    """비교 결과 시각화 - 개선된 버전"""
    methods = list(stats.keys())
    method_labels = {
        'gaussian_before_clahe': 'Gaussian\n→ CLAHE',
        'gaussian_after_clahe': 'CLAHE\n→ Gaussian',
        'bilateral_before_clahe': 'Bilateral\n→ CLAHE',
        'bilateral_after_clahe': 'CLAHE\n→ Bilateral'
    }
    
    # 한글 폰트 설정
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3, height_ratios=[1, 1, 0.8])
    fig.suptitle('Blur Method and Timing Comparison Experiment', 
                fontsize=18, fontweight='bold', y=0.98)
    
    methods_clean = [method_labels[m] for m in methods]
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    # 1. 노이즈 제거 효과 (상단 왼쪽)
    ax1 = fig.add_subplot(gs[0, 0])
    noise_reduction = [stats[m]['mean_noise_reduction'] for m in methods]
    noise_std = [stats[m]['std_noise_reduction'] for m in methods]
    
    bars1 = ax1.bar(range(len(methods_clean)), noise_reduction, yerr=noise_std, 
                    capsize=8, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_xticks(range(len(methods_clean)))
    ax1.set_xticklabels(methods_clean, fontsize=11, fontweight='bold')
    ax1.set_ylabel('Noise Reduction\n(Laplacian Variance)', fontsize=12, fontweight='bold')
    ax1.set_title('1. Noise Reduction Effect\n(Higher is Better)', 
                 fontsize=13, fontweight='bold', pad=10)
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    # 값 표시
    for i, (bar, val, std) in enumerate(zip(bars1, noise_reduction, noise_std)):
        height = bar.get_height()
        y_pos = height + std + 50 if height >= 0 else height - std - 100
        ax1.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{val:.1f}\n±{std:.1f}', ha='center', va='bottom' if height >= 0 else 'top',
                fontsize=10, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', 
                facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # 2. HOG 특징 유사도 (상단 중앙)
    ax2 = fig.add_subplot(gs[0, 1])
    hog_similarity = [stats[m]['mean_hog_similarity'] for m in methods]
    hog_std = [stats[m]['std_hog_similarity'] for m in methods]
    
    bars2 = ax2.bar(range(len(methods_clean)), hog_similarity, yerr=hog_std,
                    capsize=8, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_xticks(range(len(methods_clean)))
    ax2.set_xticklabels(methods_clean, fontsize=11, fontweight='bold')
    ax2.set_ylabel('HOG Feature Similarity\n(to Original)', fontsize=12, fontweight='bold')
    ax2.set_title('2. HOG Feature Preservation\n(Higher is Better)', 
                 fontsize=13, fontweight='bold', pad=10)
    ax2.set_ylim([0.94, 0.98])
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # 값 표시
    for i, (bar, val, std) in enumerate(zip(bars2, hog_similarity, hog_std)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + std + 0.002,
                f'{val:.3f}\n±{std:.3f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3',
                facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # 3. 최종 노이즈 레벨 (상단 오른쪽)
    ax3 = fig.add_subplot(gs[0, 2])
    final_noise = [stats[m]['mean_final_noise'] for m in methods]
    final_noise_std = [stats[m]['std_final_noise'] for m in methods]
    
    bars3 = ax3.bar(range(len(methods_clean)), final_noise, yerr=final_noise_std,
                    capsize=8, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax3.set_xticks(range(len(methods_clean)))
    ax3.set_xticklabels(methods_clean, fontsize=11, fontweight='bold')
    ax3.set_ylabel('Final Noise Level\n(Laplacian Variance)', fontsize=12, fontweight='bold')
    ax3.set_title('3. Final Image Quality\n(Lower is Better)', 
                 fontsize=13, fontweight='bold', pad=10)
    ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # 값 표시
    for i, (bar, val, std) in enumerate(zip(bars3, final_noise, final_noise_std)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + std + 50,
                f'{val:.1f}\n±{std:.1f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3',
                facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # 4. 종합 비교 (중앙 - 큰 차트)
    ax4 = fig.add_subplot(gs[1, :])
    
    # 정규화된 점수 계산
    # 노이즈 제거: 양수일수록 좋음 (정규화)
    noise_min, noise_max = min(noise_reduction), max(noise_reduction)
    noise_range = noise_max - noise_min if noise_max != noise_min else 1
    noise_norm = [(nr - noise_min) / noise_range for nr in noise_reduction]
    
    # HOG 유사도: 높을수록 좋음 (이미 0.94-0.98 범위)
    hog_min, hog_max = min(hog_similarity), max(hog_similarity)
    hog_range = hog_max - hog_min if hog_max != hog_min else 1
    hog_norm = [(hs - hog_min) / hog_range for hs in hog_similarity]
    
    # 최종 노이즈: 낮을수록 좋음 (역수로 변환 후 정규화)
    noise_final_max, noise_final_min = max(final_noise), min(final_noise)
    noise_final_range = noise_final_max - noise_final_min if noise_final_max != noise_final_min else 1
    noise_final_norm = [(noise_final_max - nf) / noise_final_range for nf in final_noise]
    
    # 종합 점수: 노이즈 제거 40% + HOG 보존 40% + 최종 품질 20%
    composite_score = [0.4 * n + 0.4 * h + 0.2 * nf 
                      for n, h, nf in zip(noise_norm, hog_norm, noise_final_norm)]
    
    bars4 = ax4.bar(range(len(methods_clean)), composite_score, 
                    color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax4.set_xticks(range(len(methods_clean)))
    ax4.set_xticklabels(methods_clean, fontsize=12, fontweight='bold')
    ax4.set_ylabel('Composite Score\n(Normalized)', fontsize=13, fontweight='bold')
    ax4.set_title('4. Overall Performance Score\n(Noise Reduction 40% + HOG Preservation 40% + Final Quality 20%)',
                 fontsize=14, fontweight='bold', pad=15)
    ax4.set_ylim([0, 1.1])
    ax4.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # 값 표시
    for i, (bar, val) in enumerate(zip(bars4, composite_score)):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.03,
                f'{val:.3f}', ha='center', va='bottom',
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                         alpha=0.9, edgecolor='black', linewidth=1.5))
    
    # 최고 방법 표시
    best_idx = np.argmax(composite_score)
    best_method = methods_clean[best_idx]
    bars4[best_idx].set_edgecolor('gold')
    bars4[best_idx].set_linewidth(3)
    ax4.text(best_idx, composite_score[best_idx] + 0.08,
            '🏆 BEST', ha='center', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='gold', 
                     alpha=0.9, edgecolor='black', linewidth=2))
    
    # 5. 요약 테이블 (하단)
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    # 테이블 데이터 준비
    table_data = []
    for i, method in enumerate(methods):
        table_data.append([
            methods_clean[i],
            f"{noise_reduction[i]:.1f} ± {noise_std[i]:.1f}",
            f"{hog_similarity[i]:.3f} ± {hog_std[i]:.3f}",
            f"{final_noise[i]:.1f} ± {final_noise_std[i]:.1f}",
            f"{composite_score[i]:.3f}"
        ])
    
    table = ax5.table(cellText=table_data,
                      colLabels=['Method', 'Noise Reduction', 'HOG Similarity', 
                                'Final Noise', 'Composite Score'],
                      cellLoc='center',
                      loc='center',
                      bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # 헤더 스타일
    for i in range(5):
        table[(0, i)].set_facecolor('#4A90E2')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 최고 방법 행 강조
    best_row = best_idx + 1
    for i in range(5):
        table[(best_row, i)].set_facecolor('#FFD700')
        table[(best_row, i)].set_text_props(weight='bold')
    
    # 일반 행 스타일
    for i in range(1, len(table_data) + 1):
        if i != best_row:
            for j in range(5):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#F5F5F5')
                else:
                    table[(i, j)].set_facecolor('white')
    
    plt.savefig(output_dir / 'blur_comparison.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Saved improved comparison visualization to {output_dir / 'blur_comparison.png'}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Blur 방법 및 적용 시점 비교 실험')
    parser.add_argument('--input_dir', type=str, default='data/processed/resized',
                       help='Input directory (resized images)')
    parser.add_argument('--output_dir', type=str, default='data/experiments/blur_comparison',
                       help='Output directory for experiment results')
    parser.add_argument('--num_samples', type=int, default=50,
                       help='Number of images to test')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        print("Please run resize step first: python src/preprocess/main.py --step resize")
        sys.exit(1)
    
    stats = compare_blur_methods(input_dir, output_dir, args.num_samples)
    
    # 결과 요약 출력
    print("\n" + "=" * 60)
    print("실험 결과 요약")
    print("=" * 60)
    
    method_labels = {
        'gaussian_before_clahe': 'Gaussian → CLAHE',
        'gaussian_after_clahe': 'CLAHE → Gaussian',
        'bilateral_before_clahe': 'Bilateral → CLAHE',
        'bilateral_after_clahe': 'CLAHE → Bilateral'
    }
    
    for method, stat in stats.items():
        print(f"\n{method_labels[method]}:")
        print(f"  노이즈 감소: {stat['mean_noise_reduction']:.1f} ± {stat['std_noise_reduction']:.1f}")
        print(f"  HOG 유사도: {stat['mean_hog_similarity']:.3f} ± {stat['std_hog_similarity']:.3f}")
        print(f"  최종 노이즈: {stat['mean_final_noise']:.1f} ± {stat['std_final_noise']:.1f}")
