"""
기존 실험 결과 JSON을 읽어서 시각화만 재생성
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def visualize_blur_comparison_from_json(json_path: Path, output_dir: Path):
    """JSON 결과를 읽어서 시각화"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    stats = data['statistics']
    
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
    
    # 결과 요약 출력
    print("\n" + "=" * 60)
    print("실험 결과 요약")
    print("=" * 60)
    for i, method in enumerate(methods):
        print(f"\n{methods_clean[i]}:")
        print(f"  노이즈 감소: {noise_reduction[i]:.1f} ± {noise_std[i]:.1f}")
        print(f"  HOG 유사도: {hog_similarity[i]:.3f} ± {hog_std[i]:.3f}")
        print(f"  최종 노이즈: {final_noise[i]:.1f} ± {final_noise_std[i]:.1f}")
        print(f"  종합 점수: {composite_score[i]:.3f}")
    
    print(f"\n🏆 최고 방법: {methods_clean[best_idx]} (점수: {composite_score[best_idx]:.3f})")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Blur 실험 결과 시각화 재생성')
    parser.add_argument('--json_path', type=str, 
                       default='data/experiments/blur_comparison/blur_comparison_results.json',
                       help='JSON 결과 파일 경로')
    parser.add_argument('--output_dir', type=str, 
                       default='data/experiments/blur_comparison',
                       help='출력 디렉토리')
    
    args = parser.parse_args()
    
    json_path = Path(args.json_path)
    output_dir = Path(args.output_dir)
    
    if not json_path.exists():
        print(f"Error: JSON file not found: {json_path}")
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    visualize_blur_comparison_from_json(json_path, output_dir)
