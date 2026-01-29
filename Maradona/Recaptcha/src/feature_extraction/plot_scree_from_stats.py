"""
기존 PCA 통계에서 Scree Plot 생성.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI 없이 사용
import matplotlib.pyplot as plt
from pathlib import Path

def plot_scree_from_stats(stats_path: Path, output_path: Path):
    """기존 PCA 통계에서 Scree Plot 생성."""
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    explained_variance = np.array(stats['pca_stats']['explained_variance_ratio'])
    cumulative_variance = np.cumsum(explained_variance)
    
    # 주요 지점 찾기
    thresholds = [0.80, 0.85, 0.90, 0.95]
    threshold_dims = {}
    
    for threshold in thresholds:
        idx = np.argmax(cumulative_variance >= threshold)
        if cumulative_variance[idx] >= threshold:
            threshold_dims[threshold] = idx + 1
    
    # 그래프 생성
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('PCA Scree Plot Analysis', fontsize=16, fontweight='bold')
    
    # 1. 개별 설명 분산 (상위 50개)
    ax1 = axes[0, 0]
    n_plot = min(50, len(explained_variance))
    ax1.plot(range(1, n_plot + 1), explained_variance[:n_plot], 'bo-', linewidth=2, markersize=4)
    ax1.set_xlabel('Principal Component', fontsize=12)
    ax1.set_ylabel('Explained Variance Ratio', fontsize=12)
    ax1.set_title(f'Individual Explained Variance (Top {n_plot})', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. 누적 설명 분산 (전체)
    ax2 = axes[0, 1]
    n_plot = min(500, len(cumulative_variance))
    ax2.plot(range(1, n_plot + 1), cumulative_variance[:n_plot], 'go-', linewidth=2, markersize=2)
    for threshold, dim in threshold_dims.items():
        if dim <= n_plot:
            ax2.axvline(dim, color='r', linestyle='--', alpha=0.5)
            ax2.axhline(threshold, color='r', linestyle='--', alpha=0.5)
            ax2.plot(dim, threshold, 'ro', markersize=8)
            ax2.text(dim, threshold + 0.02, f'{threshold*100:.0f}%\n({dim}개)', 
                    ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    ax2.set_xlabel('Number of Components', fontsize=12)
    ax2.set_ylabel('Cumulative Explained Variance', fontsize=12)
    ax2.set_title('Cumulative Explained Variance', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.0])
    
    # 3. 누적 설명 분산 (상위 300개, 확대)
    ax3 = axes[1, 0]
    n_plot = min(300, len(cumulative_variance))
    ax3.plot(range(1, n_plot + 1), cumulative_variance[:n_plot], 'go-', linewidth=2, markersize=2)
    for threshold, dim in threshold_dims.items():
        if dim <= n_plot:
            ax3.axvline(dim, color='r', linestyle='--', alpha=0.5)
            ax3.axhline(threshold, color='r', linestyle='--', alpha=0.5)
            ax3.plot(dim, threshold, 'ro', markersize=6)
    ax3.set_xlabel('Number of Components', fontsize=12)
    ax3.set_ylabel('Cumulative Explained Variance', fontsize=12)
    ax3.set_title('Cumulative Explained Variance (Top 300)', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1.0])
    
    # 4. 통계 요약
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    stats_text = f"""PCA 분석 결과 요약
{'='*50}

원본 차원: {stats['feature_dim']:,}

주요 지점:"""
    for threshold in sorted(thresholds):
        if threshold in threshold_dims:
            dim = threshold_dims[threshold]
            stats_text += f"\n  {threshold*100:.0f}% 분산: {dim:,}개 주성분"
    
    stats_text += "\n\n상위 주성분 누적 분산:"
    for n in [50, 100, 150, 200, 250, 300]:
        if n <= len(cumulative_variance):
            stats_text += f"\n  {n:3d}개: {cumulative_variance[n-1]:.4f} ({cumulative_variance[n-1]*100:.2f}%)"
    
    stats_text += "\n\n권장 사항:"
    if 100 <= len(cumulative_variance):
        if cumulative_variance[99] >= 0.80:
            stats_text += "\n  ✓ 상위 100개로 80% 이상 달성"
    if 200 <= len(cumulative_variance):
        if cumulative_variance[199] >= 0.90:
            stats_text += "\n  ✓ 상위 200개로 90% 이상 달성"
    
    if 0.80 in threshold_dims:
        stats_text += f"\n  → 80%: {threshold_dims[0.80]}개 주성분"
    if 0.90 in threshold_dims:
        stats_text += f"\n  → 90%: {threshold_dims[0.90]}개 주성분"
    
    ax4.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 결과 출력
    print("="*70)
    print("PCA Scree Plot 분석 결과")
    print("="*70)
    print(f"\n원본 차원: {stats['feature_dim']:,}")
    print(f"\n주요 지점:")
    for threshold in sorted(thresholds):
        if threshold in threshold_dims:
            dim = threshold_dims[threshold]
            print(f"  {threshold*100:.0f}% 분산: {dim:,}개 주성분")
    
    print(f"\n상위 주성분 누적 분산:")
    for n in [50, 100, 150, 200, 250, 300]:
        if n <= len(cumulative_variance):
            print(f"  {n:3d}개: {cumulative_variance[n-1]:.4f} ({cumulative_variance[n-1]*100:.2f}%)")
    
    print("\n" + "="*70)
    print("최종 권장 사항")
    print("="*70)
    
    if 100 <= len(cumulative_variance):
        print(f"\n✅ 상위 100개 주성분: {cumulative_variance[99]:.4f} ({cumulative_variance[99]*100:.2f}%)")
        if cumulative_variance[99] >= 0.80:
            print("   → 100개 주성분으로 학습 가능!")
    
    if 200 <= len(cumulative_variance):
        print(f"\n✅ 상위 200개 주성분: {cumulative_variance[199]:.4f} ({cumulative_variance[199]*100:.2f}%)")
        if cumulative_variance[199] >= 0.90:
            print("   → 200개 주성분으로 학습 가능!")
    
    if 0.90 in threshold_dims:
        dim_90 = threshold_dims[0.90]
        print(f"\n✅ 90% 분산 달성: {dim_90}개 주성분")
        print(f"   → M1 16GB에서 {dim_90}개는 충분히 처리 가능!")
    
    print("\n" + "="*70)
    
    return threshold_dims, cumulative_variance


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    stats_path = project_root / "data" / "visualization" / "feature_extraction" / "combined" / "train_visualization_stats.json"
    output_path = project_root / "data" / "visualization" / "feature_extraction" / "combined" / "pca_scree_plot.png"
    
    plot_scree_from_stats(stats_path, output_path)
