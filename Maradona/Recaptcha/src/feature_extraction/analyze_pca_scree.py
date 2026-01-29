"""
PCA Scree Plot 분석 스크립트.
누적 분산 그래프를 그려서 실제 필요한 주성분 개수를 확인.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import json

# Add parent directory to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.preprocess.utils import load_config


def analyze_pca_scree(features_path: Path, output_dir: Path, max_components: int = 500):
    """
    PCA Scree Plot 분석.
    
    Args:
        features_path: 특징 파일 경로
        output_dir: 출력 디렉토리
        max_components: 분석할 최대 주성분 개수
    """
    print("="*70)
    print("PCA Scree Plot 분석")
    print("="*70)
    
    # 특징 로드
    print(f"\n[1] 특징 로드 중...")
    features = np.load(str(features_path))
    print(f"  샘플 수: {len(features):,}")
    print(f"  특징 차원: {features.shape[1]:,}")
    
    # StandardScaler 적용
    print(f"\n[2] StandardScaler 적용 중...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # PCA 분석 (최대 max_components개까지)
    print(f"\n[3] PCA 분석 중 (최대 {max_components}개 주성분)...")
    pca = PCA(n_components=min(max_components, features.shape[1]))
    pca.fit(features_scaled)
    
    # 누적 분산 계산
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    # 주요 지점 찾기
    thresholds = [0.80, 0.85, 0.90, 0.95, 0.99]
    threshold_dims = {}
    
    print(f"\n[4] 주요 지점 분석...")
    for threshold in thresholds:
        idx = np.argmax(cumulative_variance >= threshold)
        if cumulative_variance[idx] >= threshold:
            threshold_dims[threshold] = idx + 1
            print(f"  {threshold*100:.0f}% 분산: {idx + 1}개 주성분 (누적: {cumulative_variance[idx]:.4f})")
        else:
            print(f"  {threshold*100:.0f}% 분산: {max_components}개 이상 필요")
    
    # 상위 N개 주성분의 누적 분산
    print(f"\n[5] 상위 주성분 누적 분산:")
    for n in [50, 100, 150, 200, 250, 300, 400, 500]:
        if n <= len(cumulative_variance):
            print(f"  상위 {n}개: {cumulative_variance[n-1]:.4f} ({cumulative_variance[n-1]*100:.2f}%)")
    
    # 그래프 생성
    print(f"\n[6] Scree Plot 생성 중...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Figure 1: 개별 설명 분산 (Scree Plot)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('PCA Scree Plot Analysis', fontsize=16, fontweight='bold')
    
    # 1. 개별 설명 분산 (상위 50개)
    ax1 = axes[0, 0]
    n_plot = min(50, len(explained_variance_ratio))
    ax1.plot(range(1, n_plot + 1), explained_variance_ratio[:n_plot], 'bo-', linewidth=2, markersize=4)
    ax1.set_xlabel('Principal Component', fontsize=12)
    ax1.set_ylabel('Explained Variance Ratio', fontsize=12)
    ax1.set_title(f'Individual Explained Variance (Top {n_plot})', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. 누적 설명 분산 (전체)
    ax2 = axes[0, 1]
    n_plot = min(max_components, len(cumulative_variance))
    ax2.plot(range(1, n_plot + 1), cumulative_variance[:n_plot], 'go-', linewidth=2, markersize=3)
    # 주요 지점 표시
    for threshold, dim in threshold_dims.items():
        if dim <= n_plot:
            ax2.axvline(dim, color='r', linestyle='--', alpha=0.5, linewidth=1)
            ax2.axhline(threshold, color='r', linestyle='--', alpha=0.5, linewidth=1)
            ax2.plot(dim, threshold, 'ro', markersize=8)
            ax2.text(dim, threshold + 0.02, f'{threshold*100:.0f}%\n({dim}개)', 
                    ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    ax2.set_xlabel('Number of Components', fontsize=12)
    ax2.set_ylabel('Cumulative Explained Variance', fontsize=12)
    ax2.set_title('Cumulative Explained Variance', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.0])
    
    # 3. 누적 설명 분산 (상위 500개, 확대)
    ax3 = axes[1, 0]
    n_plot = min(500, len(cumulative_variance))
    ax3.plot(range(1, n_plot + 1), cumulative_variance[:n_plot], 'go-', linewidth=2, markersize=2)
    # 주요 지점 표시
    for threshold, dim in threshold_dims.items():
        if dim <= n_plot:
            ax3.axvline(dim, color='r', linestyle='--', alpha=0.5, linewidth=1)
            ax3.axhline(threshold, color='r', linestyle='--', alpha=0.5, linewidth=1)
            ax3.plot(dim, threshold, 'ro', markersize=6)
    ax3.set_xlabel('Number of Components', fontsize=12)
    ax3.set_ylabel('Cumulative Explained Variance', fontsize=12)
    ax3.set_title('Cumulative Explained Variance (Top 500)', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1.0])
    
    # 4. 통계 요약
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    stats_text = f"""
    PCA 분석 결과 요약
    {'='*50}
    
    원본 차원: {features.shape[1]:,}
    
    주요 지점:
    """
    for threshold in sorted(thresholds):
        if threshold in threshold_dims:
            dim = threshold_dims[threshold]
            stats_text += f"  {threshold*100:.0f}% 분산: {dim:,}개 주성분\n"
    
    stats_text += f"""
    
    상위 주성분 누적 분산:
    """
    for n in [50, 100, 150, 200, 250, 300]:
        if n <= len(cumulative_variance):
            stats_text += f"  {n:3d}개: {cumulative_variance[n-1]:.4f} ({cumulative_variance[n-1]*100:.2f}%)\n"
    
    stats_text += f"""
    
    권장 사항:
    """
    if 100 <= len(cumulative_variance) and cumulative_variance[99] >= 0.80:
        stats_text += f"  ✓ 상위 100개로 80% 이상 달성 가능\n"
    if 200 <= len(cumulative_variance) and cumulative_variance[199] >= 0.90:
        stats_text += f"  ✓ 상위 200개로 90% 이상 달성 가능\n"
    
    if 80 in threshold_dims:
        stats_text += f"  → 80% 달성: {threshold_dims[80]}개 주성분 사용 권장\n"
    if 90 in threshold_dims:
        stats_text += f"  → 90% 달성: {threshold_dims[90]}개 주성분 사용 권장\n"
    
    ax4.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / "pca_scree_plot.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 저장: {output_dir / 'pca_scree_plot.png'}")
    
    # JSON 저장
    results = {
        'original_dim': int(features.shape[1]),
        'max_components_analyzed': int(max_components),
        'thresholds': {f'{int(k*100)}%': int(v) for k, v in threshold_dims.items()},
        'top_n_cumulative': {
            n: float(cumulative_variance[n-1]) 
            for n in [50, 100, 150, 200, 250, 300, 400, 500] 
            if n <= len(cumulative_variance)
        },
        'explained_variance_ratio': explained_variance_ratio[:max_components].tolist(),
        'cumulative_variance': cumulative_variance[:max_components].tolist()
    }
    
    with open(output_dir / "pca_scree_analysis.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"  ✓ 저장: {output_dir / 'pca_scree_analysis.json'}")
    
    # 최종 권장 사항
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
    
    if 90 in threshold_dims:
        dim_90 = threshold_dims[90]
        print(f"\n✅ 90% 분산 달성: {dim_90}개 주성분")
        print(f"   → M1 16GB에서 {dim_90}개는 충분히 처리 가능!")
    
    print("\n" + "="*70)
    
    return results


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    config = load_config(project_root / "config" / "config.yaml")
    
    # v2 특징 경로
    features_path = project_root / "data" / "features" / "combined_v2" / "combined" / "train" / "train_features.npy"
    output_dir = project_root / "data" / "visualization" / "feature_extraction" / "combined"
    
    if not features_path.exists():
        print(f"❌ 특징 파일을 찾을 수 없습니다: {features_path}")
        sys.exit(1)
    
    analyze_pca_scree(features_path, output_dir, max_components=500)
