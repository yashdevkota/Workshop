"""
v1과 v2 특징 추출 결과 비교 스크립트.
증거 자료 생성용.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import json

# Add parent directory to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.preprocess.utils import load_config

def load_features(features_dir: Path, split: str = "train"):
    """Load features from directory."""
    features_file = features_dir / f"{split}_features.npy"
    labels_file = features_dir / f"{split}_labels.npy"
    
    if not features_file.exists() or not labels_file.exists():
        return None, None, None
    
    features = np.load(str(features_file))
    labels = np.load(str(labels_file))
    
    # Load label mapping
    mapping_file = features_dir / f"{split}_label_mapping.json"
    label_mapping = {}
    if mapping_file.exists():
        with open(mapping_file, 'r') as f:
            label_mapping = json.load(f)
    
    return features, labels, label_mapping


def compare_features(v1_dir: Path, v2_dir: Path, output_dir: Path):
    """Compare v1 and v2 features and generate visualizations."""
    print("="*60)
    print("v1 vs v2 특징 비교 분석")
    print("="*60)
    
    # Load features
    print("\n[1] 특징 로딩 중...")
    v1_features, v1_labels, v1_mapping = load_features(v1_dir, "train")
    v2_features, v2_labels, v2_mapping = load_features(v2_dir, "train")
    
    if v1_features is None:
        print("⚠️  v1 특징을 찾을 수 없습니다.")
        return
    
    if v2_features is None:
        print("⚠️  v2 특징을 찾을 수 없습니다.")
        return
    
    print(f"  v1: {v1_features.shape} (samples, features)")
    print(f"  v2: {v2_features.shape} (samples, features)")
    
    # Ensure same number of samples
    min_samples = min(len(v1_features), len(v2_features))
    v1_features = v1_features[:min_samples]
    v2_features = v2_features[:min_samples]
    v1_labels = v1_labels[:min_samples]
    v2_labels = v2_labels[:min_samples]
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 기본 통계 비교
    print("\n[2] 기본 통계 비교...")
    stats_comparison = {
        'v1': {
            'mean': float(v1_features.mean()),
            'std': float(v1_features.std()),
            'min': float(v1_features.min()),
            'max': float(v1_features.max()),
            'shape': list(v1_features.shape)
        },
        'v2': {
            'mean': float(v2_features.mean()),
            'std': float(v2_features.std()),
            'min': float(v2_features.min()),
            'max': float(v2_features.max()),
            'shape': list(v2_features.shape)
        }
    }
    
    with open(output_dir / "stats_comparison.json", 'w') as f:
        json.dump(stats_comparison, f, indent=2)
    
    print(f"  v1 - Mean: {stats_comparison['v1']['mean']:.4f}, Std: {stats_comparison['v1']['std']:.4f}, Range: [{stats_comparison['v1']['min']:.4f}, {stats_comparison['v1']['max']:.4f}]")
    print(f"  v2 - Mean: {stats_comparison['v2']['mean']:.4f}, Std: {stats_comparison['v2']['std']:.4f}, Range: [{stats_comparison['v2']['min']:.4f}, {stats_comparison['v2']['max']:.4f}]")
    
    # 2. 분포 비교 히스토그램
    print("\n[3] 분포 비교 히스토그램 생성...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Feature value distribution
    axes[0, 0].hist(v1_features.flatten(), bins=100, alpha=0.7, label='v1', density=True)
    axes[0, 0].hist(v2_features.flatten(), bins=100, alpha=0.7, label='v2', density=True)
    axes[0, 0].set_xlabel('Feature Value')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Feature Value Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Mean per sample
    v1_sample_means = v1_features.mean(axis=1)
    v2_sample_means = v2_features.mean(axis=1)
    axes[0, 1].hist(v1_sample_means, bins=50, alpha=0.7, label='v1', density=True)
    axes[0, 1].hist(v2_sample_means, bins=50, alpha=0.7, label='v2', density=True)
    axes[0, 1].set_xlabel('Mean Feature Value per Sample')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Sample Mean Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Std per sample
    v1_sample_stds = v1_features.std(axis=1)
    v2_sample_stds = v2_features.std(axis=1)
    axes[1, 0].hist(v1_sample_stds, bins=50, alpha=0.7, label='v1', density=True)
    axes[1, 0].hist(v2_sample_stds, bins=50, alpha=0.7, label='v2', density=True)
    axes[1, 0].set_xlabel('Std Feature Value per Sample')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Sample Std Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Feature dimension statistics
    v1_feat_means = v1_features.mean(axis=0)
    v2_feat_means = v2_features.mean(axis=0)
    axes[1, 1].scatter(v1_feat_means, v2_feat_means, alpha=0.3, s=1)
    axes[1, 1].plot([v1_feat_means.min(), v1_feat_means.max()], 
                     [v1_feat_means.min(), v1_feat_means.max()], 'r--', alpha=0.5)
    axes[1, 1].set_xlabel('v1 Feature Mean')
    axes[1, 1].set_ylabel('v2 Feature Mean')
    axes[1, 1].set_title('Feature Mean Comparison')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "distribution_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 저장: {output_dir / 'distribution_comparison.png'}")
    
    # 3. PCA 비교
    print("\n[4] PCA 분석 비교...")
    for version, features, labels in [('v1', v1_features, v1_labels), ('v2', v2_features, v2_labels)]:
        # Standardize
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # PCA
        pca = PCA(n_components=50)
        pca.fit(features_scaled)
        
        # Explained variance
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        optimal_dim = np.argmax(cumsum >= 0.95) + 1
        
        pca_stats = {
            'pc1_variance': float(pca.explained_variance_ratio_[0]),
            'pc2_variance': float(pca.explained_variance_ratio_[1]),
            'cumulative_95_dim': int(optimal_dim),
            'cumulative_95_variance': float(cumsum[optimal_dim - 1]),
            'top10_variance': float(pca.explained_variance_ratio_[:10].sum())
        }
        
        with open(output_dir / f"pca_stats_{version}.json", 'w') as f:
            json.dump(pca_stats, f, indent=2)
        
        print(f"  {version} - PC1: {pca_stats['pc1_variance']:.4f}, 95% dim: {pca_stats['cumulative_95_dim']}, Top10: {pca_stats['top10_variance']:.4f}")
    
    # PCA visualization
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    for idx, (version, features, labels) in enumerate([('v1', v1_features, v1_labels), ('v2', v2_features, v2_labels)]):
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        pca_2d = PCA(n_components=2)
        features_2d = pca_2d.fit_transform(features_scaled)
        
        pca_3d = PCA(n_components=3)
        features_3d = pca_3d.fit_transform(features_scaled)
        
        # 2D scatter plot (PC1 vs PC2)
        row = idx * 2
        ax1 = fig.add_subplot(gs[row, 0])
        scatter = ax1.scatter(features_2d[:, 0], features_2d[:, 1], 
                              c=labels, cmap='tab20', alpha=0.5, s=1)
        ax1.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%})')
        ax1.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%})')
        ax1.set_title(f'{version.upper()} - 2D PCA (PC1 vs PC2)')
        ax1.grid(True, alpha=0.3)
        
        # 3D scatter plot
        ax2 = fig.add_subplot(gs[row, 1], projection='3d')
        scatter = ax2.scatter(features_3d[:, 0], features_3d[:, 1], features_3d[:, 2],
                             c=labels, cmap='tab20', alpha=0.5, s=1)
        ax2.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]:.2%})')
        ax2.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]:.2%})')
        ax2.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]:.2%})')
        ax2.set_title(f'{version.upper()} - 3D PCA Projection')
        
        # 2D scatter plot (PC1 vs PC3)
        ax3 = fig.add_subplot(gs[row, 2])
        scatter = ax3.scatter(features_3d[:, 0], features_3d[:, 2], 
                              c=labels, cmap='tab20', alpha=0.5, s=1)
        ax3.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]:.2%})')
        ax3.set_ylabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]:.2%})')
        ax3.set_title(f'{version.upper()} - 2D PCA (PC1 vs PC3)')
        ax3.grid(True, alpha=0.3)
    
    # Explained variance comparison
    pca_v1 = PCA(n_components=50)
    pca_v2 = PCA(n_components=50)
    pca_v1.fit(StandardScaler().fit_transform(v1_features))
    pca_v2.fit(StandardScaler().fit_transform(v2_features))
    
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(range(1, 51), np.cumsum(pca_v1.explained_variance_ratio_), 'o-', label='v1', markersize=3)
    ax4.plot(range(1, 51), np.cumsum(pca_v2.explained_variance_ratio_), 's-', label='v2', markersize=3)
    ax4.axhline(y=0.95, color='r', linestyle='--', alpha=0.5, label='95% threshold')
    ax4.set_xlabel('Number of Components')
    ax4.set_ylabel('Cumulative Explained Variance')
    ax4.set_title('Cumulative Explained Variance')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Top components comparison
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.bar(range(1, 11), pca_v1.explained_variance_ratio_[:10], alpha=0.7, label='v1', width=0.4)
    ax5.bar([x + 0.4 for x in range(1, 11)], pca_v2.explained_variance_ratio_[:10], alpha=0.7, label='v2', width=0.4)
    ax5.set_xlabel('Component')
    ax5.set_ylabel('Explained Variance Ratio')
    ax5.set_title('Top 10 Components Comparison')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # PC1, PC2, PC3 comparison
    ax6 = fig.add_subplot(gs[2, 2])
    components = ['PC1', 'PC2', 'PC3']
    v1_pc = [pca_v1.explained_variance_ratio_[0], pca_v1.explained_variance_ratio_[1], pca_v1.explained_variance_ratio_[2]]
    v2_pc = [pca_v2.explained_variance_ratio_[0], pca_v2.explained_variance_ratio_[1], pca_v2.explained_variance_ratio_[2]]
    x = np.arange(len(components))
    width = 0.35
    ax6.bar(x - width/2, v1_pc, width, label='v1', alpha=0.7)
    ax6.bar(x + width/2, v2_pc, width, label='v2', alpha=0.7)
    ax6.set_ylabel('Explained Variance Ratio')
    ax6.set_title('Top 3 Components Comparison')
    ax6.set_xticks(x)
    ax6.set_xticklabels(components)
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / "pca_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 저장: {output_dir / 'pca_comparison.png'}")
    
    # 4. 개선 요약
    print("\n[5] 개선 요약 생성...")
    v1_pca = PCA(n_components=50).fit(StandardScaler().fit_transform(v1_features))
    v2_pca = PCA(n_components=50).fit(StandardScaler().fit_transform(v2_features))
    
    improvement = {
        'pc1_variance_improvement': float(v2_pca.explained_variance_ratio_[0] - v1_pca.explained_variance_ratio_[0]),
        'pc1_variance_improvement_pct': float((v2_pca.explained_variance_ratio_[0] / v1_pca.explained_variance_ratio_[0] - 1) * 100),
        'range_reduction': float((stats_comparison['v1']['max'] - stats_comparison['v1']['min']) - (stats_comparison['v2']['max'] - stats_comparison['v2']['min'])),
        'std_improvement': float(stats_comparison['v2']['std'] - stats_comparison['v1']['std']),
    }
    
    with open(output_dir / "improvement_summary.json", 'w') as f:
        json.dump(improvement, f, indent=2)
    
    print("\n" + "="*60)
    print("개선 요약")
    print("="*60)
    print(f"PC1 설명 분산: {v1_pca.explained_variance_ratio_[0]:.4f} → {v2_pca.explained_variance_ratio_[0]:.4f} ({improvement['pc1_variance_improvement_pct']:+.2f}%)")
    print(f"특징 범위: [{stats_comparison['v1']['min']:.2f}, {stats_comparison['v1']['max']:.2f}] → [{stats_comparison['v2']['min']:.2f}, {stats_comparison['v2']['max']:.2f}]")
    print(f"표준편차: {stats_comparison['v1']['std']:.4f} → {stats_comparison['v2']['std']:.4f}")
    print("="*60)
    
    print(f"\n✓ 모든 비교 결과 저장: {output_dir}")


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    config = load_config(project_root / "config" / "config.yaml")
    
    features_base_dir = project_root / config['data']['features_dir']
    v1_dir = features_base_dir / "combined_v1_backup"
    v2_dir = features_base_dir / "combined_v2"
    output_dir = project_root / "data" / "visualization" / "feature_extraction" / "v1_v2_comparison"
    
    compare_features(v1_dir, v2_dir, output_dir)
