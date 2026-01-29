"""
Feature extraction visualization module.
Creates comprehensive visualizations for extracted features.
"""
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import json

try:
    from .utils import ensure_dir, load_cached_features
except ImportError:
    from src.feature_extraction.utils import ensure_dir, load_cached_features


def visualize_feature_heatmap(features: np.ndarray, 
                              output_path: Path,
                              num_samples: int = 100,
                              title: str = "Feature Heatmap") -> None:
    """
    Visualize feature vectors as heatmap.
    
    Args:
        features: Feature array (N, feature_dim)
        output_path: Path to save visualization
        num_samples: Number of samples to visualize (for large datasets)
        title: Plot title
    """
    # Sample if too many
    if len(features) > num_samples:
        indices = np.random.choice(len(features), num_samples, replace=False)
        sampled_features = features[indices]
    else:
        sampled_features = features
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(20, max(8, len(sampled_features) // 10)))
    
    # Normalize for better visualization
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(sampled_features)
    
    sns.heatmap(
        features_scaled,
        cmap='RdYlBu_r',
        center=0,
        vmin=-3,
        vmax=3,
        cbar_kws={'label': 'Normalized Feature Value'},
        ax=ax
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Feature Dimension', fontsize=12)
    ax.set_ylabel('Sample Index', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved feature heatmap to {output_path}")


def visualize_feature_distribution(features: np.ndarray,
                                  output_path: Path,
                                  feature_names: Optional[List[str]] = None,
                                  num_features: int = 20) -> None:
    """
    Visualize distribution of individual features.
    
    Args:
        features: Feature array (N, feature_dim)
        output_path: Path to save visualization
        feature_names: Optional list of feature names
        num_features: Number of features to visualize
    """
    # Select random features if too many
    if features.shape[1] > num_features:
        feature_indices = np.random.choice(features.shape[1], num_features, replace=False)
        selected_features = features[:, feature_indices]
    else:
        selected_features = features
        feature_indices = np.arange(features.shape[1])
    
    # Create subplots
    n_cols = 4
    n_rows = (len(feature_indices) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Feature Value Distributions', fontsize=16, fontweight='bold')
    
    for idx, (ax, feat_idx) in enumerate(zip(axes.ravel(), feature_indices)):
        if idx >= len(feature_indices):
            ax.axis('off')
            continue
        
        feature_values = selected_features[:, idx]
        
        ax.hist(feature_values, bins=50, edgecolor='black', alpha=0.7)
        ax.set_title(f'Feature {feat_idx}' + (f': {feature_names[feat_idx]}' if feature_names else ''),
                    fontsize=10)
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved feature distribution to {output_path}")


def visualize_feature_correlation(features: np.ndarray,
                                 output_path: Path,
                                 num_features: int = 50) -> None:
    """
    Visualize correlation matrix between features.
    
    Args:
        features: Feature array (N, feature_dim)
        output_path: Path to save visualization
        num_features: Number of features to include in correlation
    """
    # Sample features if too many
    if features.shape[1] > num_features:
        feature_indices = np.random.choice(features.shape[1], num_features, replace=False)
        selected_features = features[:, feature_indices]
    else:
        selected_features = features
        feature_indices = np.arange(features.shape[1])
    
    # Calculate correlation matrix
    correlation_matrix = np.corrcoef(selected_features.T)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 12))
    
    sns.heatmap(
        correlation_matrix,
        cmap='coolwarm',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={'label': 'Correlation Coefficient'},
        ax=ax
    )
    
    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    ax.set_xlabel('Feature Index', fontsize=12)
    ax.set_ylabel('Feature Index', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved feature correlation to {output_path}")


def visualize_pca_analysis(features: np.ndarray,
                          labels: np.ndarray,
                          output_path: Path,
                          n_components: int = 2,
                          explained_variance_threshold: float = 0.95) -> Dict:
    """
    Visualize PCA analysis of features.
    
    Args:
        features: Feature array (N, feature_dim)
        labels: Label array (N,)
        output_path: Path to save visualization
        n_components: Number of components for 2D visualization
        explained_variance_threshold: Threshold for variance explanation
        
    Returns:
        Dictionary with PCA statistics
    """
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Fit PCA
    pca = PCA()
    pca.fit(features_scaled)
    
    # Find optimal number of components
    cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
    optimal_components = np.argmax(cumsum_variance >= explained_variance_threshold) + 1
    
    # Transform to 2D for visualization
    pca_2d = PCA(n_components=n_components)
    features_2d = pca_2d.fit_transform(features_scaled)
    
    # Transform to 3D for visualization
    pca_3d = PCA(n_components=3)
    features_3d = pca_3d.fit_transform(features_scaled)
    
    # Create visualization
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('PCA Analysis of Features', fontsize=16, fontweight='bold')
    
    # Create grid for subplots
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    
    # 1. 2D scatter plot (PC1 vs PC2)
    ax1 = fig.add_subplot(gs[0, 0])
    for label, color in zip(unique_labels, colors):
        mask = labels == label
        ax1.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                   c=[color], label=f'Class {label}', alpha=0.6, s=20)
    ax1.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
    ax1.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
    ax1.set_title('2D PCA Projection (PC1 vs PC2)', fontsize=13, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. 3D scatter plot
    ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    for label, color in zip(unique_labels, colors):
        mask = labels == label
        ax2.scatter(features_3d[mask, 0], features_3d[mask, 1], features_3d[mask, 2],
                   c=[color], label=f'Class {label}', alpha=0.6, s=20)
    ax2.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]:.2%})', fontsize=10)
    ax2.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]:.2%})', fontsize=10)
    ax2.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]:.2%})', fontsize=10)
    ax2.set_title('3D PCA Projection', fontsize=13, fontweight='bold')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # 3. 2D scatter plot (PC1 vs PC3)
    ax3 = fig.add_subplot(gs[0, 2])
    for label, color in zip(unique_labels, colors):
        mask = labels == label
        ax3.scatter(features_3d[mask, 0], features_3d[mask, 2], 
                   c=[color], label=f'Class {label}', alpha=0.6, s=20)
    ax3.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
    ax3.set_ylabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]:.2%} variance)', fontsize=12)
    ax3.set_title('2D PCA Projection (PC1 vs PC3)', fontsize=13, fontweight='bold')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 4. Explained variance ratio
    ax = fig.add_subplot(gs[1, 0])
    n_plot = min(50, len(pca.explained_variance_ratio_))
    ax.plot(range(1, n_plot + 1), pca.explained_variance_ratio_[:n_plot], 
           'bo-', linewidth=2, markersize=4)
    ax.axvline(optimal_components, color='r', linestyle='--', 
              label=f'Optimal ({optimal_components} components)')
    ax.set_xlabel('Principal Component', fontsize=12)
    ax.set_ylabel('Explained Variance Ratio', fontsize=12)
    ax.set_title('Explained Variance by Component', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Cumulative explained variance
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(range(1, n_plot + 1), cumsum_variance[:n_plot], 
           'go-', linewidth=2, markersize=4)
    ax.axhline(explained_variance_threshold, color='r', linestyle='--', 
              label=f'{explained_variance_threshold:.0%} threshold')
    ax.axvline(optimal_components, color='r', linestyle='--')
    ax.set_xlabel('Number of Components', fontsize=12)
    ax.set_ylabel('Cumulative Explained Variance', fontsize=12)
    ax.set_title('Cumulative Explained Variance', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Feature statistics
    ax = fig.add_subplot(gs[1, 2])
    ax.axis('off')
    stats_text = f"""
    PCA Statistics:
    --------------------
    Original dimensions: {features.shape[1]}
    Optimal dimensions ({explained_variance_threshold:.0%} variance): {optimal_components}
    
    Explained Variance:
    - PC1: {pca_2d.explained_variance_ratio_[0]:.2%}
    - PC2: {pca_2d.explained_variance_ratio_[1]:.2%}
    - PC3: {pca_3d.explained_variance_ratio_[2]:.2%}
    - Top 3 components: {cumsum_variance[2]:.2%}
    - Top 10 components: {cumsum_variance[9]:.2%}
    - Top 50 components: {cumsum_variance[49]:.2%}
    
    Feature Statistics:
    - Mean: {features.mean():.4f}
    - Std: {features.std():.4f}
    - Min: {features.min():.4f}
    - Max: {features.max():.4f}
    """
    ax.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
           verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Return statistics
    stats = {
        'original_dim': features.shape[1],
        'optimal_dim': optimal_components,
        'explained_variance_ratio': pca.explained_variance_ratio_[:optimal_components].tolist(),
        'cumulative_variance': cumsum_variance[optimal_components - 1],
        'pc1_variance': float(pca_2d.explained_variance_ratio_[0]),
        'pc2_variance': float(pca_2d.explained_variance_ratio_[1]),
        'pc3_variance': float(pca_3d.explained_variance_ratio_[2])
    }
    
    print(f"✓ Saved PCA analysis to {output_path}")
    
    return stats


def visualize_class_feature_distribution(features: np.ndarray,
                                        labels: np.ndarray,
                                        label_mapping: Dict[int, str],
                                        output_path: Path,
                                        num_features: int = 10) -> None:
    """
    Visualize feature distribution by class.
    
    Args:
        features: Feature array (N, feature_dim)
        labels: Label array (N,)
        label_mapping: Mapping from label index to class name
        output_path: Path to save visualization
        num_features: Number of features to visualize
    """
    # Select random features
    if features.shape[1] > num_features:
        feature_indices = np.random.choice(features.shape[1], num_features, replace=False)
    else:
        feature_indices = np.arange(features.shape[1])
    
    unique_labels = np.unique(labels)
    n_cols = 2
    n_rows = (len(feature_indices) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 3))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Feature Distribution by Class', fontsize=16, fontweight='bold')
    
    for idx, (ax, feat_idx) in enumerate(zip(axes.ravel(), feature_indices)):
        if idx >= len(feature_indices):
            ax.axis('off')
            continue
        
        # Plot distribution for each class
        for label in unique_labels[:10]:  # Limit to 10 classes for readability
            mask = labels == label
            class_name = label_mapping.get(int(label), f'Class {label}')
            ax.hist(features[mask, feat_idx], bins=30, alpha=0.5, 
                   label=class_name if idx == 0 else "")
        
        ax.set_title(f'Feature {feat_idx}', fontsize=11)
        ax.set_xlabel('Feature Value')
        ax.set_ylabel('Frequency')
        if idx == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved class feature distribution to {output_path}")


def visualize_scree_plot(features: np.ndarray,
                        output_path: Path,
                        max_components: int = 500) -> Dict:
    """
    Create Scree Plot for PCA analysis.
    
    Args:
        features: Feature array (N, feature_dim)
        output_path: Path to save visualization
        max_components: Maximum number of components to analyze
        
    Returns:
        Dictionary with analysis results
    """
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # PCA analysis
    n_components = min(max_components, features_scaled.shape[1], features_scaled.shape[0] - 1)
    pca = PCA(n_components=n_components)
    pca.fit(features_scaled)
    
    # Calculate cumulative variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    # Find key thresholds
    thresholds = [0.80, 0.90, 0.95]
    threshold_results = {}
    for threshold in thresholds:
        idx = np.argmax(cumulative_variance >= threshold)
        threshold_results[threshold] = {
            'n_components': int(idx + 1),
            'variance': float(cumulative_variance[idx])
        }
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('PCA Scree Plot Analysis', fontsize=16, fontweight='bold')
    
    # 1. Individual explained variance (Scree Plot)
    ax1 = axes[0, 0]
    n_plot = min(100, len(explained_variance))
    ax1.plot(range(1, n_plot + 1), explained_variance[:n_plot], 'bo-', markersize=3, linewidth=1)
    ax1.set_xlabel('Principal Component', fontsize=12)
    ax1.set_ylabel('Explained Variance Ratio', fontsize=12)
    ax1.set_title('Scree Plot (Individual Variance)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.01, color='r', linestyle='--', alpha=0.5, label='1% threshold')
    ax1.legend()
    
    # 2. Cumulative explained variance
    ax2 = axes[0, 1]
    n_plot = min(500, len(cumulative_variance))
    ax2.plot(range(1, n_plot + 1), cumulative_variance[:n_plot], 'go-', markersize=2, linewidth=1.5)
    for threshold in thresholds:
        n_comp = threshold_results[threshold]['n_components']
        if n_comp <= n_plot:
            ax2.axvline(x=n_comp, color='r', linestyle='--', alpha=0.5)
            ax2.axhline(y=threshold, color='r', linestyle='--', alpha=0.3)
            ax2.text(n_comp, threshold + 0.02, f'{threshold*100:.0f}% ({n_comp})', 
                    fontsize=9, ha='center')
    ax2.set_xlabel('Number of Components', fontsize=12)
    ax2.set_ylabel('Cumulative Explained Variance', fontsize=12)
    ax2.set_title('Cumulative Explained Variance', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Zoomed view (0-200)
    ax3 = axes[1, 0]
    n_plot = min(200, len(cumulative_variance))
    ax3.plot(range(1, n_plot + 1), cumulative_variance[:n_plot], 'mo-', markersize=3, linewidth=1.5)
    for threshold in thresholds:
        n_comp = threshold_results[threshold]['n_components']
        if n_comp <= n_plot:
            ax3.axvline(x=n_comp, color='r', linestyle='--', alpha=0.7)
            ax3.text(n_comp, threshold, f'{n_comp}', fontsize=9, ha='center', va='bottom')
            ax3.axhline(y=threshold, color='r', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Number of Components (0-200)', fontsize=12)
    ax3.set_ylabel('Cumulative Explained Variance', fontsize=12)
    ax3.set_title('Cumulative Variance (Zoom: 0-200)', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Statistics summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Key points
    key_points = {
        50: cumulative_variance[49] if len(cumulative_variance) > 49 else 0,
        100: cumulative_variance[99] if len(cumulative_variance) > 99 else 0,
        200: cumulative_variance[199] if len(cumulative_variance) > 199 else 0,
        300: cumulative_variance[299] if len(cumulative_variance) > 299 else 0,
        500: cumulative_variance[499] if len(cumulative_variance) > 499 else 0,
    }
    
    stats_text = f"""
PCA 분석 요약
{'='*50}

주성분 개수별 누적 분산:
  • 50개:   {key_points[50]*100:.2f}%
  • 100개:  {key_points[100]*100:.2f}%
  • 200개:  {key_points[200]*100:.2f}%
  • 300개:  {key_points[300]*100:.2f}%
  • 500개:  {key_points[500]*100:.2f}%

목표 지점:
  • 80%:    {threshold_results[0.80]['n_components']}개 주성분
  • 90%:    {threshold_results[0.90]['n_components']}개 주성분
  • 95%:    {threshold_results[0.95]['n_components']}개 주성분

Top 10 주성분:
  • PC1:    {explained_variance[0]*100:.2f}%
  • PC2:    {explained_variance[1]*100:.2f}%
  • PC3:    {explained_variance[2]*100:.2f}%
  • PC4:    {explained_variance[3]*100:.2f}%
  • PC5:    {explained_variance[4]*100:.2f}%
  • PC6:    {explained_variance[5]*100:.2f}%
  • PC7:    {explained_variance[6]*100:.2f}%
  • PC8:    {explained_variance[7]*100:.2f}%
  • PC9:    {explained_variance[8]*100:.2f}%
  • PC10:   {explained_variance[9]*100:.2f}%

권장 사항:
  • 빠른 학습: 100-200개 주성분
  • 균형: 300-400개 주성분
  • 최대 정보: {threshold_results[0.95]['n_components']}개 주성분
"""
    ax4.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Return results
    results = {
        'thresholds': threshold_results,
        'key_points': {k: float(v) for k, v in key_points.items()},
        'top10_variance': [float(v) for v in explained_variance[:10]],
        'max_components_analyzed': n_components
    }
    
    print(f"✓ Saved Scree Plot to {output_path}")
    
    return results


def create_feature_extraction_report(features_dir: Path,
                                    output_dir: Path,
                                    split: str = 'train') -> None:
    """
    Create comprehensive feature extraction visualization report.
    
    Args:
        features_dir: Directory containing extracted features
        output_dir: Directory to save visualizations
        split: Split to visualize (train/val/test)
    """
    ensure_dir(output_dir)
    
    print(f"\n{'='*60}")
    print(f"Creating Feature Extraction Visualizations for {split} split")
    print(f"{'='*60}")
    
    # Load features and labels
    features, labels = load_cached_features(features_dir, split)
    
    # Load label mapping
    label_mapping_path = features_dir / f"{split}_label_mapping.json"
    if label_mapping_path.exists():
        with open(label_mapping_path, 'r') as f:
            label_mapping = {int(k): v for k, v in json.load(f).items()}
    else:
        unique_labels = np.unique(labels)
        label_mapping = {int(label): f'Class {label}' for label in unique_labels}
    
    print(f"Loaded {len(features)} samples with {features.shape[1]} features")
    
    # Create all visualizations
    print("\n1. Creating feature heatmap...")
    visualize_feature_heatmap(
        features,
        output_dir / f"{split}_feature_heatmap.png",
        num_samples=100,
        title=f"Feature Heatmap ({split})"
    )
    
    print("\n2. Creating feature distribution...")
    visualize_feature_distribution(
        features,
        output_dir / f"{split}_feature_distribution.png",
        num_features=20
    )
    
    print("\n3. Creating feature correlation...")
    visualize_feature_correlation(
        features,
        output_dir / f"{split}_feature_correlation.png",
        num_features=50
    )
    
    print("\n4. Creating PCA analysis...")
    pca_stats = visualize_pca_analysis(
        features,
        labels,
        output_dir / f"{split}_pca_analysis.png"
    )
    
    print("\n5. Creating Scree Plot analysis...")
    scree_results = visualize_scree_plot(
        features,
        output_dir / f"{split}_scree_plot.png",
        max_components=500
    )
    
    # Save scree plot results
    scree_results_path = output_dir / f"{split}_scree_plot_results.json"
    with open(scree_results_path, 'w') as f:
        json.dump(scree_results, f, indent=2)
    print(f"✓ Saved Scree Plot results to {scree_results_path}")
    
    print("\n5. Creating class feature distribution...")
    visualize_class_feature_distribution(
        features,
        labels,
        label_mapping,
        output_dir / f"{split}_class_feature_distribution.png",
        num_features=10
    )
    
    # Save statistics
    stats = {
        'split': split,
        'num_samples': len(features),
        'feature_dim': features.shape[1],
        'num_classes': len(np.unique(labels)),
        'pca_stats': pca_stats,
        'feature_stats': {
            'mean': float(features.mean()),
            'std': float(features.std()),
            'min': float(features.min()),
            'max': float(features.max())
        }
    }
    
    stats_path = output_dir / f"{split}_visualization_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    
    print(f"\n✓ Feature extraction visualizations complete!")
    print(f"  Visualizations saved to: {output_dir}")
    print(f"  Statistics saved to: {stats_path}")
