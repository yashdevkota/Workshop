"""
Visualization for feature selection results.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict
import json


def plot_pca_scree(selector, output_path: Path, title: str = "PCA Scree Plot") -> None:
    """
    Plot PCA Scree Plot (explained variance).
    
    Args:
        selector: PCASelector instance
        output_path: Output file path
        title: Plot title
    """
    explained_variance = selector.get_explained_variance_ratio()
    cumulative_variance = selector.get_cumulative_variance()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Individual explained variance
    ax1 = axes[0]
    n_plot = min(50, len(explained_variance))
    ax1.plot(range(1, n_plot + 1), explained_variance[:n_plot], 'bo-', linewidth=2, markersize=4)
    ax1.set_xlabel('Principal Component', fontsize=11)
    ax1.set_ylabel('Explained Variance Ratio', fontsize=11)
    ax1.set_title(f'Individual Explained Variance (Top {n_plot})', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Cumulative explained variance
    ax2 = axes[1]
    ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'go-', linewidth=2, markersize=3)
    ax2.axhline(selector.variance_threshold, color='r', linestyle='--', alpha=0.7, label=f'{selector.variance_threshold*100:.0f}% threshold')
    ax2.set_xlabel('Number of Components', fontsize=11)
    ax2.set_ylabel('Cumulative Explained Variance', fontsize=11)
    ax2.set_title(f'Cumulative Explained Variance ({len(cumulative_variance)} components)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim([0, 1.0])
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_feature_importance(selector, output_path: Path, title: str = "Feature Importance") -> None:
    """
    Plot feature importance/scores.
    
    Args:
        selector: SelectKBest or RFE selector
        output_path: Output file path
        title: Plot title
    """
    if hasattr(selector, 'get_feature_scores'):
        scores = selector.get_feature_scores()
        selected = selector.get_selected_features()
    elif hasattr(selector, 'get_feature_ranking'):
        ranking = selector.get_feature_ranking()
        scores = 1.0 / ranking  # Convert ranking to scores (lower rank = higher score)
        selected = selector.get_selected_features()
    else:
        raise ValueError("Selector does not support feature importance visualization")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Top features
    top_k = min(50, len(selected))
    top_indices = selected[:top_k]
    top_scores = scores[top_indices]
    
    ax1 = axes[0]
    ax1.barh(range(len(top_scores)), top_scores, color='steelblue')
    ax1.set_xlabel('Feature Score', fontsize=11)
    ax1.set_ylabel('Feature Index', fontsize=11)
    ax1.set_title(f'Top {top_k} Selected Features', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Score distribution
    ax2 = axes[1]
    ax2.hist(scores, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.axvline(scores[selected].min(), color='r', linestyle='--', linewidth=2, label='Selection threshold')
    ax2.set_xlabel('Feature Score', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Feature Score Distribution', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_feature_selection_comparison(
    original_dim: int,
    selected_dims: Dict[str, int],
    explained_variances: Optional[Dict[str, float]] = None,
    output_path: Path = None
) -> None:
    """
    Plot comparison of different feature selection methods.
    
    Args:
        original_dim: Original feature dimension
        selected_dims: Dictionary of {method: selected_dim}
        explained_variances: Dictionary of {method: explained_variance} (for PCA)
        output_path: Output file path
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Feature Selection Comparison', fontsize=14, fontweight='bold')
    
    methods = list(selected_dims.keys())
    dims = [selected_dims[m] for m in methods]
    reduction_ratios = [(original_dim - d) / original_dim * 100 for d in dims]
    
    # Dimension comparison
    ax1 = axes[0]
    x_pos = np.arange(len(methods))
    ax1.bar(x_pos, dims, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axhline(original_dim, color='r', linestyle='--', linewidth=2, label='Original')
    ax1.set_xlabel('Method', fontsize=11)
    ax1.set_ylabel('Selected Dimensions', fontsize=11)
    ax1.set_title('Dimension Reduction', fontsize=12)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Reduction ratio
    ax2 = axes[1]
    ax2.bar(x_pos, reduction_ratios, color='coral', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Method', fontsize=11)
    ax2.set_ylabel('Reduction Ratio (%)', fontsize=11)
    ax2.set_title('Dimension Reduction Ratio', fontsize=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(methods, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add explained variance for PCA if available
    if explained_variances and 'pca' in explained_variances:
        ax1.text(0, dims[0], f'{explained_variances["pca"]*100:.1f}%', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_selection_stats(
    selector,
    method: str,
    original_dim: int,
    selected_dim: int,
    output_path: Path
) -> None:
    """
    Save feature selection statistics to JSON.
    
    Args:
        selector: Feature selector instance
        method: Selection method name
        original_dim: Original feature dimension
        selected_dim: Selected feature dimension
        output_path: Output file path
    """
    stats = {
        'method': method,
        'original_dim': int(original_dim),
        'selected_dim': int(selected_dim),
        'reduction_ratio': float((original_dim - selected_dim) / original_dim * 100)
    }
    
    if hasattr(selector, 'get_explained_variance_ratio'):
        explained_variance = float(np.sum(selector.get_explained_variance_ratio()))
        stats['explained_variance'] = explained_variance
        stats['n_components'] = int(selector.optimal_n_components)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
