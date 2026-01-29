"""
Visualization for model training results.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
import json

sns.set_style("whitegrid")


def plot_cv_results(cv_results: Dict, output_path: Path, title: str = "K-fold CV Results") -> None:
    """
    Plot K-fold Cross-Validation results.
    
    Args:
        cv_results: Dictionary with CV results
        output_path: Output file path
        title: Plot title
    """
    scores = cv_results.get('scores', [])
    mean_score = cv_results.get('mean_score', 0)
    std_score = cv_results.get('std_score', 0)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Box plot
    ax1 = axes[0]
    ax1.boxplot(scores, vert=True, patch_artist=True,
                boxprops=dict(facecolor='steelblue', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
    ax1.axhline(mean_score, color='green', linestyle='--', linewidth=2, label=f'Mean: {mean_score:.4f}')
    ax1.set_ylabel('Accuracy', fontsize=11)
    ax1.set_xlabel('Fold', fontsize=11)
    ax1.set_title('Cross-Validation Scores', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Bar plot with error bars
    ax2 = axes[1]
    x_pos = np.arange(len(scores))
    ax2.bar(x_pos, scores, color='steelblue', alpha=0.7, edgecolor='black',
            yerr=[std_score] * len(scores), capsize=5)
    ax2.axhline(mean_score, color='green', linestyle='--', linewidth=2, label=f'Mean: {mean_score:.4f}')
    ax2.set_ylabel('Accuracy', fontsize=11)
    ax2.set_xlabel('Fold', fontsize=11)
    ax2.set_title('Fold-wise Scores', fontsize=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'Fold {i+1}' for i in range(len(scores))])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_hyperparameter_tuning(grid_search_results: Dict, output_path: Path, 
                                title: str = "Hyperparameter Tuning") -> None:
    """
    Plot hyperparameter tuning results.
    
    Args:
        grid_search_results: Dictionary with grid search results
        output_path: Output file path
        title: Plot title
    """
    param_scores = grid_search_results.get('param_scores', {})
    
    if not param_scores:
        return
    
    # Create heatmap if 2D parameter grid
    if len(param_scores) > 0:
        # Try to extract 2D grid
        params = list(param_scores.keys())
        if len(params) >= 2:
            # Use first two parameters for heatmap
            param1_name = params[0]
            param2_name = params[1]
            
            # Extract unique values
            param1_values = sorted(set([p[0] for p in param_scores.keys()]))
            param2_values = sorted(set([p[1] for p in param_scores.keys()]))
            
            # Create score matrix
            score_matrix = np.zeros((len(param1_values), len(param2_values)))
            for (p1, p2), score in param_scores.items():
                i = param1_values.index(p1)
                j = param2_values.index(p2)
                score_matrix[i, j] = score
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(score_matrix, annot=True, fmt='.4f', cmap='YlOrRd',
                       xticklabels=[str(v) for v in param2_values],
                       yticklabels=[str(v) for v in param1_values],
                       ax=ax)
            ax.set_xlabel(param2_name, fontsize=12)
            ax.set_ylabel(param1_name, fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()


def plot_model_comparison(model_results: List[Dict], output_path: Path,
                          title: str = "Model Comparison") -> None:
    """
    Plot comparison of multiple models.
    
    Args:
        model_results: List of dictionaries with model results
        output_path: Output file path
        title: Plot title
    """
    model_names = [r['model_name'] for r in model_results]
    mean_scores = [r.get('mean_score', 0) for r in model_results]
    std_scores = [r.get('std_score', 0) for r in model_results]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = np.arange(len(model_names))
    
    bars = ax.bar(x_pos, mean_scores, yerr=std_scores, capsize=5,
                  color='steelblue', alpha=0.7, edgecolor='black')
    
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(mean_scores, std_scores)):
        ax.text(i, mean + std + 0.01, f'{mean:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_training_stats(results: Dict, output_path: Path) -> None:
    """
    Save training statistics to JSON.
    
    Args:
        results: Dictionary with training results
        output_path: Output file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
