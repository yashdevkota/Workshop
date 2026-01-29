"""
Main feature selection pipeline.
"""
import sys
import argparse
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import yaml

# Add parent directory to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def load_config(config_path: Path) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
from src.feature_selection.utils import load_features, save_selector, ensure_dir
from src.feature_selection.feature_selectors import create_selector
from src.feature_selection import visualization as vis
from sklearn.preprocessing import StandardScaler


def run_feature_selection(
    config_path: Path,
    method: str = "pca",
    feature_type: str = "combined",
    split: str = "train",
    output_suffix: str = ""
) -> Dict:
    """
    Run feature selection pipeline.
    
    Args:
        config_path: Path to config.yaml
        method: Selection method ('pca', 'selectkbest', 'rfe')
        feature_type: Feature type ('combined', 'hog', etc.)
        split: Split to use for fitting ('train')
        output_suffix: Suffix for output directory
        
    Returns:
        Dictionary with selection results
    """
    # Load configuration
    config = load_config(config_path)
    
    # Setup paths
    project_root = Path(__file__).parent.parent.parent
    
    # Input: features from feature extraction
    if feature_type == "combined":
        features_dir = project_root / config['data']['features_dir'] / "combined_v2" / "combined" / "train"
    else:
        features_dir = project_root / config['data']['features_dir'] / f"{feature_type}_v2" / feature_type / "train"
    
    # Output: selected features
    output_dir = project_root / config['data']['features_dir'] / f"{feature_type}_v2_selected{output_suffix}" / method
    vis_dir = project_root / config['data']['visualization_dir'] / "feature_selection" / f"{method}{output_suffix}"
    
    print("\n" + "="*70)
    print("Feature Selection Pipeline")
    print("="*70)
    print(f"Method: {method.upper()}")
    print(f"Feature type: {feature_type}")
    print(f"Input: {features_dir}")
    print(f"Output: {output_dir}")
    print("="*70)
    
    # Load features
    print(f"\n[1] Loading features for {split} split...")
    X_train, y_train, label_mapping = load_features(features_dir, split)
    print(f"  ✓ Loaded {len(X_train):,} samples, {X_train.shape[1]:,} features")
    print(f"  ✓ Classes: {len(np.unique(y_train))}")
    
    # StandardScaler (PCA requires scaling)
    print(f"\n[2] Applying StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    print(f"  ✓ Scaled features: mean={X_train_scaled.mean():.6f}, std={X_train_scaled.std():.6f}")
    
    # Create selector based on method
    print(f"\n[3] Creating {method.upper()} selector...")
    selector_kwargs = {}
    
    if method == 'pca':
        # Use 90% variance threshold (1,491 components from analysis)
        pca_config = config.get('training', {}).get('pca', {})
        variance_threshold = pca_config.get('variance_threshold', 0.90)
        auto_optimal = pca_config.get('auto_optimal_dim', True)
        
        if auto_optimal:
            selector_kwargs = {'variance_threshold': variance_threshold, 'n_components': None}
            print(f"  → Auto-determining components for {variance_threshold*100:.0f}% variance")
        else:
            # Use specific number from analysis
            selector_kwargs = {'n_components': 1491, 'variance_threshold': 0.90}
            print(f"  → Using 1,491 components (90% variance)")
    
    elif method == 'selectkbest':
        k = config.get('training', {}).get('feature_selection', {}).get('k', 200)
        selector_kwargs = {'k': k}
        print(f"  → Selecting top {k} features")
    
    elif method == 'rfe':
        n_features = config.get('training', {}).get('feature_selection', {}).get('n_features', 200)
        selector_kwargs = {'n_features': n_features}
        print(f"  → Selecting {n_features} features")
    
    selector = create_selector(method, **selector_kwargs)
    
    # Fit selector
    print(f"\n[4] Fitting {method.upper()} selector...")
    if method == 'pca':
        selector.fit(X_train_scaled)
    else:
        selector.fit(X_train_scaled, y_train)
    
    # Transform features
    print(f"\n[5] Transforming features...")
    X_train_selected = selector.transform(X_train_scaled)
    print(f"  ✓ Reduced from {X_train.shape[1]:,} to {X_train_selected.shape[1]:,} features")
    
    if method == 'pca':
        explained_variance = np.sum(selector.get_explained_variance_ratio())
        print(f"  ✓ Explained variance: {explained_variance:.4f} ({explained_variance*100:.2f}%)")
    
    # Save selector and scaler
    print(f"\n[6] Saving selector and scaler...")
    ensure_dir(output_dir)
    save_selector(selector, output_dir / f"{method}_selector.pkl")
    save_selector(scaler, output_dir / "scaler.pkl")
    
    # Save selected features
    print(f"\n[7] Saving selected features...")
    np.save(output_dir / f"{split}_features.npy", X_train_selected.astype(np.float32))
    np.save(output_dir / f"{split}_labels.npy", y_train)
    
    # Save label mapping
    import json
    with open(output_dir / f"{split}_label_mapping.json", 'w') as f:
        json.dump(label_mapping, f, indent=2)
    
    print(f"  ✓ Saved to {output_dir}")
    
    # Visualization
    print(f"\n[8] Generating visualizations...")
    ensure_dir(vis_dir)
    
    if method == 'pca':
        vis.plot_pca_scree(selector, vis_dir / "pca_scree_plot.png", 
                          title=f"PCA Scree Plot ({method.upper()})")
        vis.save_selection_stats(selector, method, X_train.shape[1], 
                                X_train_selected.shape[1], vis_dir / "selection_stats.json")
    
    elif method in ['selectkbest', 'rfe']:
        vis.plot_feature_importance(selector, vis_dir / "feature_importance.png",
                                   title=f"Feature Importance ({method.upper()})")
        vis.save_selection_stats(selector, method, X_train.shape[1],
                                X_train_selected.shape[1], vis_dir / "selection_stats.json")
    
    print(f"  ✓ Visualizations saved to {vis_dir}")
    
    # Results summary
    results = {
        'method': method,
        'original_dim': int(X_train.shape[1]),
        'selected_dim': int(X_train_selected.shape[1]),
        'reduction_ratio': float((X_train.shape[1] - X_train_selected.shape[1]) / X_train.shape[1] * 100),
        'num_samples': int(len(X_train)),
        'num_classes': int(len(np.unique(y_train)))
    }
    
    if method == 'pca':
        results['explained_variance'] = float(np.sum(selector.get_explained_variance_ratio()))
        results['n_components'] = int(selector.optimal_n_components)
    
    print("\n" + "="*70)
    print("Feature Selection Results")
    print("="*70)
    print(f"Method: {method.upper()}")
    print(f"Original dimension: {results['original_dim']:,}")
    print(f"Selected dimension: {results['selected_dim']:,}")
    print(f"Reduction ratio: {results['reduction_ratio']:.2f}%")
    if method == 'pca':
        print(f"Explained variance: {results['explained_variance']:.4f} ({results['explained_variance']*100:.2f}%)")
        print(f"Number of components: {results['n_components']:,}")
    print("="*70)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature Selection Pipeline")
    parser.add_argument("--method", type=str, default="pca", choices=['pca', 'selectkbest', 'rfe'],
                       help="Feature selection method")
    parser.add_argument("--feature_type", type=str, default="combined",
                       help="Feature type (combined, hog, etc.)")
    parser.add_argument("--split", type=str, default="train",
                       help="Split to use for fitting")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                       help="Path to config.yaml")
    parser.add_argument("--output_suffix", type=str, default="",
                       help="Suffix for output directory")
    
    args = parser.parse_args()
    
    config_path = project_root / args.config
    run_feature_selection(
        config_path=config_path,
        method=args.method,
        feature_type=args.feature_type,
        split=args.split,
        output_suffix=args.output_suffix
    )
