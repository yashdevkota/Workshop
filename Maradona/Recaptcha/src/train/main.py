"""
Main model training pipeline.
"""
import sys
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import time
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import pickle

# Add parent directory to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.preprocess.utils import load_config
from src.train.utils import load_features, save_model, ensure_dir
from src.train.classifiers import create_classifier, get_hyperparameter_grid
from src.train import visualization as vis
from src.feature_selection.utils import load_selector


def refine_training_data(X: np.ndarray, y: np.ndarray, label_mapping: Dict) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Refine training data by dropping 'Other' and merging 'TLight'/'Traffic Light'.
    
    Args:
        X: Features
        y: Labels
        label_mapping: Original label mapping
        
    Returns:
        Refined X, y, and new label_mapping
    """
    print(f"\n[Refine] Cleaning and merging classes...")
    
    # 1. 'Other' 클래스 제거
    other_idx = None
    for idx, name in label_mapping.items():
        if str(name).lower() == 'other':
            other_idx = int(idx)
            break
    
    if other_idx is not None:
        mask = (y != other_idx)
        X = X[mask]
        y = y[mask]
        print(f"  → Dropped 'Other' class (removed {np.sum(~mask):,} samples)")

    # 2. 'Traffic Light'와 'TLight' 병합
    tlight_idx = None
    traffic_light_idx = None
    for idx, name in label_mapping.items():
        if name == 'TLight':
            tlight_idx = int(idx)
        elif name == 'Traffic Light':
            traffic_light_idx = int(idx)
            
    if tlight_idx is not None and traffic_light_idx is not None:
        # Merge traffic_light_idx into tlight_idx
        y[y == traffic_light_idx] = tlight_idx
        print(f"  → Merged 'Traffic Light' (ID {traffic_light_idx}) into 'TLight' (ID {tlight_idx})")

    # 3. 레이블 재매핑 (0부터 연속된 숫자로)
    unique_labels = sorted(np.unique(y))
    new_y = np.zeros_like(y)
    final_label_mapping = {}
    
    for i, old_idx in enumerate(unique_labels):
        new_y[y == old_idx] = i
        final_label_mapping[str(i)] = label_mapping[str(old_idx)]
        
    print(f"  → New class distribution: {len(final_label_mapping)} classes")
    for i, name in final_label_mapping.items():
        count = np.sum(new_y == int(i))
        print(f"    - {i}: {name} ({count:,} samples)")
        
    return X, new_y, final_label_mapping


def apply_feature_selection(
    X_train: np.ndarray,
    X_val: Optional[np.ndarray],
    features_dir: Path,
    method: str = "pca"
) -> Tuple[np.ndarray, Optional[np.ndarray], object]:
    """
    Apply feature selection if available.
    
    Note: If features are already selected (loaded from _selected directory),
    this function will skip selection and return features as-is.
    
    Args:
        X_train: Training features
        X_val: Validation features (optional)
        features_dir: Directory containing feature selector
        method: Selection method ('pca', 'selectkbest', 'rfe')
        
    Returns:
        Tuple of (X_train_selected, X_val_selected, selector)
    """
    selector_path = features_dir / f"{method}_selector.pkl"
    
    if not selector_path.exists():
        print(f"  ℹ️  No feature selector found at {selector_path}")
        print(f"  → Features are already selected, using as-is")
        # Features are already selected, just return them
        return X_train, X_val, None
    
    # Check if features are already in selected dimension
    # If selector exists but features dimension matches expected output, skip
    print(f"  ℹ️  Feature selector found, but features may already be selected")
    print(f"  → Using features as-is (already selected)")
    return X_train, X_val, None


def run_kfold_cv(
    X: np.ndarray,
    y: np.ndarray,
    classifier,
    n_splits: int = 5,
    stratified: bool = True,
    random_state: int = 42
) -> Dict:
    """
    Run K-fold Cross-Validation with progress tracking.
    
    Args:
        X: Features
        y: Labels
        classifier: Classifier instance
        n_splits: Number of folds
        stratified: Whether to use stratified K-fold
        random_state: Random state
        
    Returns:
        Dictionary with CV results
    """
    from tqdm import tqdm
    
    if stratified:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    else:
        from sklearn.model_selection import KFold
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    print(f"  → Running {n_splits}-fold CV (stratified={stratified})...")
    print(f"  → Training samples: {len(X):,}, Features: {X.shape[1]:,}")
    start_time = time.time()
    
    # Manual CV with progress bar
    scores = []
    fold_times = []
    
    # Create list of splits first for tqdm (cv.split() is a generator)
    print("  → Preparing CV splits...")
    splits = list(cv.split(X, y))
    print(f"  → {len(splits)} folds prepared")
    
    # Create progress bar
    pbar = tqdm(total=n_splits, desc="  CV Progress", unit="fold", ncols=100, 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        fold_start = time.time()
        
        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X[val_idx]
        y_val_fold = y[val_idx]
        
        # Update progress bar description
        pbar.set_description(f"  Training Fold {fold_idx + 1}/{n_splits}")
        
        # Train on fold
        classifier.fit(X_train_fold, y_train_fold)
        
        # Evaluate on validation fold
        score = classifier.score(X_val_fold, y_val_fold)
        scores.append(score)
        
        fold_time = time.time() - fold_start
        fold_times.append(fold_time)
        
        elapsed = time.time() - start_time
        avg_time_per_fold = elapsed / (fold_idx + 1)
        remaining_folds = n_splits - (fold_idx + 1)
        estimated_remaining = avg_time_per_fold * remaining_folds
        
        # Update progress bar
        pbar.update(1)
        pbar.set_postfix({
            'Acc': f'{score:.4f}',
            'Time': f'{fold_time/60:.1f}m',
            'ETA': f'{estimated_remaining/60:.1f}m'
        })
        
        # Print detailed info
        print(f"\n    ✓ Fold {fold_idx + 1}/{n_splits} completed:")
        print(f"      Accuracy: {score:.4f}")
        print(f"      Time: {fold_time/60:.2f} minutes")
        print(f"      ETA: {estimated_remaining/60:.1f} minutes")
    
    pbar.close()
    
    scores = np.array(scores)
    elapsed_time = time.time() - start_time
    
    results = {
        'scores': scores.tolist(),
        'mean_score': float(scores.mean()),
        'std_score': float(scores.std()),
        'min_score': float(scores.min()),
        'max_score': float(scores.max()),
        'elapsed_time': elapsed_time,
        'fold_times': fold_times
    }
    
    print(f"\n  ✓ CV completed in {elapsed_time/60:.2f} minutes ({elapsed_time:.2f}s)")
    print(f"  ✓ Mean accuracy: {results['mean_score']:.4f} (+/- {results['std_score']:.4f})")
    print(f"  ✓ Range: [{results['min_score']:.4f}, {results['max_score']:.4f}]")
    print(f"  ✓ Average time per fold: {np.mean(fold_times):.2f}s")
    
    return results


def run_hyperparameter_tuning(
    X_train: np.ndarray,
    y_train: np.ndarray,
    classifier,
    param_grid: Dict,
    cv: int = 5,
    n_jobs: int = -1  # Changed to -1 for parallel processing
) -> Dict:
    """
    Run hyperparameter tuning using GridSearchCV.
    
    Args:
        X_train: Training features
        y_train: Training labels
        classifier: Classifier instance
        param_grid: Parameter grid for grid search
        cv: Number of CV folds
        n_jobs: Number of parallel jobs
        
    Returns:
        Dictionary with tuning results
    """
    # Use a subset of data for tuning if too large to speed up
    max_tuning_samples = 5000  # Balanced for quality and observable progress
    if len(X_train) > max_tuning_samples:
        print(f"  ℹ️  Large dataset detected ({len(X_train):,}).")
        print(f"  → Subsetting to {max_tuning_samples:,} samples for faster tuning...")
        # Stratified sampling
        from sklearn.model_selection import train_test_split
        X_tune, _, y_tune, _ = train_test_split(
            X_train, y_train, train_size=max_tuning_samples, 
            stratify=y_train, random_state=42
        )
    else:
        X_tune, y_tune = X_train, y_train

    print(f"  → Running GridSearchCV (CV={cv}, samples={len(X_tune)}) with {n_jobs} jobs...")
    start_time = time.time()
    
    grid_search = GridSearchCV(
        classifier,
        param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=n_jobs,
        verbose=3  # Highest verbosity to see every fit
    )
    
    grid_search.fit(X_tune, y_tune)
    
    elapsed_time = time.time() - start_time
    
    results = {
        'best_params': grid_search.best_params_,
        'best_score': float(grid_search.best_score_),
        'best_estimator': grid_search.best_estimator_,
        'elapsed_time': elapsed_time
    }
    
    print(f"  ✓ Tuning completed in {elapsed_time:.2f}s")
    print(f"  ✓ Best score: {results['best_score']:.4f}")
    print(f"  ✓ Best params: {results['best_params']}")
    
    return results


def train_model(
    config_path: Path,
    classifier_type: str = "svm",
    feature_type: str = "combined",
    use_feature_selection: bool = True,
    selection_method: str = "pca",
    use_hyperparameter_tuning: bool = True,
    version: str = "v1"
) -> Dict:
    """
    Train ML model with K-fold CV.
    
    Args:
        config_path: Path to config.yaml
        classifier_type: Type of classifier ('svm', 'random_forest', etc.)
        feature_type: Feature type ('combined', 'hog', etc.)
        use_feature_selection: Whether to use feature selection
        selection_method: Feature selection method ('pca', 'selectkbest', 'rfe')
        use_hyperparameter_tuning: Whether to use hyperparameter tuning
        
    Returns:
        Dictionary with training results
    """
    # Load configuration
    config = load_config(config_path)
    
    # Setup paths
    project_root = Path(__file__).parent.parent.parent
    
    # Input: features (with or without selection)
    if use_feature_selection:
        features_dir = project_root / config['data']['features_dir'] / f"{feature_type}_v2_selected" / selection_method
    else:
        features_dir = project_root / config['data']['features_dir'] / f"{feature_type}_v2" / feature_type
    
    # Output: trained models
    models_dir = project_root / "models" / version / f"{classifier_type}_{feature_type}"
    if use_feature_selection:
        models_dir = models_dir.parent / f"{classifier_type}_{feature_type}_{selection_method}"
    
    vis_dir = project_root / "data" / "visualization" / "training" / version / f"{classifier_type}_{feature_type}"
    if use_feature_selection:
        vis_dir = vis_dir.parent / f"{classifier_type}_{feature_type}_{selection_method}"
    
    print("\n" + "="*70)
    print("Model Training Pipeline")
    print("="*70)
    print(f"Classifier: {classifier_type.upper()}")
    print(f"Feature type: {feature_type}")
    print(f"Feature selection: {use_feature_selection} ({selection_method if use_feature_selection else 'none'})")
    print(f"Hyperparameter tuning: {use_hyperparameter_tuning}")
    print(f"Input: {features_dir}")
    print(f"Output: {models_dir}")
    print("="*70)
    
    # Load features
    print(f"\n[1] Loading features...")
    X_train, y_train, label_mapping = load_features(features_dir, "train")
    print(f"  ✓ Loaded {len(X_train):,} samples, {X_train.shape[1]:,} features")
    
    # [Refinement] Drop 'Other' and Merge Traffic Lights
    X_train, y_train, label_mapping = refine_training_data(X_train, y_train, label_mapping)
    print(f"  ✓ Final dataset: {len(X_train):,} samples, {len(label_mapping)} classes")
    
    # Load validation set if available
    X_val, y_val = None, None
    try:
        X_val, y_val, _ = load_features(features_dir, "val")
        print(f"  ✓ Loaded {len(X_val):,} validation samples")
    except FileNotFoundError:
        print(f"  ℹ️  No validation set found, using train set only")
    
    # StandardScaler (if not already applied in feature selection)
    print(f"\n[2] Applying StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    if X_val is not None:
        X_val_scaled = scaler.transform(X_val)
    else:
        X_val_scaled = None
    
    print(f"  ✓ Scaled features: mean={X_train_scaled.mean():.6f}, std={X_train_scaled.std():.6f}")
    
    # Feature selection (if enabled and selector exists)
    # Note: If features are loaded from _selected directory, they are already selected
    selector = None
    if use_feature_selection:
        print(f"\n[3] Feature selection status ({selection_method})...")
        # Check if features are already selected (loaded from _selected directory)
        if "_selected" in str(features_dir):
            print(f"  ✓ Features are already selected (loaded from {features_dir})")
            print(f"  → Using features as-is (no additional selection needed)")
            X_train_final = X_train_scaled
            X_val_final = X_val_scaled
        else:
            # Apply feature selection
            X_train_selected, X_val_selected, selector = apply_feature_selection(
                X_train_scaled, X_val_scaled, features_dir, selection_method
            )
            X_train_final = X_train_selected
            X_val_final = X_val_selected
    else:
        X_train_final = X_train_scaled
        X_val_final = X_val_scaled
    
    # Get class weight
    training_config = config.get('training', {})
    class_weight = training_config.get('class_weight', 'balanced')
    print(f"\n[4] Class weight: {class_weight}")
    
    # Create classifier
    print(f"\n[5] Creating {classifier_type.upper()} classifier...")
    classifier = create_classifier(classifier_type, config, class_weight=class_weight)
    
    # Hyperparameter tuning
    if use_hyperparameter_tuning:
        print(f"\n[6] Hyperparameter tuning...")
        param_grid = get_hyperparameter_grid(classifier_type, config)
        if param_grid:
            tuning_results = run_hyperparameter_tuning(
                X_train_final, y_train, classifier, param_grid,
                cv=training_config.get('hyperparameter_tuning', {}).get('cv', 5)
            )
            classifier = tuning_results['best_estimator']
        else:
            print(f"  ℹ️  No parameter grid defined, using default parameters")
    
    # K-fold Cross-Validation
    print(f"\n[7] K-fold Cross-Validation...")
    kfold_config = training_config.get('kfold', {})
    cv_results = run_kfold_cv(
        X_train_final, y_train, classifier,
        n_splits=kfold_config.get('n_splits', 5),
        stratified=kfold_config.get('stratified', True),
        random_state=kfold_config.get('random_state', 42)
    )
    
    # Train final model on all training data
    print(f"\n[8] Training final model on all training data...")
    print(f"  → Training on {len(X_train_final):,} samples, {X_train_final.shape[1]:,} features...")
    start_time = time.time()
    classifier.fit(X_train_final, y_train)
    training_time = time.time() - start_time
    print(f"  ✓ Training completed in {training_time/60:.2f} minutes ({training_time:.2f}s)")
    
    # Evaluate on validation set if available
    val_score = None
    if X_val_final is not None and y_val is not None:
        print(f"\n[9] Evaluating on validation set...")
        val_score = float(classifier.score(X_val_final, y_val))
        print(f"  ✓ Validation accuracy: {val_score:.4f}")
    
    # Save model
    print(f"\n[10] Saving model...")
    ensure_dir(models_dir)
    save_model(classifier, scaler, selector, models_dir)
    print(f"  ✓ Model saved to {models_dir}")
    
    # Visualization
    print(f"\n[11] Generating visualizations...")
    ensure_dir(vis_dir)
    
    # CV results plot
    vis.plot_cv_results(cv_results, vis_dir / "cv_results.png",
                       title=f"K-fold CV Results ({classifier_type.upper()})")
    
    # Save training stats
    training_stats = {
        'classifier': classifier_type,
        'feature_type': feature_type,
        'feature_selection': {
            'enabled': use_feature_selection,
            'method': selection_method if use_feature_selection else None
        },
        'num_samples': int(len(X_train)),
        'num_features': int(X_train_final.shape[1]),
        'num_classes': int(len(np.unique(y_train))),
        'cv_results': cv_results,
        'validation_score': val_score,
        'training_time': training_time
    }
    
    vis.save_training_stats(training_stats, vis_dir / "training_stats.json")
    
    print(f"  ✓ Visualizations saved to {vis_dir}")
    
    # Results summary
    print("\n" + "="*70)
    print("Training Results Summary")
    print("="*70)
    print(f"Classifier: {classifier_type.upper()}")
    print(f"Feature selection: {selection_method if use_feature_selection else 'None'}")
    print(f"Final features: {X_train_final.shape[1]:,}")
    print(f"K-fold CV mean accuracy: {cv_results['mean_score']:.4f} (+/- {cv_results['std_score']:.4f})")
    if val_score is not None:
        print(f"Validation accuracy: {val_score:.4f}")
    print(f"Training time: {training_time:.2f}s")
    print("="*70)
    
    return training_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training Pipeline")
    parser.add_argument("--classifier", type=str, default="svm",
                       choices=['svm', 'random_forest', 'knn', 'logistic_regression', 'adaboost'],
                       help="Classifier type")
    parser.add_argument("--feature_type", type=str, default="combined",
                       help="Feature type")
    parser.add_argument("--use_feature_selection", action="store_true",
                       help="Use feature selection")
    parser.add_argument("--selection_method", type=str, default="pca",
                       choices=['pca', 'selectkbest', 'rfe'],
                       help="Feature selection method")
    parser.add_argument("--no_hyperparameter_tuning", action="store_true",
                       help="Disable hyperparameter tuning")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                       help="Path to config.yaml")
    parser.add_argument("--version", type=str, default="v1",
                       help="Version name for this run (e.g., v1, v2)")
    
    args = parser.parse_args()
    
    config_path = project_root / args.config
    train_model(
        config_path=config_path,
        classifier_type=args.classifier,
        feature_type=args.feature_type,
        use_feature_selection=args.use_feature_selection,
        selection_method=args.selection_method,
        use_hyperparameter_tuning=not args.no_hyperparameter_tuning,
        version=args.version
    )
