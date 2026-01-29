"""
Feature Selection Methods: PCA, SelectKBest, RFE.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.svm import SVC
from typing import Dict, Tuple, Optional


class PCASelector:
    """PCA-based feature selection."""
    
    def __init__(self, n_components: Optional[int] = None, variance_threshold: float = 0.90):
        """
        Initialize PCA selector.
        
        Args:
            n_components: Number of components (None = auto based on variance_threshold)
            variance_threshold: Cumulative variance threshold (0.90 = 90%)
        """
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.pca = None
        self.optimal_n_components = None
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'PCASelector':
        """
        Fit PCA to data.
        
        Args:
            X: Feature matrix (N, D)
            y: Labels (optional, not used for PCA)
        """
        # If n_components not specified, find optimal based on variance_threshold
        if self.n_components is None:
            # Fit with all components first
            temp_pca = PCA()
            temp_pca.fit(X)
            
            # Find optimal number of components
            cumulative_variance = np.cumsum(temp_pca.explained_variance_ratio_)
            self.optimal_n_components = np.argmax(cumulative_variance >= self.variance_threshold) + 1
            
            print(f"  → Optimal components for {self.variance_threshold*100:.0f}% variance: {self.optimal_n_components}")
        else:
            self.optimal_n_components = self.n_components
        
        # Fit PCA with optimal number of components
        self.pca = PCA(n_components=self.optimal_n_components)
        self.pca.fit(X)
        
        # Verify variance
        explained_variance = np.sum(self.pca.explained_variance_ratio_)
        print(f"  → Actual explained variance: {explained_variance:.4f} ({explained_variance*100:.2f}%)")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features using PCA."""
        if self.pca is None:
            raise ValueError("PCA not fitted. Call fit() first.")
        return self.pca.transform(X)
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def get_explained_variance_ratio(self) -> np.ndarray:
        """Get explained variance ratio for each component."""
        if self.pca is None:
            raise ValueError("PCA not fitted.")
        return self.pca.explained_variance_ratio_
    
    def get_cumulative_variance(self) -> np.ndarray:
        """Get cumulative explained variance."""
        return np.cumsum(self.get_explained_variance_ratio())


class SelectKBestSelector:
    """SelectKBest-based feature selection."""
    
    def __init__(self, k: int = 100, score_func=f_classif):
        """
        Initialize SelectKBest selector.
        
        Args:
            k: Number of top features to select
            score_func: Scoring function (default: f_classif for classification)
        """
        self.k = k
        self.selector = SelectKBest(score_func=score_func, k=k)
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SelectKBestSelector':
        """Fit selector to data."""
        self.selector.fit(X, y)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features using SelectKBest."""
        return self.selector.transform(X)
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def get_feature_scores(self) -> np.ndarray:
        """Get feature scores."""
        return self.selector.scores_
    
    def get_selected_features(self) -> np.ndarray:
        """Get indices of selected features."""
        return self.selector.get_support(indices=True)


class RFESelector:
    """RFE-based feature selection."""
    
    def __init__(self, n_features: int = 100, estimator=None):
        """
        Initialize RFE selector.
        
        Args:
            n_features: Number of features to select
            estimator: Base estimator (default: LinearSVC)
        """
        from sklearn.svm import LinearSVC
        
        if estimator is None:
            estimator = LinearSVC(random_state=42, max_iter=10000)
        
        self.n_features = n_features
        self.selector = RFE(estimator=estimator, n_features_to_select=n_features)
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RFESelector':
        """Fit selector to data."""
        self.selector.fit(X, y)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features using RFE."""
        return self.selector.transform(X)
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def get_feature_ranking(self) -> np.ndarray:
        """Get feature ranking (1 = best)."""
        return self.selector.ranking_
    
    def get_selected_features(self) -> np.ndarray:
        """Get indices of selected features."""
        return self.selector.get_support(indices=True)


def create_selector(method: str, **kwargs) -> object:
    """
    Factory function to create feature selector.
    
    Args:
        method: Selection method ('pca', 'selectkbest', 'rfe')
        **kwargs: Method-specific parameters
        
    Returns:
        Feature selector instance
    """
    if method == 'pca':
        return PCASelector(**kwargs)
    elif method == 'selectkbest':
        return SelectKBestSelector(**kwargs)
    elif method == 'rfe':
        return RFESelector(**kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")
