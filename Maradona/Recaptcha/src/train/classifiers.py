"""
Classifier factory and configuration.
"""
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from typing import Dict, Any, Optional
import numpy as np


def create_classifier(classifier_type: str, config: Dict[str, Any], 
                     class_weight: Optional[str] = "balanced") -> object:
    """
    Create classifier instance.
    
    Args:
        classifier_type: Type of classifier ('svm', 'random_forest', etc.)
        config: Configuration dictionary
        class_weight: Class weight strategy ('balanced', None, or custom)
        
    Returns:
        Classifier instance
    """
    classifier_config = config.get('training', {}).get(classifier_type, {})
    
    if classifier_type == 'svm':
        kernel = classifier_config.get('kernel', 'rbf')
        C = classifier_config.get('C', [1.0])
        gamma = classifier_config.get('gamma', ['scale'])
        
        # Use first value if list, otherwise use value
        C_val = C[0] if isinstance(C, list) else C
        gamma_val = gamma[0] if isinstance(gamma, list) else gamma
        
        return SVC(
            kernel=kernel,
            C=C_val,
            gamma=gamma_val,
            class_weight=class_weight,
            probability=True,
            random_state=42
        )
    
    elif classifier_type == 'random_forest':
        n_estimators = classifier_config.get('n_estimators', [100])
        max_depth = classifier_config.get('max_depth', [None])
        min_samples_split = classifier_config.get('min_samples_split', [2])
        
        n_estimators_val = n_estimators[0] if isinstance(n_estimators, list) else n_estimators
        max_depth_val = max_depth[0] if isinstance(max_depth, list) else max_depth
        min_samples_split_val = min_samples_split[0] if isinstance(min_samples_split, list) else min_samples_split
        
        # Handle string "None" (common in some YAML parsers)
        if max_depth_val == "None":
            max_depth_val = None
            
        return RandomForestClassifier(
            n_estimators=n_estimators_val,
            max_depth=max_depth_val,
            min_samples_split=min_samples_split_val,
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1
        )
    
    elif classifier_type == 'knn':
        n_neighbors = classifier_config.get('n_neighbors', [5])
        weights = classifier_config.get('weights', ['uniform'])
        
        n_neighbors_val = n_neighbors[0] if isinstance(n_neighbors, list) else n_neighbors
        weights_val = weights[0] if isinstance(weights, list) else weights
        
        return KNeighborsClassifier(
            n_neighbors=n_neighbors_val,
            weights=weights_val,
            n_jobs=1
        )
    
    elif classifier_type == 'logistic_regression':
        C = classifier_config.get('C', [1.0])
        max_iter = classifier_config.get('max_iter', 1000)
        solver = classifier_config.get('solver', 'lbfgs')
        
        C_val = C[0] if isinstance(C, list) else C
        
        return LogisticRegression(
            C=C_val,
            max_iter=max_iter,
            solver=solver,
            class_weight=class_weight,
            random_state=42,
            n_jobs=1
        )
    
    elif classifier_type == 'adaboost':
        n_estimators = classifier_config.get('n_estimators', [50])
        learning_rate = classifier_config.get('learning_rate', [1.0])
        
        n_estimators_val = n_estimators[0] if isinstance(n_estimators, list) else n_estimators
        learning_rate_val = learning_rate[0] if isinstance(learning_rate, list) else learning_rate
        
        return AdaBoostClassifier(
            n_estimators=n_estimators_val,
            learning_rate=learning_rate_val,
            random_state=42
        )
    
    elif classifier_type == 'decision_tree':
        return DecisionTreeClassifier(
            class_weight=class_weight,
            random_state=42
        )
    
    elif classifier_type == 'naive_bayes':
        return GaussianNB()
    
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")


def get_hyperparameter_grid(classifier_type: str, config: Dict[str, Any]) -> Dict[str, list]:
    """
    Get hyperparameter grid for grid search.
    
    Args:
        classifier_type: Type of classifier
        config: Configuration dictionary
        
    Returns:
        Dictionary of hyperparameter grids
    """
    classifier_config = config.get('training', {}).get(classifier_type, {})
    
    if classifier_type == 'svm':
        return {
            'C': classifier_config.get('C', [0.1, 1.0, 10.0]),
            'gamma': classifier_config.get('gamma', ['scale', 'auto', 0.001, 0.01])
        }
    
    elif classifier_type == 'random_forest':
        return {
            'n_estimators': classifier_config.get('n_estimators', [50, 100, 200]),
            'max_depth': classifier_config.get('max_depth', [None, 10, 20, 30]),
            'min_samples_split': classifier_config.get('min_samples_split', [2, 5, 10])
        }
    
    elif classifier_type == 'knn':
        return {
            'n_neighbors': classifier_config.get('n_neighbors', [3, 5, 7, 9]),
            'weights': classifier_config.get('weights', ['uniform', 'distance'])
        }
    
    elif classifier_type == 'logistic_regression':
        return {
            'C': classifier_config.get('C', [0.1, 1.0, 10.0])
        }
    
    elif classifier_type == 'adaboost':
        return {
            'n_estimators': classifier_config.get('n_estimators', [50, 100, 200]),
            'learning_rate': classifier_config.get('learning_rate', [0.5, 1.0, 1.5])
        }
    
    else:
        return {}
