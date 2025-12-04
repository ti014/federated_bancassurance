"""Baseline models for Lapse Prediction comparison."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Dict, Any, Optional
import warnings

warnings.filterwarnings('ignore')


class LogisticRegressionBaseline:
    """Logistic Regression baseline model."""
    
    def __init__(
        self,
        random_state: int = 42,
        max_iter: int = 1000,
        C: float = 1.0,
    ):
        """Initialize Logistic Regression model.
        
        Args:
            random_state: Random seed
            max_iter: Maximum iterations
            C: Inverse regularization strength
        """
        self.model = LogisticRegression(
            random_state=random_state,
            max_iter=max_iter,
            C=C,
            solver='lbfgs',
        )
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train model.
        
        Args:
            X: Training features
            y: Training labels
        """
        self.model.fit(X, y)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels.
        
        Args:
            X: Features
        
        Returns:
            Binary predictions
        """
        if not self.is_fitted:
            raise ValueError("Model chưa được train. Gọi fit() trước.")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities.
        
        Args:
            X: Features
        
        Returns:
            Probability predictions (n_samples, 2)
        """
        if not self.is_fitted:
            raise ValueError("Model chưa được train. Gọi fit() trước.")
        return self.model.predict_proba(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model.
        
        Args:
            X: Test features
            y: Test labels
        
        Returns:
            Dictionary với metrics
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]
        
        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y, y_proba) if len(np.unique(y)) > 1 else 0.0,
        }


class RandomForestBaseline:
    """Random Forest baseline model."""
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        random_state: int = 42,
    ):
        """Initialize Random Forest model.
        
        Args:
            n_estimators: Số trees
            max_depth: Maximum depth
            random_state: Random seed
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train model."""
        self.model.fit(X, y)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels."""
        if not self.is_fitted:
            raise ValueError("Model chưa được train.")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_fitted:
            raise ValueError("Model chưa được train.")
        return self.model.predict_proba(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model."""
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]
        
        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y, y_proba) if len(np.unique(y)) > 1 else 0.0,
        }

