"""Evaluation metrics utilities."""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from typing import Dict, Any, Optional


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Calculate classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional, for ROC-AUC)
    
    Returns:
        Dictionary với các metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    
    # ROC-AUC nếu có probabilities và có cả 2 classes
    if y_proba is not None and len(np.unique(y_true)) > 1:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics['roc_auc'] = 0.0
    else:
        metrics['roc_auc'] = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        metrics['tn'] = int(cm[0, 0])  # True negatives
        metrics['fp'] = int(cm[0, 1])  # False positives
        metrics['fn'] = int(cm[1, 0])  # False negatives
        metrics['tp'] = int(cm[1, 1])  # True positives
    
    return metrics


def evaluate_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    device: Optional[torch.device] = None,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Evaluate PyTorch model.
    
    Args:
        model: PyTorch model
        X: Features
        y: Labels
        device: PyTorch device
        threshold: Classification threshold
    
    Returns:
        Dictionary với metrics
    """
    model.eval()
    
    # Convert to tensor
    if isinstance(X, np.ndarray):
        X_tensor = torch.FloatTensor(X)
    else:
        X_tensor = X
    
    if device is not None:
        model = model.to(device)
        X_tensor = X_tensor.to(device)
    
    import torch.nn as nn
    
    with torch.no_grad():
        # Get predictions
        outputs = model(X_tensor)
        
        if isinstance(outputs, torch.Tensor):
            y_proba = outputs.cpu().numpy().flatten()
            y_tensor = torch.FloatTensor(y).to(device) if device else torch.FloatTensor(y)
            if len(y_tensor.shape) == 1:
                y_tensor = y_tensor.unsqueeze(1)
            # Calculate loss
            criterion = nn.BCELoss()
            loss = criterion(outputs, y_tensor).item()
        else:
            y_proba = outputs.flatten()
            loss = calculate_loss(y, y_proba, loss_fn='bce')
        
        # Tìm optimal threshold dựa trên F1 score nếu có imbalance
        if len(np.unique(y)) > 1:
            from sklearn.metrics import f1_score
            thresholds = np.arange(0.1, 0.9, 0.05)
            best_threshold = threshold
            best_f1 = 0
            
            for thresh in thresholds:
                y_pred_temp = (y_proba > thresh).astype(int)
                f1_temp = f1_score(y, y_pred_temp, zero_division=0)
                if f1_temp > best_f1:
                    best_f1 = f1_temp
                    best_threshold = thresh
            
            # Nếu optimal threshold tốt hơn, dùng nó
            if best_f1 > 0:
                threshold = best_threshold
        
        y_pred = (y_proba > threshold).astype(int)
    
    metrics = calculate_metrics(y, y_pred, y_proba)
    metrics['loss'] = loss
    metrics['threshold_used'] = threshold
    
    return metrics


def calculate_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    loss_fn: str = 'bce',
) -> float:
    """Calculate loss.
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        loss_fn: Loss function ('bce' hoặc 'mse')
    
    Returns:
        Loss value
    """
    if loss_fn == 'bce':
        # Binary cross-entropy
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        )
    elif loss_fn == 'mse':
        # Mean squared error
        loss = np.mean((y_true - y_pred) ** 2)
    else:
        raise ValueError(f"Unknown loss function: {loss_fn}")
    
    return float(loss)

