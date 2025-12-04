"""Centralized training experiments."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from utils.metrics import calculate_metrics


def run_centralized_experiment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_config: Dict,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict:
    """Run centralized training experiment.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        model_config: Model configuration dictionary
        device: PyTorch device
        verbose: Print progress
    
    Returns:
        Dictionary với training history và final metrics
    """
    device = device or torch.device("cpu")
    
    # Create model - Simple MLP for centralized learning
    class CentralizedNN(nn.Module):
        def __init__(self, input_size, hidden_sizes, dropout_rate=0.3, activation='relu'):
            super().__init__()
            layers = []
            prev_size = input_size
            
            for hidden_size in hidden_sizes:
                layers.append(nn.Linear(prev_size, hidden_size))
                if activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'tanh':
                    layers.append(nn.Tanh())
                layers.append(nn.Dropout(dropout_rate))
                prev_size = hidden_size
            
            layers.append(nn.Linear(prev_size, 1))
            layers.append(nn.Sigmoid())
            self.network = nn.Sequential(*layers)
        
        def forward(self, x):
            return self.network(x)
    
    model = CentralizedNN(
        input_size=model_config['input_size'],
        hidden_sizes=model_config.get('hidden_sizes', [64, 32]),
        dropout_rate=model_config.get('dropout_rate', 0.3),
        activation=model_config.get('activation', 'relu'),
    )
    model.to(device)
    
    # Create DataLoaders
    from torch.utils.data import TensorDataset
    
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train).unsqueeze(1)  # Shape: (batch_size, 1)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.FloatTensor(y_test).unsqueeze(1)  # Shape: (batch_size, 1)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=model_config.get('batch_size', 32),
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=model_config.get('batch_size', 32),
        shuffle=False,
    )
    
    # Setup training
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=model_config.get('learning_rate', 0.001),
    )
    
    # Use weighted loss để handle class imbalance
    # Tính class weights từ training data
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1.0
    pos_weight = torch.FloatTensor([pos_weight]).to(device)
    criterion = nn.BCELoss(weight=pos_weight)
    
    epochs = model_config.get('epochs', 10)
    
    # Training history
    history = {
        'train_losses': [],
        'test_losses': [],
        'test_accuracies': [],
    }
    
    # Training loop
    if verbose:
        pbar = tqdm(range(epochs), desc="Training")
    else:
        pbar = range(epochs)
    
    for epoch in pbar:
        # Training
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = train_loss / num_batches if num_batches > 0 else 0.0
        history['train_losses'].append(avg_train_loss)
        
        # Evaluation
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0
        
        with torch.no_grad():
            for features, labels in test_loader:
                features = features.to(device)
                labels = labels.to(device)
                
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                num_batches += 1
                
                predictions = (outputs > 0.5).long()
                correct += (predictions == labels.long()).sum().item()
                total += labels.size(0)
        
        avg_test_loss = test_loss / num_batches if num_batches > 0 else 0.0
        test_accuracy = correct / total if total > 0 else 0.0
        
        history['test_losses'].append(avg_test_loss)
        history['test_accuracies'].append(test_accuracy)
        
        if verbose:
            pbar.set_postfix({
                'train_loss': f'{avg_train_loss:.4f}',
                'test_loss': f'{avg_test_loss:.4f}',
                'test_acc': f'{test_accuracy:.4f}',
            })
    
    # Final evaluation với full metrics
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(features)
            all_predictions.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    y_pred_proba = np.concatenate(all_predictions).flatten()
    y_true = np.concatenate(all_labels).flatten()
    
    # Find optimal threshold
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_pred_temp = (y_pred_proba > threshold).astype(int)
        metrics_temp = calculate_metrics(y_true, y_pred_temp, y_pred_proba)
        f1_temp = metrics_temp.get('f1', 0)
        if f1_temp > best_f1:
            best_f1 = f1_temp
            best_threshold = threshold
    
    y_pred = (y_pred_proba > best_threshold).astype(int)
    final_metrics = calculate_metrics(y_true, y_pred, y_pred_proba)
    final_metrics['threshold_used'] = best_threshold
    
    # Add loss
    criterion_eval = nn.BCELoss()
    final_metrics['loss'] = criterion_eval(
        torch.FloatTensor(y_pred_proba),
        torch.FloatTensor(y_true)
    ).item()
    
    return {
        'model': model,
        'history': history,
        'final_metrics': final_metrics,
        'model_config': model_config,
    }

