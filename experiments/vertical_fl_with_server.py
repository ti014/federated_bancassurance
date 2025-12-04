"""Vertical Federated Learning training với SplitNN architecture và Server Coordinator.

Module này sử dụng VerticalFLServer để điều phối training giữa Bank và Insurance.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import logging

# Package imports - rõ ràng và nhất quán
from models.splitnn import BottomModel, TopModel, create_splitnn_models
from federated.vertical_client import BankBottomClient, InsuranceTopClient
from federated.vertical_server import VerticalFLServer
from utils.metrics import evaluate_model
from utils.logger import get_logger

# Try to import SMOTE for class imbalance handling
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

# Setup logger
logger = get_logger("vertical_fl_experiment")


class VerticalFLDataset:
    """Dataset wrapper cho Vertical FL."""
    
    def __init__(
        self,
        bank_features: np.ndarray,
        insurance_features: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ):
        """Initialize dataset.
        
        Args:
            bank_features: Bank features (n_samples, n_bank_features)
            insurance_features: Insurance features (n_samples, n_insurance_features)
            labels: Labels (n_samples,) - optional
        """
        self.bank_features = torch.FloatTensor(bank_features)
        self.insurance_features = torch.FloatTensor(insurance_features)
        self.labels = torch.FloatTensor(labels).unsqueeze(1) if labels is not None else None
    
    def __len__(self):
        return len(self.bank_features)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.bank_features[idx], self.insurance_features[idx], self.labels[idx]
        else:
            return self.bank_features[idx], self.insurance_features[idx]


def run_vertical_fl_training_with_server(
    X_bank_train: np.ndarray,
    X_insurance_train: np.ndarray,
    y_train: np.ndarray,
    X_bank_test: np.ndarray,
    X_insurance_test: np.ndarray,
    y_test: np.ndarray,
    model_config: Dict,
    device: Optional[torch.device] = None,
    verbose: bool = True,
    use_smote: bool = True,
) -> Dict:
    """Run Vertical FL training với Server Coordinator.
    
    Args:
        X_bank_train: Bank training features
        X_insurance_train: Insurance training features
        y_train: Training labels
        X_bank_test: Bank test features
        X_insurance_test: Insurance test features
        y_test: Test labels
        model_config: Model configuration
        device: PyTorch device
        verbose: Print progress
        use_smote: Use SMOTE for class imbalance
    
    Returns:
        Dictionary với training history và final metrics
    """
    device = device or torch.device('cpu')
    
    # Apply SMOTE để xử lý class imbalance
    if use_smote and SMOTE_AVAILABLE:
        logger.info("Applying SMOTE to handle class imbalance...")
        logger.info(f"  Before: Positive={y_train.sum()}/{len(y_train)} ({y_train.mean():.2%})")
        
        # Combine features for SMOTE
        X_combined_train = np.concatenate([X_bank_train, X_insurance_train], axis=1)
        
        # Apply SMOTE
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_combined_resampled, y_resampled = smote.fit_resample(X_combined_train, y_train)
        
        # Split back to bank and insurance features
        bank_feature_count = X_bank_train.shape[1]
        X_bank_train = X_combined_resampled[:, :bank_feature_count]
        X_insurance_train = X_combined_resampled[:, bank_feature_count:]
        y_train = y_resampled
        
        logger.info(f"  After: Positive={y_train.sum()}/{len(y_train)} ({y_train.mean():.2%})")
        logger.info(f"  Samples increased from {len(X_combined_train)} to {len(X_combined_resampled)}")
    elif use_smote and not SMOTE_AVAILABLE:
        logger.warning("SMOTE requested but not available. Using weighted loss instead.")
    
    # Extract config
    bank_input_size = X_bank_train.shape[1]
    insurance_input_size = X_insurance_train.shape[1]
    embedding_size = model_config.get('embedding_size', 16)
    bottom_hidden_sizes = model_config.get('bottom_hidden_sizes', [64, 32])
    top_hidden_sizes = model_config.get('top_hidden_sizes', [32, 16])
    batch_size = model_config.get('batch_size', 32)
    learning_rate = model_config.get('learning_rate', 0.001)
    epochs = model_config.get('epochs', 10)
    
    # Create models
    bottom_model, top_model = create_splitnn_models(
        bank_input_size=bank_input_size,
        insurance_input_size=insurance_input_size,
        bottom_hidden_sizes=bottom_hidden_sizes,
        top_hidden_sizes=top_hidden_sizes,
        embedding_size=embedding_size,
        dropout_rate=model_config.get('dropout_rate', 0.3),
        activation=model_config.get('activation', 'relu'),
    )
    
    # Create datasets
    train_dataset = VerticalFLDataset(X_bank_train, X_insurance_train, y_train)
    test_dataset = VerticalFLDataset(X_bank_test, X_insurance_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create clients
    bank_train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_bank_train)),
        batch_size=batch_size,
        shuffle=True,
    )
    bank_test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_bank_test)),
        batch_size=batch_size,
        shuffle=False,
    )
    
    bank_client = BankBottomClient(
        bottom_model=bottom_model,
        train_loader=bank_train_loader,
        test_loader=bank_test_loader,
        device=device,
        learning_rate=learning_rate,
    )
    
    insurance_train_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(X_insurance_train),
            torch.FloatTensor(y_train).unsqueeze(1),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    insurance_test_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(X_insurance_test),
            torch.FloatTensor(y_test).unsqueeze(1),
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    
    insurance_client = InsuranceTopClient(
        top_model=top_model,
        train_loader=insurance_train_loader,
        test_loader=insurance_test_loader,
        device=device,
        learning_rate=learning_rate,
    )
    
    # Add weight decay to optimizers
    weight_decay = model_config.get('weight_decay', 0)
    bank_client.optimizer = torch.optim.Adam(
        bank_client.model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    insurance_client.optimizer = torch.optim.Adam(
        insurance_client.model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    
    # Create Server để điều phối training
    logger.info("Creating Vertical FL Server to coordinate training...")
    server = VerticalFLServer(
        bank_client=bank_client,
        insurance_client=insurance_client,
        device=device,
    )
    
    # Learning rate scheduler
    scheduler_bank = torch.optim.lr_scheduler.ReduceLROnPlateau(
        bank_client.optimizer, mode='min', factor=0.5, patience=10
    )
    scheduler_insurance = torch.optim.lr_scheduler.ReduceLROnPlateau(
        insurance_client.optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Training loop với Server
    if verbose:
        pbar = tqdm(range(epochs), desc="Vertical FL Training (with Server)")
    else:
        pbar = range(epochs)
    
    best_test_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 20
    
    for epoch in pbar:
        # Training phase - Server điều phối
        train_metrics = server.train_round(train_loader, y_train, verbose=False)
        
        # Evaluation phase - Server điều phối
        test_metrics = server.evaluate(test_loader)
        
        # Learning rate scheduling
        scheduler_bank.step(test_metrics['loss'])
        scheduler_insurance.step(test_metrics['loss'])
        
        # Store history
        server.history['train_losses'].append(train_metrics['loss'])
        server.history['train_accuracies'].append(train_metrics['accuracy'])
        server.history['test_losses'].append(test_metrics['loss'])
        server.history['test_accuracies'].append(test_metrics['accuracy'])
        server.history['epochs'].append(epoch + 1)
        
        # Early stopping
        if test_metrics['loss'] < best_test_loss:
            best_test_loss = test_metrics['loss']
            patience_counter = 0
        else:
            patience_counter += 1
        
        if verbose:
            current_lr = bank_client.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'train_loss': f'{train_metrics["loss"]:.4f}',
                'train_acc': f'{train_metrics["accuracy"]:.4f}',
                'test_loss': f'{test_metrics["loss"]:.4f}',
                'test_acc': f'{test_metrics["accuracy"]:.4f}',
                'lr': f'{current_lr:.6f}',
            })
        
        # Early stopping check
        if patience_counter >= early_stop_patience:
            if verbose:
                logger.info(f"Early stopping at epoch {epoch + 1}")
            break
    
    # Final evaluation với full metrics
    logger.info("Running final evaluation...")
    bank_client.model.eval()
    insurance_client.model.eval()
    
    final_metrics = evaluate_model(
        bottom_model=bottom_model,
        top_model=top_model,
        X_bank_test=X_bank_test,
        X_insurance_test=X_insurance_test,
        y_test=y_test,
        device=device,
    )
    
    logger.info(f"Final metrics: {final_metrics}")
    
    return {
        'bottom_model': bottom_model,
        'top_model': top_model,
        'final_metrics': final_metrics,
        'history': server.history,
        'bank_client': bank_client,
        'insurance_client': insurance_client,
        'server': server,
    }

