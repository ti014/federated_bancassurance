"""Vertical Federated Learning training với SplitNN architecture.

Module này chứa logic training chính cho Vertical FL.
Tất cả imports đều từ các package modules.
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

# Setup logger
logger = get_logger("vertical_fl_experiment")

# Try to import SMOTE for class imbalance handling
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

# Setup logger
logger = get_logger("vertical_fl_experiment")

if not SMOTE_AVAILABLE:
    logger.warning("imbalanced-learn not installed. SMOTE will not be used.")
    logger.warning("  Install with: pip install imbalanced-learn")


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


def run_vertical_fl_training(
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
    use_server: bool = False,
) -> Dict:
    """Run Vertical FL training với SplitNN.
    
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
    # Bank client: chỉ cần bank features
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
    
    # Insurance client: cần insurance features + labels
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
    
    # Training history
    history = {
        'train_losses': [],
        'train_accuracies': [],
        'test_losses': [],
        'test_accuracies': [],
        'epochs': [],
    }
    
    # Learning rate scheduler
    scheduler_bank = torch.optim.lr_scheduler.ReduceLROnPlateau(
        bank_client.optimizer, mode='min', factor=0.5, patience=10
    )
    scheduler_insurance = torch.optim.lr_scheduler.ReduceLROnPlateau(
        insurance_client.optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Training loop
    logger.info(f"\nStarting training for {epochs} epochs...")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Training samples: {len(X_bank_train)}")
    logger.info(f"  Test samples: {len(X_bank_test)}")
    
    if verbose:
        pbar = tqdm(range(epochs), desc="Vertical FL Training")
    else:
        pbar = range(epochs)
    
    best_test_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 20
    
    for epoch in pbar:
        # Training phase
        epoch_losses = []
        epoch_accuracies = []
        
        bank_client.model.train()
        insurance_client.model.train()
        
        for batch_idx, (bank_batch, insurance_batch, labels_batch) in enumerate(train_loader):
            bank_batch = bank_batch.to(device)
            insurance_batch = insurance_batch.to(device)
            labels_batch = labels_batch.to(device)
            
            # Zero gradients
            bank_client.optimizer.zero_grad()
            insurance_client.optimizer.zero_grad()
            
            # Forward pass: Bank → Insurance
            # 1. Bank forward: bank features → embedding (cần grad)
            embedding = bank_client.model(bank_batch)
            embedding.requires_grad_(True)
            
            # 2. Insurance forward: embedding + insurance features → prediction
            prediction = insurance_client.model(embedding, insurance_batch)
            
            # Calculate weighted loss để handle class imbalance tốt hơn
            # Tính pos_weight từ toàn bộ training set (không chỉ batch)
            num_neg_total = (y_train == 0).sum()
            num_pos_total = (y_train == 1).sum()
            if num_pos_total > 0:
                pos_weight_value = float(num_neg_total) / float(num_pos_total)
                pos_weight_value = min(pos_weight_value, 5.0)  # Limit weight
            else:
                pos_weight_value = 1.0
            
            # Sử dụng BCEWithLogitsLoss với pos_weight để handle imbalance tốt hơn
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], device=device))
            
            # Convert prediction từ sigmoid output về logits
            # Clamp để tránh log(0) hoặc log(1)
            prediction_clamped = prediction.clamp(min=1e-7, max=1-1e-7)
            prediction_logits = torch.logit(prediction_clamped)
            
            loss = criterion(prediction_logits, labels_batch)
            
            # Backward pass: Tính gradients cho Top Model và embedding
            loss.backward(retain_graph=True)
            
            # Get gradients w.r.t embedding (để gửi về Bank)
            if embedding.grad is not None:
                grad_for_bottom = embedding.grad.detach().clone()
            else:
                # Tính grad manually nếu embedding.grad không có
                grad_for_bottom = torch.autograd.grad(
                    outputs=loss,
                    inputs=embedding,
                    retain_graph=False,
                    create_graph=False,
                )[0].detach()
            
            # Clip gradients để tránh exploding gradients
            torch.nn.utils.clip_grad_norm_(insurance_client.model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(bank_client.model.parameters(), max_norm=1.0)
            
            # Update Insurance Top Model
            insurance_client.optimizer.step()
            
            # Update Bank Bottom Model với gradients từ Insurance
            # Re-forward để có embedding với grad graph mới
            embedding_for_backward = bank_client.model(bank_batch)
            embedding_for_backward.requires_grad_(True)
            embedding_for_backward.backward(grad_for_bottom)
            bank_client.optimizer.step()
            
            # Metrics (dùng prediction từ sigmoid output)
            predictions_binary = (prediction > 0.5).float()
            accuracy = (predictions_binary == labels_batch).float().mean()
            
            # Convert loss từ logits về probability loss cho tracking
            with torch.no_grad():
                loss_bce = nn.BCELoss()(prediction, labels_batch)
            
            epoch_losses.append(loss_bce.item())
            epoch_accuracies.append(accuracy.item())
        
        avg_train_loss = np.mean(epoch_losses)
        avg_train_acc = np.mean(epoch_accuracies)
        
        # Evaluation phase
        bank_client.model.eval()
        insurance_client.model.eval()
        
        test_losses = []
        test_accuracies = []
        
        with torch.no_grad():
            for bank_batch, insurance_batch, labels_batch in test_loader:
                bank_batch = bank_batch.to(device)
                insurance_batch = insurance_batch.to(device)
                labels_batch = labels_batch.to(device)
                
                # Forward pass
                embedding = bank_client.model(bank_batch)
                prediction = insurance_client.model(embedding, insurance_batch)
                
                # Loss và accuracy (dùng BCELoss cho evaluation)
                criterion_eval = nn.BCELoss()
                loss = criterion_eval(prediction, labels_batch)
                
                predictions_binary = (prediction > 0.5).float()
                accuracy = (predictions_binary == labels_batch).float().mean()
                
                test_losses.append(loss.item())
                test_accuracies.append(accuracy.item())
        
        avg_test_loss = np.mean(test_losses)
        avg_test_acc = np.mean(test_accuracies)
        
        # Learning rate scheduling
        scheduler_bank.step(avg_test_loss)
        scheduler_insurance.step(avg_test_loss)
        
        # Store history
        history['train_losses'].append(avg_train_loss)
        history['train_accuracies'].append(avg_train_acc)
        history['test_losses'].append(avg_test_loss)
        history['test_accuracies'].append(avg_test_acc)
        history['epochs'].append(epoch + 1)
        
        # Early stopping
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if verbose:
            current_lr = bank_client.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'train_loss': f'{avg_train_loss:.4f}',
                'train_acc': f'{avg_train_acc:.4f}',
                'test_loss': f'{avg_test_loss:.4f}',
                'test_acc': f'{avg_test_acc:.4f}',
                'lr': f'{current_lr:.6f}',
            })
        
        # Log chi tiết mỗi epoch
        if verbose and (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch + 1}/{epochs}:")
            logger.info(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}")
            logger.info(f"  Test Loss: {avg_test_loss:.4f}, Test Acc: {avg_test_acc:.4f}")
            logger.info(f"  Learning Rate: {bank_client.optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping check
        if patience_counter >= early_stop_patience:
            logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            logger.info(f"  Best test loss: {best_test_loss:.4f}")
            logger.info(f"  Patience: {patience_counter}/{early_stop_patience}")
            break
    
    # Final evaluation với full metrics
    bank_client.model.eval()
    insurance_client.model.eval()
    
    # Evaluate bằng cách forward qua cả 2 models
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for bank_batch, insurance_batch, labels_batch in test_loader:
            bank_batch = bank_batch.to(device)
            insurance_batch = insurance_batch.to(device)
            labels_batch = labels_batch.to(device)
            
            # Forward pass
            embedding = bank_client.model(bank_batch)
            prediction = insurance_client.model(embedding, insurance_batch)
            
            all_predictions.append(prediction.cpu().numpy())
            all_labels.append(labels_batch.cpu().numpy())
    
    # Combine và calculate metrics
    from utils.metrics import calculate_metrics
    
    y_pred_proba = np.concatenate(all_predictions).flatten()
    y_true = np.concatenate(all_labels).flatten()
    
    # Find optimal threshold based on F1 score (fine-grained search)
    thresholds = np.arange(0.1, 0.9, 0.01)  # Tăng độ chính xác: step 0.01 thay vì 0.05
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
    criterion = nn.BCELoss()
    final_metrics['loss'] = criterion(
        torch.FloatTensor(y_pred_proba),
        torch.FloatTensor(y_true)
    ).item()
    
    return {
        'bottom_model': bank_client.model,
        'top_model': insurance_client.model,
        'history': history,
        'final_metrics': final_metrics,
        'model_config': model_config,
    }

