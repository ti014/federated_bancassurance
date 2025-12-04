"""Vertical FL Server để điều phối training giữa Bank và Insurance clients.

Server đóng vai trò coordinator:
- Điều phối forward/backward pass giữa Bank và Insurance
- Quản lý training rounds
- Synchronize embeddings và gradients
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score

from federated.vertical_client import BankBottomClient, InsuranceTopClient
from utils.metrics import evaluate_model


class VerticalFLServer:
    """Server để điều phối Vertical FL training giữa Bank và Insurance.
    
    Server không có model riêng, chỉ điều phối việc trao đổi:
    - Embeddings từ Bank → Insurance
    - Gradients từ Insurance → Bank
    """
    
    def __init__(
        self,
        bank_client: BankBottomClient,
        insurance_client: InsuranceTopClient,
        device: Optional[torch.device] = None,
    ):
        """Initialize Vertical FL Server.
        
        Args:
            bank_client: Bank Bottom Client
            insurance_client: Insurance Top Client
            device: PyTorch device
        """
        self.bank_client = bank_client
        self.insurance_client = insurance_client
        self.device = device or torch.device('cpu')
        
        # Training history
        self.history = {
            'train_losses': [],
            'train_accuracies': [],
            'test_losses': [],
            'test_accuracies': [],
            'epochs': [],
        }
    
    def coordinate_forward_pass(
        self,
        bank_features: torch.Tensor,
        insurance_features: torch.Tensor,
    ) -> torch.Tensor:
        """Điều phối forward pass: Bank → Insurance.
        
        Args:
            bank_features: Bank features (batch_size, n_bank_features)
            insurance_features: Insurance features (batch_size, n_insurance_features)
        
        Returns:
            Prediction từ Top Model
        """
        # 1. Bank forward: bank features → embedding
        embedding = self.bank_client.model(bank_features.to(self.device))
        embedding.requires_grad_(True)
        
        # 2. Server gửi embedding cho Insurance (simulation: trực tiếp gọi)
        # Trong thực tế: Server sẽ gửi embedding qua network đến Insurance client
        
        # 3. Insurance forward: embedding + insurance features → prediction
        prediction = self.insurance_client.model(embedding, insurance_features.to(self.device))
        
        return prediction, embedding
    
    def coordinate_backward_pass(
        self,
        prediction: torch.Tensor,
        embedding: torch.Tensor,
        labels: torch.Tensor,
        bank_features: torch.Tensor,
        insurance_features: torch.Tensor,
        pos_weight: float = 1.0,
    ) -> Dict:
        """Điều phối backward pass: Insurance → Bank.
        
        Args:
            prediction: Prediction từ Top Model
            embedding: Embedding từ Bottom Model
            labels: True labels
            bank_features: Bank features (để backward cho Bottom Model)
            insurance_features: Insurance features
            pos_weight: Weight cho positive class (class imbalance)
        
        Returns:
            Dictionary với loss và metrics
        """
        import torch.nn as nn
        
        # 1. Tính loss tại Insurance (có labels)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=self.device))
        
        # Convert prediction từ sigmoid output về logits
        prediction_clamped = prediction.clamp(min=1e-7, max=1-1e-7)
        prediction_logits = torch.logit(prediction_clamped)
        
        loss = criterion(prediction_logits, labels.to(self.device))
        
        # 2. Backward cho Top Model (Insurance)
        self.insurance_client.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.insurance_client.model.parameters(), max_norm=1.0)
        
        # 3. Get gradients w.r.t embedding (để gửi về Bank)
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
        
        # 4. Server gửi gradients cho Bank (simulation: trực tiếp gọi)
        # Trong thực tế: Server sẽ gửi gradients qua network đến Bank client
        
        # 5. Backward cho Bottom Model (Bank)
        self.bank_client.optimizer.zero_grad()
        
        # Re-forward để có embedding với grad graph mới
        embedding_for_backward = self.bank_client.model(bank_features.to(self.device))
        embedding_for_backward.requires_grad_(True)
        embedding_for_backward.backward(grad_for_bottom)
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.bank_client.model.parameters(), max_norm=1.0)
        
        # 6. Update models (Server quản lý optimizer.step())
        self.insurance_client.optimizer.step()
        self.bank_client.optimizer.step()
        
        # Metrics
        predictions_binary = (prediction > 0.5).float()
        labels_tensor = labels.to(self.device)
        accuracy = (predictions_binary == labels_tensor).float().mean()
        
        # Convert loss từ logits về probability loss cho tracking
        with torch.no_grad():
            loss_bce = nn.BCELoss()(prediction, labels_tensor)
        
        # Tính F1-Score và AUC (quan trọng cho imbalanced data)
        y_true = labels_tensor.cpu().numpy().flatten()
        y_pred_proba = prediction.detach().cpu().numpy().flatten()
        y_pred_binary = predictions_binary.cpu().numpy().flatten()
        
        # F1-Score: Check xem có cả 2 classes không để tránh lỗi
        if len(np.unique(y_true)) > 1:
            f1 = f1_score(y_true, y_pred_binary, zero_division=0)
            try:
                auc = roc_auc_score(y_true, y_pred_proba)
            except ValueError:
                # Nếu chỉ có 1 class trong batch, AUC không tính được
                auc = 0.5
        else:
            # Nếu chỉ có 1 class trong batch
            f1 = 0.0
            auc = 0.5
        
        return {
            'loss': loss_bce.item(),
            'accuracy': accuracy.item(),
            'f1': f1,
            'auc': auc,
        }
    
    def train_round(
        self,
        train_loader,
        y_train: np.ndarray,
        verbose: bool = True,
    ) -> Dict:
        """Train một round (epoch).
        
        Args:
            train_loader: DataLoader với (bank_features, insurance_features, labels)
            y_train: Training labels (để tính pos_weight)
            verbose: Print progress
        
        Returns:
            Dictionary với average loss và accuracy
        """
        self.bank_client.model.train()
        self.insurance_client.model.train()
        
        epoch_losses = []
        epoch_accuracies = []
        
        # Tính pos_weight từ toàn bộ training set
        num_neg_total = (y_train == 0).sum()
        num_pos_total = (y_train == 1).sum()
        if num_pos_total > 0:
            pos_weight_value = float(num_neg_total) / float(num_pos_total)
            pos_weight_value = min(pos_weight_value, 5.0)  # Limit weight
        else:
            pos_weight_value = 1.0
        
        for bank_batch, insurance_batch, labels_batch in train_loader:
            bank_batch = bank_batch.to(self.device)
            insurance_batch = insurance_batch.to(self.device)
            labels_batch = labels_batch.to(self.device)
            
            # Forward pass (Server điều phối)
            prediction, embedding = self.coordinate_forward_pass(
                bank_batch,
                insurance_batch,
            )
            
            # Backward pass (Server điều phối)
            metrics = self.coordinate_backward_pass(
                prediction=prediction,
                embedding=embedding,
                labels=labels_batch,
                bank_features=bank_batch,
                insurance_features=insurance_batch,
                pos_weight=pos_weight_value,
            )
            
            epoch_losses.append(metrics['loss'])
            epoch_accuracies.append(metrics['accuracy'])
        
        return {
            'loss': np.mean(epoch_losses),
            'accuracy': np.mean(epoch_accuracies),
        }
    
    def evaluate(
        self,
        test_loader,
    ) -> Dict:
        """Evaluate trên test set.
        
        Args:
            test_loader: DataLoader với (bank_features, insurance_features, labels)
        
        Returns:
            Dictionary với test loss và accuracy
        """
        self.bank_client.model.eval()
        self.insurance_client.model.eval()
        
        test_losses = []
        test_accuracies = []
        
        import torch.nn as nn
        
        with torch.no_grad():
            for bank_batch, insurance_batch, labels_batch in test_loader:
                bank_batch = bank_batch.to(self.device)
                insurance_batch = insurance_batch.to(self.device)
                labels_batch = labels_batch.to(self.device)
                
                # Forward pass
                embedding = self.bank_client.model(bank_batch)
                prediction = self.insurance_client.model(embedding, insurance_batch)
                
                # Loss và accuracy
                criterion_eval = nn.BCELoss()
                loss = criterion_eval(prediction, labels_batch)
                
                predictions_binary = (prediction > 0.5).float()
                accuracy = (predictions_binary == labels_batch).float().mean()
                
                test_losses.append(loss.item())
                test_accuracies.append(accuracy.item())
        
        return {
            'loss': np.mean(test_losses),
            'accuracy': np.mean(test_accuracies),
        }

