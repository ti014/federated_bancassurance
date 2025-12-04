"""Vertical FL clients cho SplitNN architecture.

- BankBottomClient: Client cho Bank (chỉ có Bottom Model)
- InsuranceTopClient: Client cho Insurance (có Top Model)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from torch.utils.data import DataLoader

from models.splitnn import BottomModel, TopModel


class BankBottomClient:
    """Client cho Bank side trong Vertical FL.
    
    Chỉ có Bottom Model, nhận financial features và output embedding.
    """
    
    def __init__(
        self,
        bottom_model: BottomModel,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: Optional[torch.device] = None,
        learning_rate: float = 0.001,
    ):
        """Initialize Bank Bottom Client.
        
        Args:
            bottom_model: Bottom Model instance
            train_loader: DataLoader cho training data (bank features)
            test_loader: DataLoader cho test data (bank features)
            device: PyTorch device
            learning_rate: Learning rate
        """
        self.model = bottom_model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device or torch.device('cpu')
        self.learning_rate = learning_rate
        
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
    
    def forward(self, bank_features: torch.Tensor) -> torch.Tensor:
        """Forward pass: bank features → embedding.
        
        Args:
            bank_features: Financial features (batch_size, n_bank_features)
        
        Returns:
            Embedding (batch_size, embedding_size)
        """
        self.model.eval()
        with torch.no_grad():
            embedding = self.model(bank_features.to(self.device))
        return embedding
    
    def backward(
        self,
        bank_features: torch.Tensor,
        grad_from_top: torch.Tensor,
    ) -> Dict:
        """Backward pass: nhận gradients từ Top Model và tính gradients cho Bottom Model.
        
        LƯU Ý: Không gọi optimizer.step() ở đây. Server sẽ quản lý việc update weights.
        
        Args:
            bank_features: Financial features
            grad_from_top: Gradients từ Top Model (batch_size, embedding_size)
        
        Returns:
            Dictionary với loss và metrics
        """
        self.model.train()
        # KHÔNG gọi zero_grad() ở đây - Server sẽ quản lý
        
        # Forward pass để tạo lại computational graph
        # (Vì embedding cũ đã mất graph khi qua hàm forward tách rời)
        embedding = self.model(bank_features.to(self.device))
        
        # Backward pass với gradients từ Top Model
        # Chỉ tính gradients, KHÔNG update weights
        embedding.backward(grad_from_top.to(self.device))
        
        # Calculate loss (approximate)
        loss = torch.mean(torch.sum(grad_from_top * embedding, dim=1))
        
        return {
            'loss': loss.item(),
            'embedding': embedding.detach().cpu(),
        }
    
    def get_parameters(self) -> List[np.ndarray]:
        """Get model parameters as NumPy arrays.
        
        Returns:
            List of NumPy arrays
        """
        return [param.detach().cpu().numpy() for param in self.model.parameters()]
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters from NumPy arrays.
        
        Args:
            parameters: List of NumPy arrays
        """
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {
            k: torch.tensor(v) for k, v in params_dict
        }
        self.model.load_state_dict(state_dict, strict=False)


class InsuranceTopClient:
    """Client cho Insurance side trong Vertical FL.
    
    Có Top Model, nhận embedding từ Bank + insurance features và output prediction.
    """
    
    def __init__(
        self,
        top_model: TopModel,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: Optional[torch.device] = None,
        learning_rate: float = 0.001,
    ):
        """Initialize Insurance Top Client.
        
        Args:
            top_model: Top Model instance
            train_loader: DataLoader cho training data (insurance features + labels)
            test_loader: DataLoader cho test data (insurance features + labels)
            device: PyTorch device
            learning_rate: Learning rate
        """
        self.model = top_model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device or torch.device('cpu')
        self.learning_rate = learning_rate
        
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()
    
    def forward(
        self,
        embedding: torch.Tensor,
        insurance_features: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass: embedding + insurance features → prediction.
        
        Args:
            embedding: Embedding từ Bottom Model (batch_size, embedding_size)
            insurance_features: Insurance features (batch_size, n_insurance_features)
        
        Returns:
            Prediction probability (batch_size, 1)
        """
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(embedding.to(self.device), insurance_features.to(self.device))
        return prediction
    
    def backward(
        self,
        embedding: torch.Tensor,
        insurance_features: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """Backward pass: tính loss và gradients.
        
        LƯU Ý: Không gọi optimizer.step() ở đây. Server sẽ quản lý việc update weights.
        
        Args:
            embedding: Embedding từ Bottom Model
            insurance_features: Insurance features
            labels: True labels (batch_size, 1)
        
        Returns:
            Tuple (gradients_for_bottom, metrics_dict)
        """
        self.model.train()
        # KHÔNG gọi zero_grad() ở đây - Server sẽ quản lý
        
        # Forward pass
        prediction = self.model(embedding.to(self.device), insurance_features.to(self.device))
        
        # Calculate loss
        labels_tensor = labels.to(self.device).float()
        loss = self.criterion(prediction, labels_tensor)
        
        # Backward pass để tính gradients
        # Chỉ tính gradients, KHÔNG update weights
        loss.backward()
        
        # Get gradients w.r.t embedding (để gửi về Bottom Model)
        # Note: embedding.requires_grad phải True
        if embedding.requires_grad:
            grad_for_bottom = embedding.grad
        else:
            # Nếu embedding không có grad, tính từ prediction
            # Đây là approximation
            grad_for_bottom = torch.autograd.grad(
                outputs=loss,
                inputs=embedding,
                create_graph=True,
                retain_graph=True,
            )[0]
        
        # KHÔNG update weights ở đây - Server sẽ gọi optimizer.step()
        
        # Calculate metrics
        predictions_binary = (prediction > 0.5).float()
        accuracy = (predictions_binary == labels_tensor).float().mean()
        
        return grad_for_bottom.detach(), {
            'loss': loss.item(),
            'accuracy': accuracy.item(),
            'prediction': prediction.detach().cpu(),
        }
    
    def get_parameters(self) -> List[np.ndarray]:
        """Get model parameters as NumPy arrays.
        
        Returns:
            List of NumPy arrays
        """
        return [param.detach().cpu().numpy() for param in self.model.parameters()]
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters from NumPy arrays.
        
        Args:
            parameters: List of NumPy arrays
        """
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {
            k: torch.tensor(v) for k, v in params_dict
        }
        self.model.load_state_dict(state_dict, strict=False)

