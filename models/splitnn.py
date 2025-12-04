"""Split Neural Network (SplitNN) models for Vertical Federated Learning.

Kiến trúc SplitNN:
- Bottom Model (Bank): Xử lý financial features → embedding h_B
- Top Model (Insurance): Nhận h_B + insurance features → prediction y
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class BottomModel(nn.Module):
    """Bottom Model cho Bank side trong SplitNN.
    
    Input: Financial features từ Bank (X_B)
    Output: Embedding vector (h_B) được gửi cho Insurance
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int] = [64, 32],
        embedding_size: int = 16,
        dropout_rate: float = 0.3,
        activation: str = 'relu',
    ):
        """Initialize Bottom Model.
        
        Args:
            input_size: Số lượng financial features từ Bank
            hidden_sizes: Số units trong mỗi hidden layer
            embedding_size: Kích thước embedding vector h_B
            dropout_rate: Dropout rate
            activation: Activation function
        """
        super(BottomModel, self).__init__()
        
        self.input_size = input_size
        self.embedding_size = embedding_size
        
        # Build layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            
            # Activation
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            
            # Dropout
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_size = hidden_size
        
        # Output layer: tạo embedding
        layers.append(nn.Linear(prev_size, embedding_size))
        # Không dùng activation ở output để có thể có giá trị âm/dương
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor (batch_size, input_size) - financial features
        
        Returns:
            Embedding tensor (batch_size, embedding_size) - h_B
        """
        return self.network(x)


class TopModel(nn.Module):
    """Top Model cho Insurance side trong SplitNN.
    
    Input: Embedding từ Bank (h_B) + Insurance features (X_I)
    Output: Prediction probability (y_pred)
    """
    
    def __init__(
        self,
        embedding_size: int,
        insurance_input_size: int,
        hidden_sizes: List[int] = [32, 16],
        dropout_rate: float = 0.3,
        activation: str = 'relu',
    ):
        """Initialize Top Model.
        
        Args:
            embedding_size: Kích thước embedding từ Bottom Model
            insurance_input_size: Số lượng insurance features
            hidden_sizes: Số units trong mỗi hidden layer
            dropout_rate: Dropout rate
            activation: Activation function
        """
        super(TopModel, self).__init__()
        
        self.embedding_size = embedding_size
        self.insurance_input_size = insurance_input_size
        
        # Input size = embedding_size + insurance_input_size
        input_size = embedding_size + insurance_input_size
        
        # Build layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            
            # Activation
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            
            # Dropout
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_size = hidden_size
        
        # Output layer: binary classification
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(
        self, 
        embedding: torch.Tensor, 
        insurance_features: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            embedding: Embedding từ Bottom Model (batch_size, embedding_size)
            insurance_features: Insurance features (batch_size, insurance_input_size)
        
        Returns:
            Prediction probability (batch_size, 1)
        """
        # Concatenate embedding và insurance features
        combined = torch.cat([embedding, insurance_features], dim=1)
        return self.network(combined)


class SplitNN(nn.Module):
    """Complete SplitNN model (for centralized training/testing).
    
    Kết hợp Bottom Model và Top Model để train/test trong môi trường centralized.
    Chỉ dùng để baseline comparison, không dùng trong VFL training.
    """
    
    def __init__(
        self,
        bank_input_size: int,
        insurance_input_size: int,
        bottom_hidden_sizes: List[int] = [64, 32],
        top_hidden_sizes: List[int] = [32, 16],
        embedding_size: int = 16,
        dropout_rate: float = 0.3,
        activation: str = 'relu',
    ):
        """Initialize SplitNN.
        
        Args:
            bank_input_size: Số lượng financial features
            insurance_input_size: Số lượng insurance features
            bottom_hidden_sizes: Hidden sizes cho Bottom Model
            top_hidden_sizes: Hidden sizes cho Top Model
            embedding_size: Kích thước embedding
            dropout_rate: Dropout rate
            activation: Activation function
        """
        super(SplitNN, self).__init__()
        
        self.bottom_model = BottomModel(
            input_size=bank_input_size,
            hidden_sizes=bottom_hidden_sizes,
            embedding_size=embedding_size,
            dropout_rate=dropout_rate,
            activation=activation,
        )
        
        self.top_model = TopModel(
            embedding_size=embedding_size,
            insurance_input_size=insurance_input_size,
            hidden_sizes=top_hidden_sizes,
            dropout_rate=dropout_rate,
            activation=activation,
        )
    
    def forward(
        self,
        bank_features: torch.Tensor,
        insurance_features: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            bank_features: Financial features (batch_size, bank_input_size)
            insurance_features: Insurance features (batch_size, insurance_input_size)
        
        Returns:
            Prediction probability (batch_size, 1)
        """
        # Bottom model: bank features → embedding
        embedding = self.bottom_model(bank_features)
        
        # Top model: embedding + insurance features → prediction
        prediction = self.top_model(embedding, insurance_features)
        
        return prediction


def create_splitnn_models(
    bank_input_size: int,
    insurance_input_size: int,
    bottom_hidden_sizes: List[int] = [64, 32],
    top_hidden_sizes: List[int] = [32, 16],
    embedding_size: int = 16,
    dropout_rate: float = 0.3,
    activation: str = 'relu',
) -> tuple:
    """Factory function để tạo Bottom và Top models.
    
    Args:
        bank_input_size: Số lượng financial features
        insurance_input_size: Số lượng insurance features
        bottom_hidden_sizes: Hidden sizes cho Bottom Model
        top_hidden_sizes: Hidden sizes cho Top Model
        embedding_size: Kích thước embedding
        dropout_rate: Dropout rate
        activation: Activation function
    
    Returns:
        Tuple (bottom_model, top_model)
    """
    bottom_model = BottomModel(
        input_size=bank_input_size,
        hidden_sizes=bottom_hidden_sizes,
        embedding_size=embedding_size,
        dropout_rate=dropout_rate,
        activation=activation,
    )
    
    top_model = TopModel(
        embedding_size=embedding_size,
        insurance_input_size=insurance_input_size,
        hidden_sizes=top_hidden_sizes,
        dropout_rate=dropout_rate,
        activation=activation,
    )
    
    return bottom_model, top_model

