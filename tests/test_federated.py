"""Unit tests cho federated module."""

import unittest
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from models.splitnn import BottomModel, TopModel
from federated.vertical_client import BankBottomClient, InsuranceTopClient
from federated.vertical_server import VerticalFLServer


class TestFederatedModule(unittest.TestCase):
    """Test cases cho federated module."""
    
    def setUp(self):
        """Setup test fixtures."""
        self.bank_input_size = 10
        self.insurance_input_size = 6
        self.embedding_size = 16
        self.batch_size = 32
        self.n_samples = 100
        
        # Create models
        self.bottom_model = BottomModel(
            input_size=self.bank_input_size,
            embedding_size=self.embedding_size,
            hidden_sizes=[64, 32],
        )
        
        self.top_model = TopModel(
            hidden_sizes=[32, 16],
            embedding_size=self.embedding_size,
            insurance_input_size=self.insurance_input_size,
        )
        
        # Create data loaders
        bank_train_loader = DataLoader(
            TensorDataset(torch.randn(self.n_samples, self.bank_input_size)),
            batch_size=self.batch_size,
        )
        bank_test_loader = DataLoader(
            TensorDataset(torch.randn(20, self.bank_input_size)),
            batch_size=self.batch_size,
        )
        
        insurance_train_loader = DataLoader(
            TensorDataset(
                torch.randn(self.n_samples, self.insurance_input_size),
                torch.randint(0, 2, (self.n_samples, 1)).float(),
            ),
            batch_size=self.batch_size,
        )
        insurance_test_loader = DataLoader(
            TensorDataset(
                torch.randn(20, self.insurance_input_size),
                torch.randint(0, 2, (20, 1)).float(),
            ),
            batch_size=self.batch_size,
        )
        
        # Create clients
        self.bank_client = BankBottomClient(
            bottom_model=self.bottom_model,
            train_loader=bank_train_loader,
            test_loader=bank_test_loader,
            learning_rate=0.001,
        )
        
        self.insurance_client = InsuranceTopClient(
            top_model=self.top_model,
            train_loader=insurance_train_loader,
            test_loader=insurance_test_loader,
            learning_rate=0.001,
        )
    
    def test_bank_client_forward(self):
        """Test BankBottomClient forward pass."""
        bank_features = torch.randn(self.batch_size, self.bank_input_size)
        embedding = self.bank_client.forward(bank_features)
        
        self.assertEqual(embedding.shape, (self.batch_size, self.embedding_size))
    
    def test_insurance_client_forward(self):
        """Test InsuranceTopClient forward pass."""
        embedding = torch.randn(self.batch_size, self.embedding_size)
        insurance_features = torch.randn(self.batch_size, self.insurance_input_size)
        
        prediction = self.insurance_client.forward(embedding, insurance_features)
        
        self.assertEqual(prediction.shape, (self.batch_size, 1))
        self.assertTrue(torch.all(prediction >= 0))
        self.assertTrue(torch.all(prediction <= 1))
    
    def test_vertical_fl_server_init(self):
        """Test VerticalFLServer initialization."""
        server = VerticalFLServer(
            bank_client=self.bank_client,
            insurance_client=self.insurance_client,
        )
        
        self.assertIsNotNone(server.bank_client)
        self.assertIsNotNone(server.insurance_client)
        self.assertIsNotNone(server.history)
    
    def test_server_coordinate_forward_pass(self):
        """Test Server coordinate_forward_pass."""
        server = VerticalFLServer(
            bank_client=self.bank_client,
            insurance_client=self.insurance_client,
        )
        
        bank_features = torch.randn(self.batch_size, self.bank_input_size)
        insurance_features = torch.randn(self.batch_size, self.insurance_input_size)
        
        prediction, embedding = server.coordinate_forward_pass(
            bank_features,
            insurance_features,
        )
        
        self.assertEqual(embedding.shape, (self.batch_size, self.embedding_size))
        self.assertEqual(prediction.shape, (self.batch_size, 1))


if __name__ == '__main__':
    unittest.main()

