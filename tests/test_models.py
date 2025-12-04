"""Unit tests cho models module."""

import unittest
import torch
import numpy as np

from models.splitnn import BottomModel, TopModel, create_splitnn_models


class TestSplitNNModels(unittest.TestCase):
    """Test cases cho SplitNN models."""
    
    def setUp(self):
        """Setup test fixtures."""
        self.bank_input_size = 10
        self.insurance_input_size = 6
        self.embedding_size = 16
        self.batch_size = 32
    
    def test_bottom_model_forward(self):
        """Test BottomModel forward pass."""
        model = BottomModel(
            input_size=self.bank_input_size,
            embedding_size=self.embedding_size,
            hidden_sizes=[64, 32],
        )
        
        x = torch.randn(self.batch_size, self.bank_input_size)
        output = model(x)
        
        self.assertEqual(output.shape, (self.batch_size, self.embedding_size))
    
    def test_top_model_forward(self):
        """Test TopModel forward pass."""
        embedding_size = self.embedding_size
        
        model = TopModel(
            embedding_size=embedding_size,
            insurance_input_size=self.insurance_input_size,
            hidden_sizes=[32, 16],
        )
        
        embedding = torch.randn(self.batch_size, embedding_size)
        insurance_features = torch.randn(self.batch_size, self.insurance_input_size)
        
        output = model(embedding, insurance_features)
        
        self.assertEqual(output.shape, (self.batch_size, 1))
        # Output should be in [0, 1] range (sigmoid)
        self.assertTrue(torch.all(output >= 0))
        self.assertTrue(torch.all(output <= 1))
    
    def test_create_splitnn_models(self):
        """Test create_splitnn_models factory function."""
        bottom_model, top_model = create_splitnn_models(
            bank_input_size=self.bank_input_size,
            insurance_input_size=self.insurance_input_size,
            bottom_hidden_sizes=[64, 32],
            top_hidden_sizes=[32, 16],
            embedding_size=self.embedding_size,
            dropout_rate=0.2,
            activation='relu',
        )
        
        self.assertIsInstance(bottom_model, BottomModel)
        self.assertIsInstance(top_model, TopModel)
        
        # Test forward pass
        bank_features = torch.randn(self.batch_size, self.bank_input_size)
        insurance_features = torch.randn(self.batch_size, self.insurance_input_size)
        
        embedding = bottom_model(bank_features)
        prediction = top_model(embedding, insurance_features)
        
        self.assertEqual(embedding.shape, (self.batch_size, self.embedding_size))
        self.assertEqual(prediction.shape, (self.batch_size, 1))


if __name__ == '__main__':
    unittest.main()

