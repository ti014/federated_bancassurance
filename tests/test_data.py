"""Unit tests cho data module."""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path

from data.vertical_split import (
    preprocess_vertical_fl_data,
    generate_vertical_fl_data,
)


class TestDataModule(unittest.TestCase):
    """Test cases cho data module."""
    
    def setUp(self):
        """Setup test fixtures."""
        self.n_samples = 100
    
    def test_generate_vertical_fl_data(self):
        """Test generate_vertical_fl_data function."""
        bank_df, insurance_df = generate_vertical_fl_data(
            n_samples=self.n_samples,
            random_state=42,
        )
        
        self.assertIsInstance(bank_df, pd.DataFrame)
        self.assertIsInstance(insurance_df, pd.DataFrame)
        self.assertEqual(len(bank_df), self.n_samples)
        self.assertEqual(len(insurance_df), self.n_samples)
        
        # Check customer_id matching
        self.assertTrue('customer_id' in bank_df.columns)
        self.assertTrue('customer_id' in insurance_df.columns)
        self.assertTrue(bank_df['customer_id'].equals(insurance_df['customer_id']))
    
    def test_preprocess_vertical_fl_data(self):
        """Test preprocess_vertical_fl_data function."""
        bank_df, insurance_df = generate_vertical_fl_data(
            n_samples=self.n_samples,
            random_state=42,
        )
        
        processed = preprocess_vertical_fl_data(
            bank_df=bank_df,
            insurance_df=insurance_df,
            test_size=0.2,
            random_state=42,
        )
        
        # Check keys
        required_keys = [
            'X_bank_train', 'X_bank_test',
            'X_insurance_train', 'X_insurance_test',
            'y_train', 'y_test',
            'bank_features', 'insurance_features',
        ]
        for key in required_keys:
            self.assertIn(key, processed)
        
        # Check shapes
        train_size = int(self.n_samples * 0.8)
        test_size = self.n_samples - train_size
        
        self.assertEqual(len(processed['X_bank_train']), train_size)
        self.assertEqual(len(processed['X_bank_test']), test_size)
        self.assertEqual(len(processed['y_train']), train_size)
        self.assertEqual(len(processed['y_test']), test_size)


if __name__ == '__main__':
    unittest.main()

