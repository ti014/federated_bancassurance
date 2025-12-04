"""Run privacy analysis với trained model."""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.splitnn import BottomModel
from data.vertical_split import (
    create_vertical_fl_from_bank_churn,
    preprocess_vertical_fl_data,
)
from scripts.privacy_analysis import analyze_privacy
from utils.logger import setup_logger
import logging

logger = setup_logger(
    name="privacy_analysis",
    log_file="results/logs/privacy_analysis.log",
    level=logging.INFO,
)


def load_trained_bottom_model(model_path: str = "results/models/bank_bottom_model.pth"):
    """Load trained bottom model."""
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    
    bottom_config = checkpoint.get("model_config", {})
    
    bank_input_size = bottom_config.get("bank_input_size", bottom_config.get("input_size", 10))
    
    bottom_model = BottomModel(
        input_size=bank_input_size,
        embedding_size=bottom_config.get("embedding_size", 64),
        hidden_sizes=bottom_config.get("bottom_hidden_sizes", bottom_config.get("hidden_sizes", [256, 128, 64])),
        dropout_rate=bottom_config.get("dropout_rate", 0.2),
    )
    
    bottom_model.load_state_dict(checkpoint["model_state_dict"])
    bottom_model.eval()
    
    return bottom_model


def main():
    """Main function."""
    logger.info("=" * 80)
    logger.info("PRIVACY ANALYSIS")
    logger.info("=" * 80)
    
    # 1. Load data
    logger.info("\nLoading data...")
    bank_churn_file = Path("data/raw/bank_churn.csv")
    
    if not bank_churn_file.exists():
        logger.error(f"Bank Churn dataset not found: {bank_churn_file}")
        return
    
    bank_df, insurance_df = create_vertical_fl_from_bank_churn(
        bank_file=str(bank_churn_file),
        data_dir="data/raw",
        save_dir=None,
    )
    
    processed = preprocess_vertical_fl_data(
        bank_df=bank_df,
        insurance_df=insurance_df,
        test_size=0.2,
        random_state=42,
    )
    
    # 2. Load model
    logger.info("\nLoading trained Bottom Model...")
    bottom_model = load_trained_bottom_model()
    logger.info("Model loaded successfully!")
    
    # 3. Get bank feature names (after preprocessing, we need to know which features)
    # For Bank Churn dataset, we have 10 features after encoding
    bank_feature_names = [
        'CreditScore', 'Geography_France', 'Geography_Germany', 'Geography_Spain',
        'Gender_Female', 'Gender_Male', 'Age', 'Tenure', 'Balance',
        'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
    ]
    # Adjust based on actual preprocessed features
    # We'll use indices instead
    
    # Use test set for privacy analysis
    X_bank_test = processed['X_bank_test']
    n_features = X_bank_test.shape[1]
    bank_feature_names = [f'Feature_{i}' for i in range(n_features)]
    
    # 4. Run privacy analysis
    logger.info("\nRunning privacy analysis...")
    recovery_results = analyze_privacy(
        bottom_model=bottom_model,
        X_bank=X_bank_test,
        bank_feature_names=bank_feature_names,
        save_dir='results/privacy_analysis',
    )
    
    logger.info("\n" + "=" * 80)
    logger.info("PRIVACY ANALYSIS COMPLETED!")
    logger.info("=" * 80)
    logger.info(f"Results saved to: results/privacy_analysis/")


if __name__ == "__main__":
    main()

