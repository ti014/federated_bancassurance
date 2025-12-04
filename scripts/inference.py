"""Script để inference với trained Vertical FL models."""

import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.load_model import load_vertical_fl_models


def predict_vertical_fl(
    bottom_model,
    top_model,
    X_bank: np.ndarray,
    X_insurance: np.ndarray,
    device='cpu',
    threshold=0.5
):
    """Predict với Vertical FL models.
    
    Args:
        bottom_model: Trained Bottom Model
        top_model: Trained Top Model
        X_bank: Bank features (n_samples, n_bank_features)
        X_insurance: Insurance features (n_samples, n_insurance_features)
        device: Device
        threshold: Classification threshold
    
    Returns:
        Predictions (binary) và probabilities
    """
    device = torch.device(device)
    bottom_model.eval()
    top_model.eval()
    
    # Convert to tensors
    X_bank_tensor = torch.FloatTensor(X_bank).to(device)
    X_insurance_tensor = torch.FloatTensor(X_insurance).to(device)
    
    with torch.no_grad():
        # Forward pass: Bank → Insurance
        embedding = bottom_model(X_bank_tensor)
        prediction = top_model(embedding, X_insurance_tensor)
        probabilities = torch.sigmoid(prediction).cpu().numpy().flatten()
        predictions = (probabilities >= threshold).astype(int)
    
    return predictions, probabilities


def main():
    parser = argparse.ArgumentParser(description="Inference with trained Vertical FL models")
    parser.add_argument('--bottom-checkpoint', type=str,
                       default='results/models/bank_bottom_model.pth',
                       help='Path to Bottom Model checkpoint')
    parser.add_argument('--top-checkpoint', type=str,
                       default='results/models/insurer_top_model.pth',
                       help='Path to Top Model checkpoint')
    parser.add_argument('--bank-data', type=str, help='Path to Bank features CSV file')
    parser.add_argument('--insurance-data', type=str, help='Path to Insurance features CSV file')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device')
    parser.add_argument('--threshold', type=float, default=0.5, help='Classification threshold')
    parser.add_argument('--output', type=str, help='Output file path for predictions')
    
    args = parser.parse_args()
    
    # Load models
    print(f"Loading models...")
    result = load_vertical_fl_models(
        args.bottom_checkpoint,
        args.top_checkpoint,
        device=args.device
    )
    bottom_model = result['bottom_model']
    top_model = result['top_model']
    
    print("Models loaded successfully!")
    
    if args.bank_data and args.insurance_data:
        # Load data
        print(f"\nLoading Bank data from {args.bank_data}...")
        bank_df = pd.read_csv(args.bank_data)
        
        print(f"Loading Insurance data from {args.insurance_data}...")
        insurance_df = pd.read_csv(args.insurance_data)
        
        # Drop customer ID if exists
        if 'customer_id' in bank_df.columns:
            bank_df = bank_df.drop(columns=['customer_id'])
        if 'customer_id' in insurance_df.columns:
            insurance_df = insurance_df.drop(columns=['customer_id'])
        
        # Drop target if exists
        if 'churn' in insurance_df.columns:
            y_true = insurance_df['churn'].values
            insurance_df = insurance_df.drop(columns=['churn'])
        else:
            y_true = None
        
        # Convert to numpy
        X_bank = bank_df.values
        X_insurance = insurance_df.values
        
        # Predict
        print("Running inference...")
        predictions, probabilities = predict_vertical_fl(
            bottom_model,
            top_model,
            X_bank,
            X_insurance,
            device=args.device,
            threshold=args.threshold
        )
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'prediction': predictions,
            'probability': probabilities,
        })
        
        # Evaluate nếu có target
        if y_true is not None:
            from utils.metrics import calculate_metrics
            metrics = calculate_metrics(y_true, predictions, probabilities)
            print("\nEvaluation metrics:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
        
        # Save results
        if args.output:
            results_df.to_csv(args.output, index=False)
            print(f"\nPredictions saved to {args.output}")
        else:
            print("\nPredictions:")
            print(results_df.head(10))
    else:
        print("\nNo data provided.")
        print("Usage:")
        print("  python scripts/inference.py --bank-data <path> --insurance-data <path>")


if __name__ == '__main__':
    main()
