"""So sánh Vertical FL vs Centralized Learning."""

import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.vertical_split import (
    preprocess_vertical_fl_data,
)
from experiments.vertical_fl import run_vertical_fl_training
from experiments.centralized import run_centralized_experiment
from models.splitnn import SplitNN
from utils.visualization import plot_loss_curves, plot_accuracy_comparison, plot_metrics_comparison


def run_centralized_baseline(
    X_bank_train: np.ndarray,
    X_insurance_train: np.ndarray,
    y_train: np.ndarray,
    X_bank_test: np.ndarray,
    X_insurance_test: np.ndarray,
    y_test: np.ndarray,
    model_config: dict,
    device: torch.device,
) -> dict:
    """Run centralized baseline với combined features.
    
    Args:
        X_bank_train, X_insurance_train: Training features
        y_train: Training labels
        X_bank_test, X_insurance_test: Test features
        y_test: Test labels
        model_config: Model configuration
        device: PyTorch device
    
    Returns:
        Dictionary với results
    """
    # Combine features
    X_train_combined = np.concatenate([X_bank_train, X_insurance_train], axis=1)
    X_test_combined = np.concatenate([X_bank_test, X_insurance_test], axis=1)
    
    # Model config cho centralized
    centralized_config = {
        'input_size': X_train_combined.shape[1],
        'hidden_sizes': model_config.get('bottom_hidden_sizes', [64, 32]) + model_config.get('top_hidden_sizes', [32, 16]),
        'dropout_rate': model_config.get('dropout_rate', 0.3),
        'activation': model_config.get('activation', 'relu'),
        'batch_size': model_config.get('batch_size', 32),
        'learning_rate': model_config.get('learning_rate', 0.001),
        'epochs': model_config.get('epochs', 20),
    }
    
    result = run_centralized_experiment(
        X_train=X_train_combined,
        y_train=y_train,
        X_test=X_test_combined,
        y_test=y_test,
        model_config=centralized_config,
        device=device,
        verbose=True,
    )
    
    return result


def main():
    """Compare Vertical FL vs Centralized."""
    print("=" * 80)
    print("COMPARISON: Vertical FL vs Centralized Learning")
    print("=" * 80)
    
    # Load data
    print("\nLoading data...")
    from data.vertical_split import create_vertical_fl_from_bank_churn
    
    bank_churn_file = Path('data/raw/bank_churn.csv')
    if not bank_churn_file.exists():
        print("ERROR: Bank Churn dataset not found!")
        return
    
    bank_df, insurance_df = create_vertical_fl_from_bank_churn(
        bank_file=str(bank_churn_file),
        data_dir='data/raw',
        save_dir=None,
    )
    
    processed = preprocess_vertical_fl_data(
        bank_df=bank_df,
        insurance_df=insurance_df,
        test_size=0.2,
        random_state=42,
    )
    
    # Model config - Use same config as improved model for fair comparison
    model_config = {
        'embedding_size': 64,
        'bottom_hidden_sizes': [256, 128, 64],
        'top_hidden_sizes': [128, 64, 32],
        'dropout_rate': 0.2,
        'activation': 'relu',
        'batch_size': 64,
        'learning_rate': 0.0005,
        'epochs': 100,
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Run Vertical FL
    print("\n" + "=" * 80)
    print("Running Vertical FL...")
    print("=" * 80)
    
    vfl_result = run_vertical_fl_training(
        X_bank_train=processed['X_bank_train'],
        X_insurance_train=processed['X_insurance_train'],
        y_train=processed['y_train'],
        X_bank_test=processed['X_bank_test'],
        X_insurance_test=processed['X_insurance_test'],
        y_test=processed['y_test'],
        model_config=model_config,
        device=device,
        verbose=True,
    )
    
    # Run Centralized Baseline
    print("\n" + "=" * 80)
    print("Running Centralized Baseline...")
    print("=" * 80)
    
    centralized_result = run_centralized_baseline(
        X_bank_train=processed['X_bank_train'],
        X_insurance_train=processed['X_insurance_train'],
        y_train=processed['y_train'],
        X_bank_test=processed['X_bank_test'],
        X_insurance_test=processed['X_insurance_test'],
        y_test=processed['y_test'],
        model_config=model_config,
        device=device,
    )
    
    # Compare results
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    
    vfl_metrics = vfl_result.get('final_metrics', {})
    centralized_metrics = centralized_result.get('final_metrics', {})
    
    print("\nFinal Metrics:")
    print(f"{'Metric':<20} {'Centralized':<15} {'Vertical FL':<15} {'Difference':<15}")
    print("-" * 65)
    
    for metric in centralized_metrics.keys():
        if metric in vfl_metrics:
            centralized_val = centralized_metrics[metric]
            vfl_val = vfl_metrics[metric]
            diff = vfl_val - centralized_val
            diff_pct = (diff / centralized_val * 100) if centralized_val != 0 else 0
            print(f"{metric:<20} {centralized_val:<15.4f} {vfl_val:<15.4f} {diff:+.4f} ({diff_pct:+.2f}%)")
    
    # Plot comparisons
    output_dir = Path('results/plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Loss curves
    if vfl_result.get('history') and centralized_result.get('history'):
        vfl_history = vfl_result['history']
        centralized_history = centralized_result['history']
        
        plot_loss_curves(
            federated_losses=vfl_history.get('test_losses', []),
            centralized_losses=centralized_history.get('test_losses', []),
            save_path=str(output_dir / 'vfl_loss_comparison.png'),
            title="Loss Comparison: Vertical FL vs Centralized",
        )
        
        plot_accuracy_comparison(
            federated_accuracies=vfl_history.get('test_accuracies', []),
            centralized_accuracies=centralized_history.get('test_accuracies', []),
            save_path=str(output_dir / 'vfl_accuracy_comparison.png'),
            title="Accuracy Comparison: Vertical FL vs Centralized",
        )
    
    # Metrics comparison
    metrics_dict = {
        'Centralized': centralized_metrics,
        'Vertical FL': vfl_metrics,
    }
    plot_metrics_comparison(
        metrics_dict=metrics_dict,
        save_path=str(output_dir / 'vfl_metrics_comparison.png'),
        title="Metrics Comparison: Vertical FL vs Centralized",
    )
    
    print(f"\n[OK] Comparison plots saved to {output_dir}")
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()

