"""Main script để chạy Vertical FL với SplitNN architecture.

Module này là entry point chính cho Vertical FL training.
Tất cả imports đều từ các package modules.
"""

import sys
from pathlib import Path

# Add parent directory to path để import packages
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import logging

# Package imports - rõ ràng và nhất quán
from data.vertical_split import (
    generate_vertical_fl_data,
    preprocess_vertical_fl_data,
    create_vertical_fl_from_bank_churn,
)
from experiments.vertical_fl import run_vertical_fl_training
from utils.logger import setup_logger

# Setup logger với level chi tiết
logger = setup_logger(
    name="vertical_fl",
    log_file="results/logs/training.log",
    level=logging.DEBUG,  # DEBUG level để có logs chi tiết nhất
)


def main():
    """Run Vertical FL experiment."""
    import argparse
    parser = argparse.ArgumentParser(description="Run Vertical FL training")
    parser.add_argument('--anti-overfit', action='store_true',
                       help='Use anti-overfitting configuration')
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    if args.anti_overfit:
        logger.info("VERTICAL FEDERATED LEARNING - SplitNN Architecture (ANTI-OVERFITTING)")
    else:
        logger.info("VERTICAL FEDERATED LEARNING - SplitNN Architecture")
    logger.info("=" * 80)
    
    # 1. Generate or load data
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Loading Vertical FL data...")
    logger.info("=" * 80)
    
    # Sử dụng Bank Customer Churn dataset (phù hợp nhất với Bancassurance)
    # Luôn generate từ bank_churn.csv trong memory (không save intermediate files)
    bank_churn_file = Path('data/raw/bank_churn.csv')
    bank_churn_file2 = Path('data/raw/Bank Customer Churn Prediction.csv')
    
    if bank_churn_file.exists() or bank_churn_file2.exists():
        bank_file_path = str(bank_churn_file) if bank_churn_file.exists() else str(bank_churn_file2)
        logger.info("Converting Bank Customer Churn dataset to Vertical FL format (Bancassurance)...")
        logger.info("  This is the most appropriate dataset for Bancassurance domain.")
        bank_df, insurance_df = create_vertical_fl_from_bank_churn(
            bank_file=bank_file_path,
            data_dir='data/raw',
            save_dir=None,  # Không save files, chỉ generate trong memory
        )
    # Last resort: Synthetic data
    else:
        logger.warning("Bank Churn dataset not found. Using synthetic data...")
        logger.warning("  Please download Bank Customer Churn dataset to data/raw/")
        logger.warning("  URL: https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction")
        bank_df, insurance_df = generate_vertical_fl_data(
            n_samples=10000,
            random_state=42,
            save_dir=None,  # Không save synthetic files nữa
        )
    
    # 2. Preprocess data
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Preprocessing data...")
    logger.info("=" * 80)
    
    processed = preprocess_vertical_fl_data(
        bank_df=bank_df,
        insurance_df=insurance_df,
        test_size=0.2,
        random_state=42,
    )
    
    logger.info(f"Training samples: {len(processed['X_bank_train'])}")
    logger.info(f"Test samples: {len(processed['X_bank_test'])}")
    logger.info(f"Bank features: {len(processed['bank_features'])}")
    logger.info(f"Insurance features: {len(processed['insurance_features'])}")
    logger.info(f"Positive ratio: {processed['y_train'].mean():.3f}")
    
    # 3. Model config - Balanced hoặc Anti-overfitting
    if args.anti_overfit:
        model_config = {
            'embedding_size': 32,  # Giảm từ 64 → 32 để giảm capacity
            'bottom_hidden_sizes': [128, 64],  # Giảm từ [256, 128, 64] → [128, 64]
            'top_hidden_sizes': [64, 32],  # Giảm từ [128, 64, 32] → [64, 32]
            'dropout_rate': 0.4,  # Tăng từ 0.2 → 0.4 để giảm overfitting
            'activation': 'relu',
            'batch_size': 64,
            'learning_rate': 0.0003,  # Giảm từ 0.0005 → 0.0003 để stable hơn
            'epochs': 100,
            'weight_decay': 1e-4,  # Tăng từ 1e-5 → 1e-4 để regularization mạnh hơn
        }
        logger.info("\n" + "=" * 80)
        logger.info("ANTI-OVERFITTING CONFIG:")
        logger.info("=" * 80)
        logger.info(f"  Embedding size: {model_config['embedding_size']} (reduced)")
        logger.info(f"  Bottom hidden: {model_config['bottom_hidden_sizes']} (reduced)")
        logger.info(f"  Top hidden: {model_config['top_hidden_sizes']} (reduced)")
        logger.info(f"  Dropout rate: {model_config['dropout_rate']} (increased)")
        logger.info(f"  Learning rate: {model_config['learning_rate']} (reduced)")
        logger.info(f"  Weight decay: {model_config['weight_decay']} (increased)")
    else:
        model_config = {
            'embedding_size': 48,  # Balanced: không quá lớn để tránh overfitting
            'bottom_hidden_sizes': [128, 64],  # Giảm capacity để tránh overfitting
            'top_hidden_sizes': [64, 32],  # Giảm capacity để tránh overfitting
            'dropout_rate': 0.3,  # Tăng dropout để giảm overfitting
            'activation': 'relu',
            'batch_size': 64,
            'learning_rate': 0.0004,  # Giảm learning rate để stable hơn
            'epochs': 100,
            'weight_decay': 5e-5,  # Tăng regularization
        }
    
    # 4. Run Vertical FL training
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Running Vertical FL Training...")
    logger.info("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Option: Sử dụng Server để điều phối (recommended cho production)
    # Set use_server=True để sử dụng VerticalFLServer
    # Note: Hiện tại code vẫn train trực tiếp (use_server=False) để backward compatible
    # Có thể dùng experiments/vertical_fl_with_server.py nếu muốn dùng Server
    
    result = run_vertical_fl_training(
        X_bank_train=processed['X_bank_train'],
        X_insurance_train=processed['X_insurance_train'],
        y_train=processed['y_train'],
        X_bank_test=processed['X_bank_test'],
        X_insurance_test=processed['X_insurance_test'],
        y_test=processed['y_test'],
        model_config=model_config,
        device=device,
        verbose=True,
        use_smote=True,  # Sử dụng SMOTE để xử lý class imbalance
        use_server=False,  # Set True để dùng Server (cần refactor code)
    )
    
    # 5. Save models
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Saving models...")
    logger.info("=" * 80)
    
    output_dir = Path('results/models')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save Bottom Model (Bank)
    model_config_with_sizes = model_config.copy()
    model_config_with_sizes['bank_input_size'] = processed['X_bank_train'].shape[1]
    
    bottom_model_name = 'bank_bottom_model_anti_overfit.pth' if args.anti_overfit else 'bank_bottom_model.pth'
    
    torch.save({
        'model_state_dict': result['bottom_model'].state_dict(),
        'model_config': model_config_with_sizes,
        'final_metrics': result.get('final_metrics', {}),
        'history': result.get('history', {}),
        'model_type': 'bottom',
    }, output_dir / bottom_model_name)
    logger.info(f"Saved Bottom Model to {output_dir / 'bank_bottom_model.pth'}")
    
    # Save Top Model (Insurance)
    model_config_with_sizes_top = model_config.copy()
    model_config_with_sizes_top['insurance_input_size'] = processed['X_insurance_train'].shape[1]
    
    top_model_name = 'insurer_top_model_anti_overfit.pth' if args.anti_overfit else 'insurer_top_model.pth'
    
    torch.save({
        'model_state_dict': result['top_model'].state_dict(),
        'model_config': model_config_with_sizes_top,
        'final_metrics': result.get('final_metrics', {}),
        'history': result.get('history', {}),
        'model_type': 'top',
    }, output_dir / top_model_name)
    logger.info(f"Saved Top Model to {output_dir / 'insurer_top_model.pth'}")
    
    # 6. Results summary
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: Results Summary")
    logger.info("=" * 80)
    
    logger.info("\nFinal Metrics:")
    if result.get('final_metrics'):
        for metric, value in result['final_metrics'].items():
            logger.info(f"  {metric}: {value:.4f}")
    
    logger.info("\nTraining History:")
    history = result.get('history', {})
    if history.get('test_accuracies'):
        logger.info(f"  Final Test Accuracy: {history['test_accuracies'][-1]:.4f}")
        logger.info(f"  Final Test Loss: {history['test_losses'][-1]:.4f}")
    
    # 7. Generate visualizations
    logger.info("\n" + "=" * 80)
    logger.info("STEP 6: Generating Visualizations...")
    logger.info("=" * 80)
    
    from utils.visualization import (
        plot_loss_curves,
        plot_accuracy_comparison,
        plot_metrics_comparison,
    )
    
    plots_dir = Path('results/plots')
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    history = result.get('history', {})
    
    if history.get('train_losses') and history.get('test_losses'):
        # Loss curves
        logger.info("Generating loss curves...")
        plot_loss_curves(
            federated_losses=history['test_losses'],
            centralized_losses=None,
            save_path=str(plots_dir / 'vfl_loss_curves.png'),
            title="Vertical FL Training Loss Curves",
        )
        logger.info(f"  Saved: {plots_dir / 'vfl_loss_curves.png'}")
        
        # Accuracy curves
        logger.info("Generating accuracy curves...")
        plot_accuracy_comparison(
            federated_accuracies=history['test_accuracies'],
            centralized_accuracies=None,
            save_path=str(plots_dir / 'vfl_accuracy_curves.png'),
            title="Vertical FL Training Accuracy Curves",
        )
        logger.info(f"  Saved: {plots_dir / 'vfl_accuracy_curves.png'}")
        
        # Combined loss plot (train vs test)
        logger.info("Generating train vs test loss comparison...")
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        
        plt.figure(figsize=(12, 6))
        epochs = history.get('epochs', range(1, len(history['train_losses']) + 1))
        plt.plot(epochs, history['train_losses'], label='Train Loss', marker='o', linewidth=2, color='#2E86AB')
        plt.plot(epochs, history['test_losses'], label='Test Loss', marker='s', linewidth=2, color='#A23B72')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Vertical FL: Train vs Test Loss', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(str(plots_dir / 'vfl_train_test_loss.png'), dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"  Saved: {plots_dir / 'vfl_train_test_loss.png'}")
        
        # Combined accuracy plot (train vs test)
        logger.info("Generating train vs test accuracy comparison...")
        plt.figure(figsize=(12, 6))
        plt.plot(epochs, history['train_accuracies'], label='Train Accuracy', marker='o', linewidth=2, color='#2E86AB')
        plt.plot(epochs, history['test_accuracies'], label='Test Accuracy', marker='s', linewidth=2, color='#A23B72')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Vertical FL: Train vs Test Accuracy', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1])
        plt.tight_layout()
        plt.savefig(str(plots_dir / 'vfl_train_test_accuracy.png'), dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"  Saved: {plots_dir / 'vfl_train_test_accuracy.png'}")
    
    # Metrics bar chart
    if result.get('final_metrics'):
        logger.info("Generating metrics bar chart...")
        metrics_dict = {
            'Vertical FL': {k: v for k, v in result['final_metrics'].items() 
                           if k not in ['tn', 'fp', 'fn', 'tp', 'threshold_used']}
        }
        plot_metrics_comparison(
            metrics_dict=metrics_dict,
            save_path=str(plots_dir / 'vfl_metrics_bar.png'),
            title="Vertical FL Final Metrics",
        )
        logger.info(f"  Saved: {plots_dir / 'vfl_metrics_bar.png'}")
    
    logger.info("\n" + "=" * 80)
    logger.info("STEP 7: Results Summary")
    logger.info("=" * 80)
    
    logger.info("\nFinal Metrics:")
    if result.get('final_metrics'):
        for metric, value in result['final_metrics'].items():
            logger.info(f"  {metric}: {value:.4f}")
    
    logger.info("\nTraining History:")
    if history.get('test_accuracies'):
        logger.info(f"  Total Epochs: {len(history['epochs'])}")
        logger.info(f"  Final Test Accuracy: {history['test_accuracies'][-1]:.4f}")
        logger.info(f"  Final Test Loss: {history['test_losses'][-1]:.4f}")
        logger.info(f"  Best Test Accuracy: {max(history['test_accuracies']):.4f}")
        logger.info(f"  Best Test Loss: {min(history['test_losses']):.4f}")
    
    logger.info("\n" + "=" * 80)
    logger.info("VERTICAL FL TRAINING COMPLETED!")
    logger.info("=" * 80)
    logger.info(f"\nModels saved to: {output_dir}")
    logger.info("  - bank_bottom_model.pth (Bank side)")
    logger.info("  - insurer_top_model.pth (Insurance side)")
    logger.info(f"\nPlots saved to: {plots_dir}")
    logger.info("  - vfl_loss_curves.png")
    logger.info("  - vfl_accuracy_curves.png")
    logger.info("  - vfl_train_test_loss.png")
    logger.info("  - vfl_train_test_accuracy.png")
    logger.info("  - vfl_metrics_bar.png")
    logger.info("\n" + "=" * 80)


if __name__ == '__main__':
    main()
