"""Privacy Analysis cho Vertical FL.

Phân tích xem từ embedding h_B có thể recover lại raw financial data không.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.splitnn import BottomModel
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def analyze_privacy(
    bottom_model: BottomModel,
    X_bank: np.ndarray,
    bank_feature_names: list,
    save_dir: str = 'results/privacy_analysis',
):
    """Analyze privacy của embedding.
    
    Thử recover raw features từ embedding bằng linear regression.
    Nếu không thể recover tốt → privacy được bảo vệ.
    
    Args:
        bottom_model: Trained Bottom Model
        X_bank: Bank features (n_samples, n_features)
        bank_feature_names: Tên các features
        save_dir: Directory để save results
    """
    print("=" * 80)
    print("PRIVACY ANALYSIS: Can we recover raw data from embedding?")
    print("=" * 80)
    
    # Generate embeddings
    bottom_model.eval()
    with torch.no_grad():
        X_bank_tensor = torch.FloatTensor(X_bank)
        embeddings = bottom_model(X_bank_tensor).numpy()
    
    print(f"\nEmbedding shape: {embeddings.shape}")
    print(f"Original features shape: {X_bank.shape}")
    
    # Try to recover each feature từ embedding
    recovery_results = {}
    
    print("\n" + "-" * 80)
    print("Attempting to recover original features from embedding...")
    print("-" * 80)
    
    for i, feature_name in enumerate(bank_feature_names):
        # Train linear regression: embedding → original feature
        reg = LinearRegression()
        reg.fit(embeddings, X_bank[:, i])
        
        # Predict
        recovered = reg.predict(embeddings)
        
        # Calculate reconstruction error
        mse = mean_squared_error(X_bank[:, i], recovered)
        mae = mean_absolute_error(X_bank[:, i], recovered)
        r2 = reg.score(embeddings, X_bank[:, i])
        
        recovery_results[feature_name] = {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'recovered': recovered,
        }
        
        print(f"\n{feature_name}:")
        print(f"  MSE: {mse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")
        print(f"  Recovery Quality: {'POOR' if r2 < 0.3 else 'MODERATE' if r2 < 0.7 else 'GOOD'}")
    
    # Summary
    print("\n" + "=" * 80)
    print("PRIVACY ANALYSIS SUMMARY")
    print("=" * 80)
    
    avg_r2 = np.mean([r['r2'] for r in recovery_results.values()])
    print(f"\nAverage R² (reconstruction quality): {avg_r2:.4f}")
    
    if avg_r2 < 0.3:
        privacy_level = "HIGH"
        conclusion = "Embedding provides strong privacy protection. Raw data cannot be recovered."
    elif avg_r2 < 0.7:
        privacy_level = "MODERATE"
        conclusion = "Embedding provides moderate privacy protection. Some information leakage."
    else:
        privacy_level = "LOW"
        conclusion = "Embedding may leak information. Consider increasing embedding size or adding noise."
    
    print(f"Privacy Level: {privacy_level}")
    print(f"Conclusion: {conclusion}")
    
    # Save results
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save recovery results
    results_df = pd.DataFrame({
        'feature': list(recovery_results.keys()),
        'mse': [r['mse'] for r in recovery_results.values()],
        'mae': [r['mae'] for r in recovery_results.values()],
        'r2': [r['r2'] for r in recovery_results.values()],
    })
    results_df.to_csv(save_path / 'privacy_analysis_results.csv', index=False)
    
    # Plot comparison
    n_features = len(bank_feature_names)
    fig, axes = plt.subplots(n_features, 1, figsize=(10, 3 * n_features))
    if n_features == 1:
        axes = [axes]
    
    for i, (feature_name, results) in enumerate(recovery_results.items()):
        ax = axes[i]
        ax.scatter(X_bank[:, i], results['recovered'], alpha=0.5, s=10)
        ax.plot([X_bank[:, i].min(), X_bank[:, i].max()], 
                [X_bank[:, i].min(), X_bank[:, i].max()], 
                'r--', label='Perfect Recovery')
        ax.set_xlabel(f'Original {feature_name}')
        ax.set_ylabel(f'Recovered {feature_name}')
        ax.set_title(f'{feature_name} Recovery (R² = {results["r2"]:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path / 'privacy_analysis_plots.png', dpi=300, bbox_inches='tight')
    print(f"\n[OK] Privacy analysis results saved to {save_path}")
    
    return recovery_results


def main():
    """Main function."""
    import argparse
    parser = argparse.ArgumentParser(description="Privacy Analysis for Vertical FL")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to bottom model checkpoint')
    parser.add_argument('--data', type=str, default='data/raw/bank_data.csv', help='Path to bank data')
    parser.add_argument('--save-dir', type=str, default='results/privacy_analysis', help='Output directory')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading Bottom Model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    model_config = checkpoint.get('model_config', {})
    bank_input_size = model_config.get('bank_input_size', model_config.get('input_size', 10))
    
    from models.splitnn import BottomModel
    bottom_model = BottomModel(
        input_size=bank_input_size,
        hidden_sizes=model_config.get('bottom_hidden_sizes', model_config.get('hidden_sizes', [64, 32])),
        embedding_size=model_config.get('embedding_size', 32),
        dropout_rate=model_config.get('dropout_rate', 0.3),
        activation=model_config.get('activation', 'relu'),
    )
    bottom_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load data
    bank_df = pd.read_csv(args.data)
    bank_features = [col for col in bank_df.columns if col != 'customer_id']
    X_bank = bank_df[bank_features].values
    
    # Normalize (should match training preprocessing)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_bank = scaler.fit_transform(X_bank)
    
    # Run analysis
    analyze_privacy(
        bottom_model=bottom_model,
        X_bank=X_bank,
        bank_feature_names=bank_features,
        save_dir=args.save_dir,
    )


if __name__ == '__main__':
    main()

