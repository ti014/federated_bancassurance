"""Generate ROC Curve cho trained model."""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.splitnn import BottomModel, TopModel
from data.vertical_split import (
    create_vertical_fl_from_bank_churn,
    preprocess_vertical_fl_data,
)
from utils.logger import setup_logger
import logging

logger = setup_logger(
    name="roc_curve",
    log_file="results/logs/roc_curve.log",
    level=logging.INFO,
)


def load_trained_models(model_dir: str = "results/models"):
    """Load trained models."""
    model_dir = Path(model_dir)
    
    bottom_checkpoint = torch.load(
        model_dir / "bank_bottom_model.pth",
        map_location="cpu",
        weights_only=False,
    )
    top_checkpoint = torch.load(
        model_dir / "insurer_top_model.pth",
        map_location="cpu",
        weights_only=False,
    )
    
    bottom_config = bottom_checkpoint.get("model_config", {})
    top_config = top_checkpoint.get("model_config", {})
    
    bank_input_size = bottom_config.get("bank_input_size", bottom_config.get("input_size", 10))
    embedding_size = bottom_config.get("embedding_size", 64)
    insurance_input_size = top_config.get("insurance_input_size", 6)
    
    bottom_model = BottomModel(
        input_size=bank_input_size,
        embedding_size=embedding_size,
        hidden_sizes=bottom_config.get("bottom_hidden_sizes", bottom_config.get("hidden_sizes", [256, 128, 64])),
        dropout_rate=bottom_config.get("dropout_rate", 0.2),
    )
    
    top_model = TopModel(
        embedding_size=embedding_size,
        insurance_input_size=insurance_input_size,
        hidden_sizes=top_config.get("top_hidden_sizes", top_config.get("hidden_sizes", [128, 64, 32])),
        dropout_rate=top_config.get("dropout_rate", 0.2),
    )
    
    bottom_model.load_state_dict(bottom_checkpoint["model_state_dict"])
    top_model.load_state_dict(top_checkpoint["model_state_dict"])
    
    bottom_model.eval()
    top_model.eval()
    
    return bottom_model, top_model


def generate_roc_curve(
    bottom_model: BottomModel,
    top_model: TopModel,
    X_bank_test: np.ndarray,
    X_insurance_test: np.ndarray,
    y_test: np.ndarray,
    save_path: str = "results/plots/roc_curve.png",
):
    """Generate ROC curve cho model.
    
    Args:
        bottom_model: Trained bottom model
        top_model: Trained top model
        X_bank_test: Bank test features
        X_insurance_test: Insurance test features
        y_test: Test labels
        save_path: Path to save ROC curve
    """
    device = torch.device("cpu")
    bottom_model.to(device)
    top_model.to(device)
    
    all_predictions = []
    all_labels = []
    
    batch_size = 64
    with torch.no_grad():
        for i in range(0, len(X_bank_test), batch_size):
            bank_batch = torch.FloatTensor(
                X_bank_test[i : i + batch_size]
            ).to(device)
            insurance_batch = torch.FloatTensor(
                X_insurance_test[i : i + batch_size]
            ).to(device)
            
            embedding = bottom_model(bank_batch)
            prediction = top_model(embedding, insurance_batch)
            
            all_predictions.append(prediction.cpu().numpy())
            all_labels.append(y_test[i : i + batch_size])
    
    y_pred_proba = np.concatenate(all_predictions).flatten()
    y_true = np.concatenate(all_labels).flatten()
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(
        fpr, tpr, 
        color='#2E86AB', 
        lw=2, 
        label=f'ROC Curve (AUC = {roc_auc:.4f})'
    )
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random (AUC = 0.5000)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('ROC Curve - Vertical FL Model', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save
    save_path_obj = Path(save_path)
    save_path_obj.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"ROC Curve saved to {save_path}")
    logger.info(f"AUC Score: {roc_auc:.4f}")
    
    return {
        'fpr': fpr,
        'tpr': tpr,
        'auc': roc_auc,
        'thresholds': thresholds,
    }


def main():
    """Main function."""
    logger.info("=" * 80)
    logger.info("GENERATING ROC CURVE")
    logger.info("=" * 80)
    
    # 1. Load data
    logger.info("\nLoading test data...")
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
    
    # 2. Load models
    logger.info("\nLoading trained models...")
    bottom_model, top_model = load_trained_models()
    logger.info("Models loaded successfully!")
    
    # 3. Generate ROC curve
    logger.info("\nGenerating ROC curve...")
    roc_data = generate_roc_curve(
        bottom_model=bottom_model,
        top_model=top_model,
        X_bank_test=processed["X_bank_test"],
        X_insurance_test=processed["X_insurance_test"],
        y_test=processed["y_test"],
        save_path="results/plots/roc_curve.png",
    )
    
    logger.info("\n" + "=" * 80)
    logger.info("ROC CURVE GENERATED!")
    logger.info("=" * 80)
    logger.info(f"AUC Score: {roc_data['auc']:.4f}")
    logger.info(f"Saved to: results/plots/roc_curve.png")


if __name__ == "__main__":
    main()

