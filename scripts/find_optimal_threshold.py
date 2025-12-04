"""Script để tìm và test optimal threshold cho F1 score."""

import argparse
import torch
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.splitnn import BottomModel, TopModel
from data.vertical_split import (
    create_vertical_fl_from_bank_churn,
    preprocess_vertical_fl_data,
)
from utils.metrics import calculate_metrics
from utils.logger import setup_logger
import logging

logger = setup_logger(
    name="threshold_utils",
    log_file="results/logs/threshold_optimization.log",
    level=logging.INFO,
)


def load_trained_models(model_dir: str = "results/models"):
    """Load trained models."""
    model_dir = Path(model_dir)
    
    # Load checkpoints
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
    
    # Get model config từ checkpoint
    bottom_config = bottom_checkpoint.get("model_config", {})
    top_config = top_checkpoint.get("model_config", {})
    
    # Create models với config từ checkpoint
    bottom_model = BottomModel(
        input_size=bottom_config.get("bank_input_size", bottom_config.get("input_size", 10)),
        embedding_size=bottom_config.get("embedding_size", 32),
        hidden_sizes=bottom_config.get("bottom_hidden_sizes", bottom_config.get("hidden_sizes", [128, 64])),
        dropout_rate=bottom_config.get("dropout_rate", 0.2),
    )
    
    embedding_size = bottom_config.get("embedding_size", 32)
    insurance_input_size = top_config.get("insurance_input_size", 6)
    
    top_model = TopModel(
        embedding_size=embedding_size,
        insurance_input_size=insurance_input_size,
        hidden_sizes=top_config.get("top_hidden_sizes", top_config.get("hidden_sizes", [64, 32])),
        dropout_rate=top_config.get("dropout_rate", 0.2),
    )
    
    # Load weights
    bottom_model.load_state_dict(bottom_checkpoint["model_state_dict"])
    top_model.load_state_dict(top_checkpoint["model_state_dict"])
    
    bottom_model.eval()
    top_model.eval()
    
    return bottom_model, top_model, bottom_checkpoint["model_config"]


def find_optimal_threshold(
    bottom_model: BottomModel,
    top_model: TopModel,
    X_bank_test: np.ndarray,
    X_insurance_test: np.ndarray,
    y_test: np.ndarray,
    threshold_range: tuple = (0.1, 0.9),
    step: float = 0.01,
) -> dict:
    """Tìm optimal threshold cho F1 score."""
    device = torch.device("cpu")
    bottom_model.to(device)
    top_model.to(device)
    
    # Get predictions
    all_predictions = []
    all_labels = []
    
    batch_size = 64
    with torch.no_grad():
        for i in range(0, len(X_bank_test), batch_size):
            bank_batch = torch.FloatTensor(X_bank_test[i : i + batch_size]).to(device)
            insurance_batch = torch.FloatTensor(X_insurance_test[i : i + batch_size]).to(device)
            
            embedding = bottom_model(bank_batch)
            prediction = top_model(embedding, insurance_batch)
            
            all_predictions.append(prediction.cpu().numpy())
            all_labels.append(y_test[i : i + batch_size])
    
    y_pred_proba = np.concatenate(all_predictions).flatten()
    y_true = np.concatenate(all_labels).flatten()
    
    # Search optimal threshold
    thresholds = np.arange(threshold_range[0], threshold_range[1] + step, step)
    best_f1 = 0
    best_threshold = 0.5
    best_metrics = {}
    
    logger.info(f"\nSearching optimal threshold in range [{threshold_range[0]}, {threshold_range[1]}]...")
    logger.info(f"Total thresholds to test: {len(thresholds)}")
    
    results = []
    
    for threshold in thresholds:
        y_pred_binary = (y_pred_proba > threshold).astype(int)
        metrics = calculate_metrics(y_true, y_pred_binary, y_pred_proba)
        
        f1 = metrics.get("f1", 0)
        results.append({
            "threshold": threshold,
            "f1": f1,
            "precision": metrics.get("precision", 0),
            "recall": metrics.get("recall", 0),
            "accuracy": metrics.get("accuracy", 0),
        })
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_metrics = metrics.copy()
            best_metrics["threshold"] = threshold
    
    # Log top 5 thresholds
    results_sorted = sorted(results, key=lambda x: x["f1"], reverse=True)
    logger.info("\n" + "=" * 80)
    logger.info("TOP 5 THRESHOLDS BY F1 SCORE:")
    logger.info("=" * 80)
    logger.info(f"{'Threshold':<12} {'F1':<8} {'Precision':<12} {'Recall':<12} {'Accuracy':<12}")
    logger.info("-" * 80)
    
    for i, r in enumerate(results_sorted[:5], 1):
        logger.info(
            f"{r['threshold']:<12.3f} {r['f1']:<8.4f} {r['precision']:<12.4f} "
            f"{r['recall']:<12.4f} {r['accuracy']:<12.4f}"
        )
    
    logger.info("\n" + "=" * 80)
    logger.info("OPTIMAL THRESHOLD FOUND:")
    logger.info("=" * 80)
    logger.info(f"Threshold: {best_threshold:.4f}")
    logger.info(f"F1 Score: {best_f1:.4f}")
    logger.info(f"Precision: {best_metrics.get('precision', 0):.4f}")
    logger.info(f"Recall: {best_metrics.get('recall', 0):.4f}")
    logger.info(f"Accuracy: {best_metrics.get('accuracy', 0):.4f}")
    logger.info("=" * 80)
    
    return {
        "optimal_threshold": best_threshold,
        "optimal_f1": best_f1,
        "metrics": best_metrics,
        "all_results": results,
    }


def evaluate_with_threshold(
    bottom_model: BottomModel,
    top_model: TopModel,
    X_bank_test: np.ndarray,
    X_insurance_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Evaluate model với threshold cụ thể."""
    device = torch.device("cpu")
    bottom_model.to(device)
    top_model.to(device)
    
    all_predictions = []
    all_labels = []
    
    batch_size = 64
    with torch.no_grad():
        for i in range(0, len(X_bank_test), batch_size):
            bank_batch = torch.FloatTensor(X_bank_test[i : i + batch_size]).to(device)
            insurance_batch = torch.FloatTensor(X_insurance_test[i : i + batch_size]).to(device)
            
            embedding = bottom_model(bank_batch)
            prediction = top_model(embedding, insurance_batch)
            
            all_predictions.append(prediction.cpu().numpy())
            all_labels.append(y_test[i : i + batch_size])
    
    y_pred_proba = np.concatenate(all_predictions).flatten()
    y_true = np.concatenate(all_labels).flatten()
    
    y_pred_binary = (y_pred_proba > threshold).astype(int)
    metrics = calculate_metrics(y_true, y_pred_binary, y_pred_proba)
    metrics["threshold"] = threshold
    
    return metrics


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Find or test optimal threshold")
    parser.add_argument('--mode', type=str, default='find', choices=['find', 'test'],
                       help='Mode: find optimal threshold or test specific thresholds')
    parser.add_argument('--thresholds', type=float, nargs='+', default=[0.50, 0.60, 0.63, 0.71],
                       help='Thresholds to test (only for test mode)')
    parser.add_argument('--range', type=float, nargs=2, default=[0.1, 0.9],
                       help='Threshold range for search (only for find mode)')
    parser.add_argument('--step', type=float, default=0.01,
                       help='Step size for threshold search')
    
    args = parser.parse_args()
    
    # Load data
    logger.info("=" * 80)
    logger.info(f"THRESHOLD UTILS - MODE: {args.mode.upper()}")
    logger.info("=" * 80)
    
    logger.info("\nLoading test data...")
    bank_churn_file = Path("data/raw/bank_churn.csv")
    bank_churn_file2 = Path("data/raw/Bank Customer Churn Prediction.csv")
    
    bank_file = None
    if bank_churn_file.exists():
        bank_file = str(bank_churn_file)
    elif bank_churn_file2.exists():
        bank_file = str(bank_churn_file2)
    
    if not bank_file:
        logger.error("Bank Churn dataset not found!")
        return
    
    bank_df, insurance_df = create_vertical_fl_from_bank_churn(
        bank_file=bank_file,
        data_dir="data/raw",
        save_dir=None,
    )
    
    processed = preprocess_vertical_fl_data(
        bank_df=bank_df,
        insurance_df=insurance_df,
        test_size=0.2,
        random_state=42,
    )
    
    # Load models
    logger.info("\nLoading trained models...")
    bottom_model, top_model, model_config = load_trained_models()
    logger.info("Models loaded successfully!")
    
    if args.mode == 'find':
        # Find optimal threshold
        logger.info("\nFinding optimal threshold...")
        result = find_optimal_threshold(
            bottom_model=bottom_model,
            top_model=top_model,
            X_bank_test=processed["X_bank_test"],
            X_insurance_test=processed["X_insurance_test"],
            y_test=processed["y_test"],
            threshold_range=tuple(args.range),
            step=args.step,
        )
        
        logger.info("\n" + "=" * 80)
        logger.info("OPTIMIZATION COMPLETED!")
        logger.info("=" * 80)
        logger.info(f"\nRecommended threshold: {result['optimal_threshold']:.4f}")
        logger.info(f"Expected F1 improvement: {result['optimal_f1']:.4f}")
        logger.info("\nYou can use this threshold in inference or retrain with this value.")
    
    elif args.mode == 'test':
        # Test specific thresholds
        logger.info("\n" + "=" * 80)
        logger.info("EVALUATION RESULTS WITH DIFFERENT THRESHOLDS:")
        logger.info("=" * 80)
        logger.info(f"{'Threshold':<12} {'F1':<8} {'Precision':<12} {'Recall':<12} {'Accuracy':<12} {'ROC-AUC':<10}")
        logger.info("-" * 80)
        
        results = {}
        
        for threshold in args.thresholds:
            metrics = evaluate_with_threshold(
                bottom_model=bottom_model,
                top_model=top_model,
                X_bank_test=processed["X_bank_test"],
                X_insurance_test=processed["X_insurance_test"],
                y_test=processed["y_test"],
                threshold=threshold,
            )
            
            results[threshold] = metrics
            
            logger.info(
                f"{threshold:<12.2f} {metrics.get('f1', 0):<8.4f} "
                f"{metrics.get('precision', 0):<12.4f} {metrics.get('recall', 0):<12.4f} "
                f"{metrics.get('accuracy', 0):<12.4f} {metrics.get('roc_auc', 0):<10.4f}"
            )
        
        # Find best threshold
        best_threshold = max(results.keys(), key=lambda k: results[k].get('f1', 0))
        logger.info("\n" + "=" * 80)
        logger.info(f"BEST THRESHOLD: {best_threshold:.4f} (F1: {results[best_threshold].get('f1', 0):.4f})")
        logger.info("=" * 80)


if __name__ == "__main__":
    main()
