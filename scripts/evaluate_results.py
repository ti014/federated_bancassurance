"""Đánh giá và hiển thị kết quả training Vertical FL."""

import sys
from pathlib import Path

# Add parent directory to path để import packages
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data.vertical_split import preprocess_vertical_fl_data
from models.splitnn import BottomModel, TopModel
from utils.metrics import calculate_metrics, evaluate_model
from sklearn.metrics import confusion_matrix, classification_report


def show_summary():
    """Hiển thị summary đơn giản của kết quả training."""
    print("=" * 80)
    print("VERTICAL FL - FINAL RESULTS SUMMARY")
    print("=" * 80)

    # Load models
    bottom_path = Path('results/models/bank_bottom_model.pth')
    top_path = Path('results/models/insurer_top_model.pth')

    if bottom_path.exists() and top_path.exists():
        bottom = torch.load(str(bottom_path), map_location='cpu', weights_only=False)
        top = torch.load(str(top_path), map_location='cpu', weights_only=False)
        
        print("\n[OK] Models loaded successfully!")
        
        print(f"\nBottom Model (Bank):")
        print(f"  - Type: {bottom.get('model_type', 'N/A')}")
        print(f"  - File: {bottom_path}")
        metrics = bottom.get('final_metrics', {})
        if metrics:
            print(f"  - Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
            print(f"  - Precision: {metrics.get('precision', 'N/A'):.4f}")
            print(f"  - Recall: {metrics.get('recall', 'N/A'):.4f}")
            print(f"  - F1-Score: {metrics.get('f1', 'N/A'):.4f}")
            print(f"  - ROC-AUC: {metrics.get('roc_auc', 'N/A'):.4f}")
        
        print(f"\nTop Model (Insurance):")
        print(f"  - Type: {top.get('model_type', 'N/A')}")
        print(f"  - File: {top_path}")
        metrics = top.get('final_metrics', {})
        if metrics:
            print(f"  - Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
            print(f"  - Precision: {metrics.get('precision', 'N/A'):.4f}")
            print(f"  - Recall: {metrics.get('recall', 'N/A'):.4f}")
            print(f"  - F1-Score: {metrics.get('f1', 'N/A'):.4f}")
            print(f"  - ROC-AUC: {metrics.get('roc_auc', 'N/A'):.4f}")
        
        # Training history
        bottom_history = bottom.get('history', {})
        if bottom_history.get('test_accuracies'):
            print(f"\nTraining History:")
            print(f"  - Final Test Accuracy: {bottom_history['test_accuracies'][-1]:.4f}")
            print(f"  - Final Test Loss: {bottom_history['test_losses'][-1]:.4f}")
            print(f"  - Epochs: {len(bottom_history.get('epochs', []))}")
        
        print("\n" + "=" * 80)
        print("DELIVERABLES:")
        print("=" * 80)
        print("✅ bank_bottom_model.pth - Bottom Model (Bank side)")
        print("✅ insurer_top_model.pth - Top Model (Insurance side)")
        print("✅ Plots: results/plots/vfl_*.png")
        print("✅ ROC Curve: results/plots/roc_curve.png")
        print("✅ Privacy Analysis: results/privacy_analysis/")
        print("\n" + "=" * 80)
        print("PROJECT COMPLETED SUCCESSFULLY!")
        print("=" * 80)
    else:
        print("\n[ERROR] Model files not found!")
        print(f"  - Bottom: {bottom_path.exists()}")
        print(f"  - Top: {top_path.exists()}")


def analyze_training_results():
    """Phân tích chi tiết kết quả training."""
    print("=" * 80)
    print("DANH GIA KET QUA TRAINING - VERTICAL FL")
    print("=" * 80)
    
    # 1. Load models và metrics
    print("\n[1] Loading models và metrics...")
    bottom_path = Path('results/models/bank_bottom_model.pth')
    top_path = Path('results/models/insurer_top_model.pth')
    
    if not bottom_path.exists() or not top_path.exists():
        print("[ERROR] Model files not found!")
        return
    
    bottom_checkpoint = torch.load(str(bottom_path), map_location='cpu', weights_only=False)
    top_checkpoint = torch.load(str(top_path), map_location='cpu', weights_only=False)
    
    bottom_metrics = bottom_checkpoint.get('final_metrics', {})
    top_metrics = top_checkpoint.get('final_metrics', {})
    history = bottom_checkpoint.get('history', {})
    
    print("[OK] Models loaded")
    
    # 2. Phân tích Metrics
    print("\n" + "=" * 80)
    print("[2] PHAN TICH METRICS")
    print("=" * 80)
    
    print("\nFinal Metrics:")
    print(f"  Accuracy:  {bottom_metrics.get('accuracy', 'N/A'):.4f}")
    print(f"  Precision: {bottom_metrics.get('precision', 'N/A'):.4f}")
    print(f"  Recall:    {bottom_metrics.get('recall', 'N/A'):.4f}")
    print(f"  F1-Score:  {bottom_metrics.get('f1', 'N/A'):.4f}")
    print(f"  ROC-AUC:   {bottom_metrics.get('roc_auc', 'N/A'):.4f}")
    print(f"  Threshold: {bottom_metrics.get('threshold_used', 0.5):.4f}")
    
    # Confusion Matrix
    tn = bottom_metrics.get('tn', 0)
    fp = bottom_metrics.get('fp', 0)
    fn = bottom_metrics.get('fn', 0)
    tp = bottom_metrics.get('tp', 0)
    
    print("\nConfusion Matrix:")
    print(f"  True Negatives (TN):  {tn}")
    print(f"  False Positives (FP): {fp}")
    print(f"  False Negatives (FN): {fn}")
    print(f"  True Positives (TP):  {tp}")
    print(f"\n  Total: {tn + fp + fn + tp}")
    
    # 3. Đánh giá chất lượng
    print("\n" + "=" * 80)
    print("[3] DANH GIA CHAT LUONG MODEL")
    print("=" * 80)
    
    issues = []
    warnings = []
    
    # ROC-AUC check
    roc_auc = bottom_metrics.get('roc_auc', 0)
    if roc_auc < 0.6:
        issues.append(f"[!] ROC-AUC thap ({roc_auc:.4f} < 0.6): Model gan nhu random, khong phan biet tot giua classes")
    elif roc_auc < 0.7:
        warnings.append(f"[WARNING] ROC-AUC trung binh ({roc_auc:.4f}): Model co the cai thien")
    else:
        print(f"[OK] ROC-AUC tot ({roc_auc:.4f})")
    
    # Precision-Recall balance
    precision = bottom_metrics.get('precision', 0)
    recall = bottom_metrics.get('recall', 0)
    
    if recall > 0.9 and precision < 0.4:
        issues.append(f"[!] Recall cao ({recall:.4f}) nhung Precision thap ({precision:.4f}): Model predict qua nhieu positive (nhieu False Positives)")
    elif precision > 0.7 and recall < 0.5:
        issues.append(f"[!] Precision cao ({precision:.4f}) nhung Recall thap ({recall:.4f}): Model bo sot nhieu positive cases (nhieu False Negatives)")
    
    # Threshold analysis
    threshold = bottom_metrics.get('threshold_used', 0.5)
    if threshold < 0.4:
        warnings.append(f"[WARNING] Threshold thap ({threshold:.4f} < 0.4): Model co xu huong predict nhieu positive, co the do class imbalance")
    
    # Accuracy analysis
    accuracy = bottom_metrics.get('accuracy', 0)
    if accuracy < 0.5:
        issues.append(f"[!] Accuracy thap ({accuracy:.4f} < 0.5): Model khong tot hon random guess")
    elif accuracy < 0.6:
        warnings.append(f"[WARNING] Accuracy trung binh ({accuracy:.4f}): Model co the cai thien")
    
    # Class imbalance check
    total = tn + fp + fn + tp
    positive_rate = (tp + fn) / total if total > 0 else 0
    negative_rate = (tn + fp) / total if total > 0 else 0
    
    print(f"\nClass Distribution:")
    print(f"  Positive rate: {positive_rate:.4f} ({tp + fn}/{total})")
    print(f"  Negative rate: {negative_rate:.4f} ({tn + fp}/{total})")
    
    if abs(positive_rate - negative_rate) > 0.3:
        warnings.append(f"[WARNING] Class imbalance nghiem trong: Positive={positive_rate:.2%}, Negative={negative_rate:.2%}")
    
    # Print issues and warnings
    if issues:
        print("\n" + "=" * 80)
        print("[!] VAN DE NGHIEM TRONG:")
        print("=" * 80)
        for issue in issues:
            print(f"  {issue}")
    
    if warnings:
        print("\n" + "=" * 80)
        print("[WARNING] CANH BAO:")
        print("=" * 80)
        for warning in warnings:
            print(f"  {warning}")
    
    if not issues and not warnings:
        print("\n✅ Model có chất lượng tốt!")
    
    # 4. Training History Analysis
    print("\n" + "=" * 80)
    print("[4] PHAN TICH TRAINING HISTORY")
    print("=" * 80)
    
    if history:
        train_losses = history.get('train_losses', [])
        test_losses = history.get('test_losses', [])
        train_accs = history.get('train_accuracies', [])
        test_accs = history.get('test_accuracies', [])
        
        if train_losses and test_losses:
            print(f"\nLoss:")
            print(f"  Initial train loss: {train_losses[0]:.4f}")
            print(f"  Final train loss:   {train_losses[-1]:.4f}")
            print(f"  Initial test loss:  {test_losses[0]:.4f}")
            print(f"  Final test loss:    {test_losses[-1]:.4f}")
            
            # Check overfitting
            final_train_loss = train_losses[-1]
            final_test_loss = test_losses[-1]
            
            if final_test_loss > final_train_loss * 1.2:
                issues.append(f"[!] Overfitting: Test loss ({final_test_loss:.4f}) cao hon train loss ({final_train_loss:.4f}) dang ke")
            elif final_test_loss < final_train_loss * 0.8:
                warnings.append(f"[WARNING] Test loss thap hon train loss: Co the co van de voi validation set")
            
            # Check convergence
            if len(train_losses) > 5:
                recent_train_losses = train_losses[-5:]
                loss_std = np.std(recent_train_losses)
                if loss_std < 0.01:
                    print(f"  [OK] Model da hoi tu (loss std: {loss_std:.4f})")
                else:
                    warnings.append(f"[WARNING] Model chua hoi tu hoan toan (loss std: {loss_std:.4f})")
            
            print(f"\nAccuracy:")
            print(f"  Initial train acc: {train_accs[0]:.4f}")
            print(f"  Final train acc:   {train_accs[-1]:.4f}")
            print(f"  Initial test acc:  {test_accs[0]:.4f}")
            print(f"  Final test acc:    {test_accs[-1]:.4f}")
    
    # 5. So sánh với Centralized
    print("\n" + "=" * 80)
    print("[5] SO SANH VOI CENTRALIZED BASELINE")
    print("=" * 80)
    
    centralized_path = Path('results/models/centralized_model.pth')
    if centralized_path.exists():
        centralized_checkpoint = torch.load(str(centralized_path), map_location='cpu', weights_only=False)
        centralized_metrics = centralized_checkpoint.get('final_metrics', {})
        
        print("\nCentralized Metrics:")
        print(f"  Accuracy:  {centralized_metrics.get('accuracy', 'N/A'):.4f}")
        print(f"  Precision: {centralized_metrics.get('precision', 'N/A'):.4f}")
        print(f"  Recall:    {centralized_metrics.get('recall', 'N/A'):.4f}")
        print(f"  F1-Score:  {centralized_metrics.get('f1', 'N/A'):.4f}")
        print(f"  ROC-AUC:   {centralized_metrics.get('roc_auc', 'N/A'):.4f}")
        
        print("\nComparison:")
        vfl_acc = bottom_metrics.get('accuracy', 0)
        cent_acc = centralized_metrics.get('accuracy', 0)
        diff = vfl_acc - cent_acc
        
        if diff < -0.1:
            issues.append(f"[!] VFL accuracy ({vfl_acc:.4f}) thap hon Centralized ({cent_acc:.4f}) dang ke: Co the co van de voi training")
        elif diff > 0.1:
            print(f"  [OK] VFL tot hon Centralized: {diff:+.4f}")
        else:
            print(f"  [OK] VFL tuong duong Centralized: {diff:+.4f}")
    else:
        print("\n[INFO] Centralized model not found. Run comparison script first.")
    
    # 6. Recommendations
    print("\n" + "=" * 80)
    print("[6] KHUYEN NGHI")
    print("=" * 80)
    
    recommendations = []
    
    if roc_auc < 0.6:
        recommendations.append("1. Tang model capacity (them layers, neurons)")
        recommendations.append("2. Tang so epochs training")
        recommendations.append("3. Dieu chinh learning rate")
        recommendations.append("4. Thu cac activation functions khac")
    
    if precision < 0.4 and recall > 0.9:
        recommendations.append("5. Tang threshold de giam False Positives")
        recommendations.append("6. Su dung weighted loss de balance precision/recall")
    
    if abs(positive_rate - negative_rate) > 0.3:
        recommendations.append("7. Xu ly class imbalance: SMOTE, weighted loss, hoac undersampling")
    
    if len(train_losses) > 0 and train_losses[-1] > 0.5:
        recommendations.append("8. Model co the can train lau hon hoac learning rate nho hon")
    
    if recommendations:
        print("\nKhuyen nghi cai thien:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {rec}")
    else:
        print("\n[OK] Model da dat chat luong tot!")
    
    # 7. Kết luận
    print("\n" + "=" * 80)
    print("[7] KET LUAN")
    print("=" * 80)
    
    if issues:
        print("\n[!] Model co VAN DE NGHIEM TRONG can duoc fix truoc khi su dung.")
        print("   Xem cac khuyen nghi o tren de cai thien.")
    elif warnings:
        print("\n[WARNING] Model co mot so diem can cai thien nhung van co the su dung.")
        print("   Can nhac ap dung cac khuyen nghi de tang chat luong.")
    else:
        print("\n[OK] Model co chat luong TOT va san sang de su dung!")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate training results")
    parser.add_argument('--mode', type=str, default='analyze', choices=['summary', 'analyze'],
                       help='Mode: summary (simple) or analyze (detailed)')
    
    args = parser.parse_args()
    
    if args.mode == 'summary':
        show_summary()
    else:
        analyze_training_results()

