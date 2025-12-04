"""Visualization utilities for experiments."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def plot_loss_curves(
    federated_losses: List[float],
    centralized_losses: Optional[List[float]] = None,
    save_path: Optional[str] = None,
    title: str = "Training Loss Comparison",
) -> None:
    """Plot loss curves cho FL vs Centralized.
    
    Args:
        federated_losses: List of loss values từ FL training
        centralized_losses: List of loss values từ centralized training (optional)
        save_path: Đường dẫn để save plot
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    
    rounds = range(1, len(federated_losses) + 1)
    plt.plot(rounds, federated_losses, label='Federated Learning', marker='o', linewidth=2)
    
    if centralized_losses is not None:
        centralized_rounds = range(1, len(centralized_losses) + 1)
        plt.plot(centralized_rounds, centralized_losses, label='Centralized Learning', marker='s', linewidth=2)
    
    plt.xlabel('Round / Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    # Don't show plot in non-interactive mode (for scripts)
    try:
        plt.show()
    except:
        plt.close()


def plot_accuracy_comparison(
    federated_accuracies: List[float],
    centralized_accuracies: Optional[List[float]] = None,
    save_path: Optional[str] = None,
    title: str = "Accuracy Comparison",
) -> None:
    """Plot accuracy comparison.
    
    Args:
        federated_accuracies: List of accuracy values từ FL
        centralized_accuracies: List of accuracy values từ centralized (optional)
        save_path: Đường dẫn để save plot
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    
    rounds = range(1, len(federated_accuracies) + 1)
    plt.plot(rounds, federated_accuracies, label='Federated Learning', marker='o', linewidth=2, color='#2E86AB')
    
    if centralized_accuracies is not None:
        centralized_rounds = range(1, len(centralized_accuracies) + 1)
        plt.plot(centralized_rounds, centralized_accuracies, label='Centralized Learning', marker='s', linewidth=2, color='#A23B72')
    
    plt.xlabel('Round / Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    # Don't show plot in non-interactive mode (for scripts)
    try:
        plt.show()
    except:
        plt.close()


def plot_client_heatmap(
    client_metrics: Dict[int, Dict[str, float]],
    metric_name: str = 'accuracy',
    save_path: Optional[str] = None,
    title: Optional[str] = None,
) -> None:
    """Plot heatmap của metrics theo clients.
    
    Args:
        client_metrics: Dictionary {client_id: {metric: value}}
        metric_name: Tên metric để visualize
        save_path: Đường dẫn để save plot
        title: Plot title
    """
    # Prepare data
    data = []
    for client_id, metrics in sorted(client_metrics.items()):
        data.append([client_id, metrics.get(metric_name, 0.0)])
    
    df = pd.DataFrame(data, columns=['Client', metric_name])
    
    # Create heatmap
    plt.figure(figsize=(10, 6))
    pivot_data = df.pivot_table(values=metric_name, index='Client', aggfunc='mean')
    
    sns.heatmap(
        pivot_data,
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        cbar_kws={'label': metric_name},
        linewidths=0.5,
    )
    
    plt.title(title or f'{metric_name.capitalize()} per Client', fontsize=14, fontweight='bold')
    plt.xlabel('')
    plt.ylabel('Client ID', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    # Don't show plot in non-interactive mode (for scripts)
    try:
        plt.show()
    except:
        plt.close()


def plot_metrics_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
    title: str = "Metrics Comparison",
) -> None:
    """Plot comparison của nhiều metrics.
    
    Args:
        metrics_dict: Dictionary {model_name: {metric: value}}
        save_path: Đường dẫn để save plot
        title: Plot title
    """
    # Prepare data
    df_data = []
    for model_name, metrics in metrics_dict.items():
        for metric_name, value in metrics.items():
            df_data.append({
                'Model': model_name,
                'Metric': metric_name,
                'Value': value,
            })
    
    df = pd.DataFrame(df_data)
    
    # Create bar plot
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='Metric', y='Value', hue='Model')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('Score', fontsize=12)
    plt.xlabel('Metric', fontsize=12)
    plt.legend(title='Model', fontsize=10)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    # Don't show plot in non-interactive mode (for scripts)
    try:
        plt.show()
    except:
        plt.close()

