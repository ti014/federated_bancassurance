"""Script để load trained Vertical FL models (BottomModel và TopModel)."""

import argparse
import torch
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.splitnn import BottomModel, TopModel


def load_vertical_fl_models(
    bottom_checkpoint: str,
    top_checkpoint: str,
    device: str = 'cpu'
):
    """Load trained Vertical FL models từ checkpoints.
    
    Args:
        bottom_checkpoint: Đường dẫn đến Bottom Model checkpoint (.pth)
        top_checkpoint: Đường dẫn đến Top Model checkpoint (.pth)
        device: Device để load model ('cpu' hoặc 'cuda')
    
    Returns:
        Dictionary với bottom_model, top_model, configs, và metrics
    """
    device = torch.device(device)
    
    # Load Bottom Model
    bottom_checkpoint_data = torch.load(bottom_checkpoint, map_location=device, weights_only=False)
    bottom_config = bottom_checkpoint_data.get('model_config', {})
    
    # Get bank input size from config or infer from model
    bank_input_size = bottom_config.get('bank_input_size', 10)
    # If not in config, try to infer from saved model
    if 'bank_input_size' not in bottom_config:
        # Try to get from model state dict
        first_layer_key = [k for k in bottom_checkpoint_data['model_state_dict'].keys() if 'network.0.weight' in k]
        if first_layer_key:
            bank_input_size = bottom_checkpoint_data['model_state_dict'][first_layer_key].shape[1]
    
    bottom_model = BottomModel(
        input_size=bank_input_size,
        embedding_size=bottom_config.get('embedding_size', 32),
        hidden_sizes=bottom_config.get('bottom_hidden_sizes', [128, 64]),
        dropout_rate=bottom_config.get('dropout_rate', 0.2),
        activation=bottom_config.get('activation', 'relu'),
    )
    bottom_model.load_state_dict(bottom_checkpoint_data['model_state_dict'])
    bottom_model.to(device)
    bottom_model.eval()
    
    # Load Top Model
    top_checkpoint_data = torch.load(top_checkpoint, map_location=device, weights_only=False)
    top_config = top_checkpoint_data.get('model_config', {})
    
    embedding_size = bottom_config.get('embedding_size', 32)
    insurance_input_size = top_config.get('insurance_input_size', 6)
    top_input_size = embedding_size + insurance_input_size
    
    top_model = TopModel(
        embedding_size=embedding_size,
        insurance_input_size=insurance_input_size,
        hidden_sizes=top_config.get('top_hidden_sizes', [64, 32]),
        dropout_rate=top_config.get('dropout_rate', 0.2),
        activation=top_config.get('activation', 'relu'),
    )
    top_model.load_state_dict(top_checkpoint_data['model_state_dict'])
    top_model.to(device)
    top_model.eval()
    
    return {
        'bottom_model': bottom_model,
        'top_model': top_model,
        'bottom_config': bottom_config,
        'top_config': top_config,
        'bottom_metrics': bottom_checkpoint_data.get('final_metrics', {}),
        'top_metrics': top_checkpoint_data.get('final_metrics', {}),
        'history': bottom_checkpoint_data.get('history', {}),
    }


def main():
    parser = argparse.ArgumentParser(description="Load trained Vertical FL models")
    parser.add_argument('--bottom-checkpoint', type=str, 
                       default='results/models/bank_bottom_model.pth',
                       help='Path to Bottom Model checkpoint')
    parser.add_argument('--top-checkpoint', type=str,
                       default='results/models/insurer_top_model.pth',
                       help='Path to Top Model checkpoint')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device')
    
    args = parser.parse_args()
    
    # Load models
    print(f"Loading Bottom Model from {args.bottom_checkpoint}...")
    print(f"Loading Top Model from {args.top_checkpoint}...")
    
    result = load_vertical_fl_models(
        args.bottom_checkpoint,
        args.top_checkpoint,
        device=args.device
    )
    
    print("\nModels loaded successfully!")
    print(f"\nBottom Model config: {result['bottom_config']}")
    print(f"Top Model config: {result['top_config']}")
    
    if result.get('bottom_metrics'):
        print("\nBottom Model metrics:")
        for metric, value in result['bottom_metrics'].items():
            print(f"  {metric}: {value:.4f}")
    
    if result.get('top_metrics'):
        print("\nTop Model metrics:")
        for metric, value in result['top_metrics'].items():
            print(f"  {metric}: {value:.4f}")
    
    print("\nModels ready for inference!")
    print("Use: python scripts/inference.py --bottom-checkpoint <path> --top-checkpoint <path>")


if __name__ == '__main__':
    main()
