"""Configuration loading utilities."""

import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    """Model configuration."""
    input_size: int
    hidden_sizes: list = field(default_factory=lambda: [64, 32])
    dropout_rate: float = 0.3
    activation: str = 'relu'
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 10


@dataclass
class FLConfig:
    """Federated Learning configuration."""
    n_clients: int = 5
    n_rounds: int = 10
    local_epochs: int = 5
    fraction_fit: float = 1.0  # Fraction of clients to train per round
    fraction_evaluate: float = 1.0  # Fraction of clients to evaluate per round
    min_fit_clients: int = 2
    min_evaluate_clients: int = 2
    min_available_clients: int = 2
    aggregation_strategy: str = 'fedavg'  # 'fedavg', 'fedprox'
    mu: float = 0.01  # For FedProx


@dataclass
class DataConfig:
    """Data configuration."""
    n_samples: int = 10000
    test_size: float = 0.2
    split_method: str = 'non_iid'  # 'iid', 'non_iid', 'by_bank'
    non_iid_alpha: float = 0.5  # For Non-IID distribution
    random_state: int = 42


@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig
    fl: FLConfig
    data: DataConfig
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create Config from dictionary."""
        # Get model config với default values
        model_dict = config_dict.get('model', {})
        if 'input_size' not in model_dict:
            model_dict['input_size'] = 9  # Default, sẽ được update sau
        
        return cls(
            model=ModelConfig(**model_dict),
            fl=FLConfig(**config_dict.get('fl', {})),
            data=DataConfig(**config_dict.get('data', {})),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Config to dictionary."""
        return {
            'model': {
                'input_size': self.model.input_size,
                'hidden_sizes': self.model.hidden_sizes,
                'dropout_rate': self.model.dropout_rate,
                'activation': self.model.activation,
                'learning_rate': self.model.learning_rate,
                'batch_size': self.model.batch_size,
                'epochs': self.model.epochs,
            },
            'fl': {
                'n_clients': self.fl.n_clients,
                'n_rounds': self.fl.n_rounds,
                'local_epochs': self.fl.local_epochs,
                'fraction_fit': self.fl.fraction_fit,
                'fraction_evaluate': self.fl.fraction_evaluate,
                'min_fit_clients': self.fl.min_fit_clients,
                'min_evaluate_clients': self.fl.min_evaluate_clients,
                'min_available_clients': self.fl.min_available_clients,
                'aggregation_strategy': self.fl.aggregation_strategy,
                'mu': self.fl.mu,
            },
            'data': {
                'n_samples': self.data.n_samples,
                'test_size': self.data.test_size,
                'split_method': self.data.split_method,
                'non_iid_alpha': self.data.non_iid_alpha,
                'random_state': self.data.random_state,
            },
        }


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file.
    
    Args:
        config_path: Đường dẫn đến YAML config file
    
    Returns:
        Config object
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    return Config.from_dict(config_dict)


def save_config(config: Config, config_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Config object
        config_path: Đường dẫn để save
    """
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, allow_unicode=True)

