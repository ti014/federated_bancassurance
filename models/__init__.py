"""Model definitions for Vertical FL."""

from .splitnn import BottomModel, TopModel, create_splitnn_models

__all__ = [
    "BottomModel",
    "TopModel",
    "create_splitnn_models",
]
