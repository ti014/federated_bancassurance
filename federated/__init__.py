"""Federated Learning components for Vertical FL."""

from .vertical_client import BankBottomClient, InsuranceTopClient
from .vertical_server import VerticalFLServer

__all__ = [
    "BankBottomClient",
    "InsuranceTopClient",
    "VerticalFLServer",
]
