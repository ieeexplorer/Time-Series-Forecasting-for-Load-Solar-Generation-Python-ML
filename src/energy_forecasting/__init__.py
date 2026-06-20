"""Tools for leakage-safe day-ahead load and solar forecasting."""

from .data import generate_synthetic_data
from .features import build_features

__all__ = ["build_features", "generate_synthetic_data"]
__version__ = "0.2.0"
