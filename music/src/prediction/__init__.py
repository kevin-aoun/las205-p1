"""
Prediction module that handles both percentage-based and ML-based predictions.
"""

from .percentage import predict_using_percentage
from .random_forest import predict_using_ml
from .train_rf import train_and_save_model

__all__ = [
    'predict_using_percentage',
    'predict_using_ml',
    'train_and_save_model'
]