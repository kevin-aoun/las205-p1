"""
Prediction module that handles both percentage-based and ML-based predictions.
"""

from .percentage_weight import predict_using_percentage
from .linear_regression import predict_using_ml
from .train_weight import train_and_save_model

__all__ = [
    'predict_using_percentage',
    'predict_using_ml',
    'train_and_save_model'
]