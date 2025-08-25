"""
House Price Prediction ML Package
"""

from .data_preprocessing import DataPreprocessor
from .model_training import ModelTrainer
from .model_evaluation import ModelEvaluator
from .prediction import HousePricePredictor

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = [
    'DataPreprocessor',
    'ModelTrainer',
    'ModelEvaluator',
    'HousePricePredictor'
]
