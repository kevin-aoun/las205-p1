from typing import Optional
import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
import joblib
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def train_and_save_model(df: pd.DataFrame, save_model: bool = True) -> Optional[RandomForestRegressor]:
    """
    Train a regression model to predict weight from height and gender.

    Args:
        df (pd.DataFrame): DataFrame with 'height', 'gender', and 'weight'
        save_model (bool): Whether to save the model to disk

    Returns:
        RandomForestRegressor or None
    """
    try:
        X = df[['Gender', 'Height']]
        y = df['Weight']

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        logger.info("Model training complete")

        if save_model:
            os.makedirs('models', exist_ok=True)
            models_path = os.path.abspath('models')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(models_path, f'weight_predictor_model_{timestamp}.joblib')
            joblib.dump(model, model_path)
            logger.info(f"Model saved to: {model_path}")
            assert os.path.exists(model_path), f"Model file not found at {model_path}"

        return model
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        return None
