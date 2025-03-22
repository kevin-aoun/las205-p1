from typing import Optional
import pandas as pd
import os
import joblib
import logging
from sklearn.ensemble import RandomForestRegressor
from .train_rf_weight import train_and_save_model

logger = logging.getLogger(__name__)

def predict_using_ml(
        height: float,
        gender: int,
        trained_model: Optional[RandomForestRegressor] = None):
    """
    Predict weight using a machine learning model.

    Args:
        height (float): User's height in cm
        gender (int): 1 for male, 0 for female
        trained_model (RandomForestRegressor, optional): A pre-trained model

    Returns:
        dict: Predicted weight and confidence
    """
    try:
        if trained_model is None:
            model_files = [f for f in os.listdir('models') if f.startswith('weight_predictor_model_')]
            if model_files:
                latest_model = sorted(model_files)[-1]
                trained_model = joblib.load(os.path.join('models', latest_model))
                logger.info(f"Loaded model: {latest_model}")
            else:
                return {"weight": "Model not found", "confidence": 0}

        input_df = pd.DataFrame([[gender, height]], columns=['Gender','Height'])
        prediction = trained_model.predict(input_df)[0]
        logger.info(f"Predicted weight: {prediction}kg")

        return {"Weight": round(prediction, 2), "confidence": 95.0}

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return {"Weight": f"Error: {str(e)}", "confidence": 0}
