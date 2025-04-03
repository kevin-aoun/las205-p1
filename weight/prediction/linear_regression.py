"""
Linear Regression prediction module for weight prediction.
"""
import pandas as pd
import joblib
from typing import Dict, Any, Optional
from sklearn.linear_model import LinearRegression

from .train_weight import train_and_save_model  # Update path if needed
from weight.core import return_latest_model
from weight.logs import logger

def predict_using_ml(data: Optional[pd.DataFrame], gender: int, Height: float,
                     trained_model: Optional[LinearRegression] = None) -> Dict[str, Any]:
    """
    Predict Weight using a trained linear regression model.

    Args:
        data (pd.DataFrame, optional): Data for training if model is not provided
        gender (int): gender label ('Male' = 1, 'Female' = 0)
        Height (float): Height in cm
        trained_model (LinearRegression, optional): Pre-trained linear regression model

    Returns:
        Dict[str, Any]: Dictionary with predicted weight value
    """
    try:
        if trained_model is None:
            logger.info("No model provided, attempting to load or train...")

            # Try to load from disk
            model_path, _ = return_latest_model(return_full_paths=True)

            if model_path:
                logger.info(f"Loading model from {model_path}")
                try:
                    trained_model = joblib.load(model_path)
                except Exception as load_err:
                    logger.error(f"Error loading saved model: {load_err}", exc_info=True)
                    trained_model = None

            # Train new model if needed
            if trained_model is None and data is not None:
                logger.info("Training new model with provided data")
                trained_model = train_and_save_model(data)

            elif trained_model is None:
                logger.warning("No model available and no data provided")
                return {
                    "Weight": "No model available and no data provided",
                    "confidence": 0
                }

        # Prepare user input
        logger.info(f"Preparing prediction for input: gender={gender}, Height={Height}")

        user_data = pd.DataFrame([[Height, gender]], columns=['Height', 'gender'])

        logger.info("Applying model to user data")
        predicted_weight = trained_model.predict(user_data)[0]
        
        logger.info(f"Predicted Weight using ML: {predicted_weight:.2f}")

        return {
            "Weight": round(predicted_weight, 2),
            "confidence": None  # Optional: could compute confidence intervals if needed
        }

    except Exception as e:
        logger.error(f"Error in ML prediction: {str(e)}", exc_info=True)
        return {
            "Weight": f"Error in ML prediction: {str(e)}",
            "confidence": 0
        }
