"""
Linear Regression prediction module for weight prediction.
"""
import pandas as pd
import logging
import joblib
from typing import Dict, Any, Optional

from sklearn.linear_model import LinearRegression

from .train_weight import train_and_save_model  # Update path if needed
from music_new.core import check_model_files  # Should check model file path only

# Setup module logger
logger = logging.getLogger(__name__)

def predict_using_ml(data: Optional[pd.DataFrame], Gender: str, Height: float,
                     trained_model: Optional[LinearRegression] = None) -> Dict[str, Any]:
    """
    Predict Weight using a trained linear regression model.

    Args:
        data (pd.DataFrame, optional): Data for training if model is not provided
        Gender (str): Gender label ('Male', 'Female', etc.)
        Height (float): Height in cm
        trained_model (LinearRegression, optional): Pre-trained linear regression model

    Returns:
        Dict[str, Any]: Dictionary with predicted weight value
    """
    try:
        if trained_model is None:
            logger.info("No model provided, attempting to load or train...")

            # Try to load from disk
            model_path, _ = check_model_files(return_full_paths=True)

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
        logger.info(f"Preparing prediction for input: Gender={Gender}, Height={Height}")

        # Convert Gender to numeric code
        user_gender_code = pd.Series([Gender]).astype('category').cat.codes[0]
        user_data = pd.DataFrame([[Height, user_gender_code]], columns=['Height', 'Gender'])

        logger.info("Applying model to user data")
        predicted_weight = trained_model.predict(user_data)[0]

        logger.info(f"Predicted Weight: {predicted_weight:.2f}")

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
