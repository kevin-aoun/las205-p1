"""
Random Forest prediction module for music preferences.
"""
import os
import pandas as pd
import streamlit as st
import logging
import joblib
from typing import Dict, Any, Optional

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from .train_rf import train_and_save_model

# Setup module logger
logger = logging.getLogger(__name__)


def check_saved_model_files():
    """
    Check if saved model files exist and return the latest ones.

    Returns:
        tuple: (model_path, encoder_path) or (None, None) if not found
    """
    config = st.session_state.config
    models_dir = config['model']['models_dir']

    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.startswith('music_preferences_model_')]
        encoder_files = [f for f in os.listdir(models_dir) if f.startswith('label_encoder_')]

        if model_files and encoder_files:
            latest_model = sorted(model_files)[-1]
            latest_encoder = sorted(encoder_files)[-1]
            return os.path.join(models_dir, latest_model), os.path.join(models_dir, latest_encoder)

    return None, None


def predict_using_ml(data: Optional[pd.DataFrame], age: int, gender: int,
                     trained_model: RandomForestClassifier = None,
                     label_encoder: LabelEncoder = None) -> Dict[str, Any]:
    """
    Predict music genre preference using machine learning.

    Args:
        data (pd.DataFrame, optional): Dataset for training if model not provided
        age (int): User's age
        gender (int): User's gender (1 for male, 0 for female)
        trained_model (RandomForestClassifier, optional): Pre-trained model
        label_encoder (LabelEncoder, optional): Fitted label encoder

    Returns:
        Dict[str, Any]: Dictionary with predicted genre and confidence
    """
    try:
        # If model or encoder not provided, try to load them
        if trained_model is None or label_encoder is None:
            logger.info("No model provided, attempting to load or train...")

            # Try to load from disk
            model_path, encoder_path = check_saved_model_files()

            if model_path and encoder_path:
                logger.info(f"Loading model from {model_path} and encoder from {encoder_path}")
                try:
                    trained_model = joblib.load(model_path)
                    label_encoder = joblib.load(encoder_path)
                except Exception as load_err:
                    logger.error(f"Error loading saved model: {load_err}", exc_info=True)
                    trained_model, label_encoder = None, None

            # If still no model and we have data, train a new one
            if (trained_model is None or label_encoder is None) and data is not None:
                logger.info("Training new model with provided data")
                result = train_and_save_model(data)
                if result:
                    trained_model, label_encoder = result
            elif trained_model is None or label_encoder is None:
                logger.warning("No model available and no data provided")
                return {
                    "genre": "No model available and no data provided",
                    "confidence": 0
                }

        # Make prediction
        logger.info(f"Preparing prediction for input: age={age}, gender={gender}")
        user_data = pd.DataFrame([[age, gender]], columns=['age', 'gender'])

        logger.info("Applying model to user data")
        prediction_encoded = trained_model.predict(user_data)[0]
        prediction_proba = trained_model.predict_proba(user_data)[0]

        logger.info(f"Raw prediction (encoded): {prediction_encoded}")
        predicted_genre = label_encoder.inverse_transform([prediction_encoded])[0]

        # Get all probabilities for debugging
        class_probabilities = {
            label_encoder.inverse_transform([i])[0]: f"{prediction_proba[i] * 100:.2f}%"
            for i in range(len(prediction_proba))
        }
        logger.info(f"Class probabilities: {class_probabilities}")

        confidence = prediction_proba[prediction_encoded] * 100
        logger.info(f"Final prediction: {predicted_genre} with {confidence:.2f}% confidence")

        return {
            "genre": predicted_genre,
            "confidence": confidence
        }
    except Exception as e:
        logger.error(f"Error in ML prediction: {str(e)}", exc_info=True)
        return {
            "genre": f"Error in ML prediction: {str(e)}",
            "confidence": 0
        }