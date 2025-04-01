"""
Random Forest prediction module for music preferences.
"""
import pandas as pd
import joblib
import os
import streamlit as st
from typing import Dict, Optional

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from .train_rf import train_and_save_model
from music.core import return_latest_model
from music.logs import logger

def predict_using_ml(
        data: Optional[pd.DataFrame],
        age: int,
        gender_value: int,
        trained_model: Optional[RandomForestClassifier]=None,
        label_encoder: Optional[LabelEncoder]=None,
        model_filename: Optional[str]=None):
    """
    Predict music genre preference using machine learning.

    Args:
        data (pd.DataFrame, optional): Dataset to use if new model needs to be trained
        age (int): User's age
        gender_value (int): User's gender (1 for male, 0 for female)
        trained_model: Pre-loaded model (optional)
        label_encoder: Pre-loaded label encoder (optional)
        model_filename (str, optional): Specific model filename to load from disk

    Returns:
        Dict: Dictionary with prediction results
    """
    try:
        # If model and encoder are provided directly, use them
        if trained_model is not None and label_encoder is not None:
            logger.info("Using provided in-memory model for prediction")
        else:
            # Load a model from disk
            if model_filename:
                # Load the specific model requested
                logger.info(f"Loading specific model from disk: {model_filename}")
                models_dir = st.session_state.config['model']['models_dir']
                model_path = os.path.join(models_dir, model_filename)

                # Find matching encoder with same timestamp
                import re
                timestamp_match = re.search(r'model_(\d+_\d+)', model_filename)
                if timestamp_match:
                    timestamp = timestamp_match.group(1)
                    encoder_filename = f"label_encoder_{timestamp}.joblib"
                    encoder_path = os.path.join(models_dir, encoder_filename)

                    if os.path.exists(model_path) and os.path.exists(encoder_path):
                        try:
                            trained_model = joblib.load(model_path)
                            label_encoder = joblib.load(encoder_path)
                            logger.info(f"Successfully loaded model: {model_filename}")
                        except Exception as e:
                            logger.error(f"Error loading model files: {str(e)}")
                            raise
                    else:
                        logger.warning(f"Model or encoder file not found for {model_filename}")
            else:
                # No specific model requested, load the latest one
                logger.info("No specific model requested, loading latest model")
                model_path, encoder_path = return_latest_model()

                if model_path and encoder_path:
                    try:
                        trained_model = joblib.load(model_path)
                        label_encoder = joblib.load(encoder_path)
                        logger.info(f"Loaded latest model from: {model_path}")
                    except Exception as e:
                        logger.error(f"Error loading latest model: {str(e)}")
                        raise

            # If no model loaded or found, train a new one if data is provided
            if (trained_model is None or label_encoder is None) and data is not None:
                logger.info("No model loaded, training new model with provided data")
                trained_model, label_encoder = train_and_save_model(data)

        # Check if we have a valid model and encoder at this point
        if trained_model is None or label_encoder is None:
            return {
                "genre": "No model available",
                "confidence": 0,
                "probabilities": {}
            }

        # Make prediction with the model
        logger.info(f"Making prediction for age={age}, gender={gender_value}")
        user_data = pd.DataFrame([[age, gender_value]], columns=['age', 'gender'])

        prediction_encoded = trained_model.predict(user_data)[0]
        prediction_proba = trained_model.predict_proba(user_data)[0]

        predicted_genre = label_encoder.inverse_transform([prediction_encoded])[0]

        # Get all class probabilities
        probabilities = {}
        for i, prob in enumerate(prediction_proba):
            genre = label_encoder.inverse_transform([i])[0]
            probabilities[genre] = prob

        confidence = prediction_proba[prediction_encoded] * 100

        # Return detailed prediction result
        return {
            "genre": predicted_genre,
            "confidence": confidence,
            "probabilities": probabilities
        }
    except Exception as e:
        logger.error(f"Error in ML prediction: {str(e)}", exc_info=True)
        return {
            "genre": f"Error: {str(e)}",
            "confidence": 0,
            "probabilities": {}
        }