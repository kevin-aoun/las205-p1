from typing import Optional
import pandas as pd
import os
import joblib
import logging

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from .train_rf import train_and_save_model

logger = logging.getLogger(__name__)

def predict_using_ml(
        data: Optional[pd.DataFrame],
        age: int,
        gender: int,
        trained_model: RandomForestClassifier=None,
        label_encoder: LabelEncoder=None):
    """
    Predict music genre preference using machine learning.

    Args:
        data (pd.DataFrame): Dataset containing age, gender, and genre (used only if model is not provided)
        age (int): User's age
        gender (int): User's gender (1 for male, 0 for female)
        trained_model (RandomForestClassifier, optional): Pre-trained model
        label_encoder (LabelEncoder, optional): Fitted label encoder

    Returns:
        dict: Dictionary with predicted genre and confidence
    """

    try:
        if trained_model is None or label_encoder is None:
            if os.path.exists('models'):
                model_files = [f for f in os.listdir('models') if f.startswith('music_preferences_model_')]
                encoder_files = [f for f in os.listdir('models') if f.startswith('label_encoder_')]

                if model_files and encoder_files:
                    latest_model = sorted(model_files)[-1]
                    latest_encoder = sorted(encoder_files)[-1]

                    try:
                        trained_model = joblib.load(os.path.join('models', latest_model))
                        label_encoder = joblib.load(os.path.join('models', latest_encoder))
                        print(f"Loaded model from {latest_model}")
                    except Exception as load_err:
                        print(f"Error loading saved model: {load_err}")

            if (trained_model is None or label_encoder is None) and data is not None:
                trained_model, label_encoder = train_and_save_model(data)
            elif trained_model is None or label_encoder is None:
                return {
                    "genre": "No model available and no data provided",
                    "confidence": 0
                }

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
        return {
            "genre": f"Error in ML prediction: {str(e)}",
            "confidence": 0
        }