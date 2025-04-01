"""
Model training module for the music preference predictor.
"""
import os
import pandas as pd

import streamlit as st
import joblib
from datetime import datetime
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from music.prediction.utils import generate_training_report
from music.logs import logger

def train_and_save_model(df: pd.DataFrame, save_model: bool = True
                        ) -> Optional[Tuple[RandomForestClassifier, LabelEncoder]]:
    """
    Train a model from a DataFrame and optionally save it to disk.
    Also generates and saves a comprehensive training report.

    Args:
        df (pd.DataFrame): DataFrame with music preference data
        save_model (bool): Flag to save the model to disk

    Returns:
        Optional[Tuple[RandomForestClassifier, LabelEncoder]]: Trained model and label encoder
    """
    try:
        config = st.session_state.config

        # Prepare data
        X = df[['age', 'gender']].values
        y = df['genre'].values

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        logger.info(f"Labels encoded. Unique genres: {list(le.classes_)}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=config['model']['random_state']
        )

        model = RandomForestClassifier(
            n_estimators=config['model']['n_estimators'],
            random_state=config['model']['random_state']
        )

        logger.info("Starting model training")

        model.fit(X_train, y_train) # training step
        logger.info("Model training complete")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        report = generate_training_report(
            model, X_train, X_test, y_train, y_test, le, timestamp
        )

        logger.info(f"Model accuracy: {report['accuracy']:.4f}")
        logger.info(f"F1 score (weighted): {report['f1_weighted']:.4f}")

        if save_model:
            models_dir = config['model']['models_dir']
            os.makedirs(models_dir, exist_ok=True)

            model_path = os.path.join(models_dir, f'music_preferences_model_{timestamp}.joblib')
            encoder_path = os.path.join(models_dir, f'label_encoder_{timestamp}.joblib')

            joblib.dump(model, model_path)
            joblib.dump(le, encoder_path)

            logger.info(f"Model saved to: {model_path}")
            logger.info(f"Label encoder saved to: {encoder_path}")

            assert os.path.exists(model_path), f"Model file not found at {model_path}"
            assert os.path.exists(encoder_path), f"Encoder file not found at {encoder_path}"

            logger.info("File verification successful")

        # Save report info in session state for easy access
        if 'training_report' not in st.session_state:
            st.session_state['training_report'] = {}

        st.session_state['training_report'] = report

        return model, le
    except Exception as e:
        logger.error(f"Error in training model: {str(e)}", exc_info=True)
        import traceback
        traceback.print_exc()
        return None