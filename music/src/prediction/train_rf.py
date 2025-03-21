from typing import Tuple, Optional

import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def train_and_save_model(df: pd.DataFrame, save_model: bool = True
    ) -> Optional[Tuple[RandomForestClassifier, LabelEncoder]]:
    """
    Train a model from a CSV file and save it to disk.

    Args:
        df (pd.DataFrame): DataFrame with music preference data
        save_model (bool): Flag to save the model to disk
    """
    try:

        # Prepare data
        X = df[['age', 'gender']]
        y = df['genre']

        # Encode target variable
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        logger.info(f"Labels encoded. Unique genres: {list(le.classes_)}")

        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y_encoded)
        logger.info("Model training complete")

        if save_model:

            os.makedirs('models', exist_ok=True)
            models_path = os.path.abspath('models')
            logger.info(f"Models will be saved to: {models_path}")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            model_path = os.path.join('models', f'music_preferences_model_{timestamp}.joblib')
            encoder_path = os.path.join('models', f'label_encoder_{timestamp}.joblib')

            joblib.dump(model, model_path)
            joblib.dump(le, encoder_path)

            logger.info(f"Model saved to: {model_path}")
            logger.info(f"Label encoder saved to: {encoder_path}")

            # Verify files exist
            assert os.path.exists(model_path), f"Model file not found at {model_path}"
            assert os.path.exists(encoder_path), f"Encoder file not found at {encoder_path}"

            logger.info("File verification successful")

        return model, le
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None