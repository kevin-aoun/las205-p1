import os
import pandas as pd
import streamlit as st
import joblib
from datetime import datetime
from typing import Optional
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from weight.prediction.utils import generate_training_report
from weight.logs import logger

def train_and_save_model(df: pd.DataFrame, save_model: bool = True
                        ) -> Optional[LinearRegression]:
    """
    Train a linear regression model to predict Weight from Height and Gender,
    and optionally save the model to disk. Also logs training performance.
    """
    try:
        config = st.session_state.config

        # Ensure categorical Gender is converted to numeric (e.g., 0/1)
        df['Gender'] = df['Gender'].astype('category').cat.codes

        # Prepare data
        X = df[['Height', 'Gender']].values
        y = df['Weight'].values

        random_state = config.get('model', {}).get('random_state', 42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )

        model = LinearRegression()
        logger.info("Starting linear regression model training")

        model.fit(X_train, y_train)
        logger.info("Model training complete")

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        logger.info(f"Mean Squared Error: {mse:.4f}")
        logger.info(f"R² Score: {r2:.4f}")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report = generate_training_report(
            model, X_train, X_test, y_train, y_test, timestamp
        )
        # Save report in session state
        st.session_state['training_report'] = {
            'mean_squared_error': mse,
            'r2_score': r2,
            'report': report
        }

        if save_model:
            models_dir = config['model']['models_dir']
            os.makedirs(models_dir, exist_ok=True)

            model_path = os.path.join(models_dir, f'weight_predictor_model_{timestamp}.joblib')
            joblib.dump(model, model_path)
            logger.info(f"Linear regression model saved to: {model_path}")
            assert os.path.exists(model_path), f"Model file not found at {model_path}"

        return model

    except Exception as e:
        logger.error(f"Error training model {type(model).__name__}: {str(e)}", exc_info=True)
        return None
