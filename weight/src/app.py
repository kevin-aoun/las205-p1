import streamlit as st
import pandas as pd
import os
from utils import config, logger
from prediction.train_rf_weight import  train_and_save_model
from prediction.percentage_weight import predict_weight_percentage
from prediction.random_forest_weight import  predict_using_ml  




def save_uploaded_file(uploaded_file):
    if not config['data']['save_uploads']:
        return None

    uploads_dir = config['data']['uploads_dir']
    os.makedirs(uploads_dir, exist_ok=True)

    file_path = os.path.join(uploads_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    logger.info(f"Saved uploaded file to {file_path}")
    return file_path

def display_model_status():
    st.sidebar.header("Model Status")
    models_dir = config['model']['models_dir']
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.startswith('weight_predictor_model_') or f.startswith('L')]
        if model_files:
            latest_model = sorted(model_files)[-1]
            st.sidebar.success(f"Found saved model: {latest_model}")
        else:
            st.sidebar.warning("No saved models found")
    else:
        st.sidebar.warning(f"Models directory ({models_dir}) not found")

    st.sidebar.header("Configuration")
    st.sidebar.write(f"- Save models: {config['model']['save_model']}")
    st.sidebar.write(f"- Use saved models: {config['model']['use_saved_model']}")

def main():
    if 'model_trained' not in st.session_state:
        st.session_state['model_trained'] = False

    st.title(config['app']['title'])
    st.write("Upload a CSV with height, gender, and weight data to train a model that predicts weight.")

    display_model_status()

    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'], key='file_uploader')

    if uploaded_file is not None and not st.session_state.get('file_processed'):
        try:
            df = pd.read_csv(uploaded_file)
            logger.info(f"File uploaded: {uploaded_file.name}, {df.shape[0]} rows")

            if config['data']['save_uploads']:
                save_path = save_uploaded_file(uploaded_file)
                if save_path:
                    st.success(f"File saved successfully")

            st.session_state['data'] = df
            st.session_state['min_height'] = df['Height'].min()
            st.session_state['max_height'] = df['Height'].max()

            model = train_and_save_model(df, save_model=config['model']['save_model'])
            st.session_state['trained_model'] = model
            st.session_state['model_trained'] = True

            if config['model']['save_model']:
                st.info("Model trained and saved successfully")
            else:
                st.info("Model trained (not saved)")

            st.session_state['file_processed'] = True
            st.dataframe(df.head())

        except Exception as e:
            logger.error(f"Error processing file: {str(e)}", exc_info=True)
            st.error(f"Error: {str(e)}")

    if uploaded_file is None:
        st.session_state['file_processed'] = False

    st.subheader("Enter Your Information")

    gender = st.radio("Select your gender:", options=["Male", "Female"], index=0)
    gender_value = 1 if gender == "Male" else 0

    height_min, height_max = config['app']['height_range']
    default_height = config['app']['default_height']
    height = st.slider("Select your height (cm):", height_min, height_max, default_height)

    if st.button("Predict Weight"):
        if 'data' not in st.session_state:
            st.warning("Please upload a CSV file first!")
        else:
            data = st.session_state['data']

            percentage_result = predict_weight_percentage(data, gender_value, height)

            if st.session_state.get('model_trained', False):
                ml_result = predict_using_ml(gender_value, height, st.session_state['trained_model'])
            else:
                ml_result = {"Weight": "N/A", "confidence": 0}

            st.subheader("Prediction Results")

            col1, col2 = st.columns(2)

            with col1:
                st.info("Based on Percentage Method")
                st.write(f"Predicted Weight: {percentage_result['weight']} kg")
                st.write(f"Confidence: {percentage_result['confidence']:.2f}%")

            with col2:
                st.info("Based on ML Model")
                st.write(f"Predicted Weight: {ml_result['Weight']} kg")
                st.write(f"Confidence: {ml_result['confidence']:.2f}%")

            logger.info(f"Predictions displayed for height={height}, gender={gender_value}")

if __name__ == "__main__":
    logger.info("Weight predictor app started")
    main()
