import streamlit as st
import pandas as pd
import os
from utils import config, logger
from prediction import predict_using_percentage, predict_using_ml, train_and_save_model

def save_uploaded_file(uploaded_file):
    """
    Save an uploaded file to disk

    Args:
        uploaded_file: StreamlitUploadedFile object

    Returns:
        str: Path where file was saved
    """
    if not config['data']['save_uploads']:
        return None

    uploads_dir = config['data']['uploads_dir']
    os.makedirs(uploads_dir, exist_ok=True)

    # Save file
    file_path = os.path.join(uploads_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    logger.info(f"Saved uploaded file to {file_path}")
    return file_path


def display_model_status():
    """Display information about saved models in the sidebar"""
    st.sidebar.header("Model Status")

    models_dir = config['model']['models_dir']
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.startswith('music_preferences_model_')]
        if model_files:
            latest_model = sorted(model_files)[-1]
            st.sidebar.success(f"Found saved model: {latest_model}")
        else:
            st.sidebar.warning("No saved models found")
    else:
        st.sidebar.warning(f"Models directory ({models_dir}) not found")

    st.sidebar.header("Configuration")
    st.sidebar.write("Model Settings:")
    st.sidebar.write(f"- Save models: {config['model']['save_model']}")
    st.sidebar.write(f"- Use saved models: {config['model']['use_saved_model']}")


def main():
    if 'model_trained' not in st.session_state:
        st.session_state['model_trained'] = False

    st.title(config['app']['title'])
    st.write("Upload your CSV file with music preference data and get predictions based on age and gender.")

    display_model_status()

    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'], key='file_uploader')

    # Only process the file if it's newly uploaded (not already processed)
    if uploaded_file is not None and not st.session_state.get('file_processed'):
        try:
            df = pd.read_csv(uploaded_file)
            logger.info(f"File uploaded: {uploaded_file.name}, {df.shape[0]} rows, {df.shape[1]} columns")

            if config['data']['save_uploads']:
                save_path = save_uploaded_file(uploaded_file)
                if save_path:
                    st.success(f"File saved successfully")

            st.session_state['data'] = df
            st.session_state['min_age'] = df['age'].min()
            st.session_state['max_age'] = df['age'].max()

            model, le = train_and_save_model(df, save_model = config['model']['save_model'])
            st.session_state['trained_model'] = model
            st.session_state['label_encoder'] = le
            st.session_state['model_trained'] = True

            if config['model']['save_model']:
                st.info("Model trained and saved successfully")
            else:
                st.info("Model trained successfully (not saved to disk)")

            st.session_state['file_processed'] = True

            st.write("Preview of the data with age groups:")
            st.dataframe(df.head())

        except Exception as e:
            logger.error(f"Error processing uploaded file: {str(e)}", exc_info=True)
            st.error(f"Error processing file: {str(e)}")

    if uploaded_file is None:
        st.session_state['file_processed'] = False

    st.subheader("Enter Your Information")

    gender = st.radio("Select your gender:",
                      options=["Male", "Female"],
                      index=0)
    gender_value = 1 if gender == "Male" else 0

    age_min, age_max = config['app']['age_range']
    default_age = config['app']['default_age']
    age = st.slider("Select your age:", age_min, age_max, default_age)

    if st.button("Predict Music Preference"):
        if 'data' not in st.session_state:
            st.warning("Please upload a CSV file first!")
        else:
            data = st.session_state['data']

            percentage_result = predict_using_percentage(data, age, gender_value)

            if st.session_state.get('model_trained', False):
                ml_result = predict_using_ml(
                    None,  # No need to pass data since we're using the pre-trained model
                    age,
                    gender_value,
                    st.session_state['trained_model'],
                    st.session_state['label_encoder']
                )
            else:
                ml_result = {"genre": "Model not trained", "confidence": 0}

            st.subheader("Prediction Results")

            col1, col2 = st.columns(2)

            with col1:
                st.info("Based on Percentage Model")
                st.write(f"Predicted Genre: {percentage_result['genre']}")
                st.write(f"Confidence: {percentage_result['confidence']:.2f}%")

            with col2:
                st.info("Based on Machine Learning Model")
                st.write(f"Predicted Genre: {ml_result['genre']}")
                st.write(f"Confidence: {ml_result['confidence']:.2f}%")

            logger.info(f"Predictions displayed for age={age}, gender={gender_value}")


if __name__ == "__main__":
    logger.info("Application started")
    main()