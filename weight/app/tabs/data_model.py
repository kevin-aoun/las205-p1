import streamlit as st
import pandas as pd

from weight.core import logger, save_uploaded_file
from weight.prediction import train_and_save_model


def render_data_model_tab(model_exists, latest_model):
    """
    Render the Data & Model tab

    Args:
        model_exists (bool): Whether a usable model exists
        latest_model (str): Filename of the latest model
    """
    config = st.session_state.config

    # Initialize button action state variables if not present
    if 'use_existing_clicked' not in st.session_state:
        st.session_state['use_existing_clicked'] = False
    if 'train_new_clicked' not in st.session_state:
        st.session_state['train_new_clicked'] = False

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Data Upload")
        # Use a unique key for the file uploader
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'], key='data_file_uploader')

        if uploaded_file is None:
            st.error("CSV upload is mandatory. Please upload a file to continue.")

    with col2:
        st.subheader("Model Status")
        if model_exists:
            st.success(f"Using saved model: {latest_model}")

            # Button callback
            def on_load_model_click():
                st.session_state['model_trained'] = True

            if st.button("Load saved model", key="load_model_button", on_click=on_load_model_click):
                # This code won't execute immediately - the callback will be called first
                pass

            # Show success message if model was loaded
            if st.session_state.get('model_trained', False):
                st.success("Saved model loaded successfully")

    # Process uploaded file if it exists
    if uploaded_file is not None:
        # Check if we need to process this file
        should_process = False

        # Only process if we have no data or it's a different file
        if 'current_data' not in st.session_state or st.session_state['current_data'] is None:
            should_process = True
        elif 'current_filename' not in st.session_state or st.session_state['current_filename'] != uploaded_file.name:
            should_process = True

        if should_process:
            try:
                # Read the CSV file
                df = pd.read_csv(uploaded_file)
                logger.info(f"File uploaded: {uploaded_file.name}, {df.shape[0]} rows, {df.shape[1]} columns")

                # Save the file if enabled in config
                if config['data']['save_uploads']:
                    save_path = save_uploaded_file(uploaded_file)
                    if save_path:
                        st.success(f"File saved successfully")

                # Store data in session state
                st.session_state['current_data'] = df
                st.session_state['current_filename'] = uploaded_file.name

                if 'Height' in df.columns:
                    st.session_state['min_height'] = df['Height'].min()
                    st.session_state['max_height'] = df['Height'].max()

                # Reset processing flags for new data
                st.session_state['file_processed'] = False
                st.session_state['use_existing_clicked'] = False
                st.session_state['train_new_clicked'] = False

            except Exception as e:
                logger.error(f"Error processing uploaded file: {str(e)}", exc_info=True)
                st.error(f"Error processing file: {str(e)}")

    # Only show data preview and model options if we have data
    if 'current_data' in st.session_state and st.session_state['current_data'] is not None:
        # Display data preview
        st.subheader("Data Preview")
        st.dataframe(st.session_state['current_data'].head())

        # Handle model training decision only if not yet processed
        if not st.session_state.get('file_processed', False):
            handle_model_training(model_exists)

        # Check for button actions after rendering
        if st.session_state['use_existing_clicked']:
            st.success("Using existing model with new data")
            # Reset so we don't show the message again on rerender
            st.session_state['use_existing_clicked'] = False

        if st.session_state['train_new_clicked']:
            # Reset flag so we don't retrain
            st.session_state['train_new_clicked'] = False
            # Train model with current data
            train_model(st.session_state['current_data'])

    elif uploaded_file is None:
        # Reset processing flags when no file is uploaded
        st.session_state['file_processed'] = False


def handle_model_training(model_exists):
    """
    Handle the model training options and process

    Args:
        model_exists (bool): Whether a usable model exists
    """
    if model_exists:
        st.subheader("Model Training Options")
        st.info("A saved model exists. You can use it or train a new model with the uploaded data.")

        col1, col2 = st.columns(2)

        # Use callbacks for buttons to avoid UI duplication
        def use_existing_callback():
            st.session_state['file_processed'] = True
            st.session_state['model_trained'] = True
            st.session_state['use_existing_clicked'] = True

        def train_new_callback():
            st.session_state['train_new_clicked'] = True

        with col1:
            st.button("Use existing model", key="use_model_button", on_click=use_existing_callback)

        with col2:
            st.button("Train new model", key="train_new_button", on_click=train_new_callback)

    # Automatically train model if no existing model
    elif not st.session_state.get('file_processed', False):
        st.info("No existing model found. Training a new model...")
        # Set processed flag to avoid retraining
        st.session_state['file_processed'] = True
        train_model(st.session_state['current_data'])


def train_model(df):
    """
    Train a new model with the provided data

    Args:
        df (pd.DataFrame): Data to train the model with
    """
    config = st.session_state.config

    with st.spinner("Training new model..."):
        try:
            model= train_and_save_model(df, save_model=config['model']['save_model'])

            if model is not None:
                st.session_state['trained_model'] = model
            
                st.session_state['model_trained'] = True
                st.session_state['file_processed'] = True
                st.session_state['show_training_report'] = True

                if config['model']['save_model']:
                    st.success("New model trained and saved successfully")
                else:
                    st.success("New model trained successfully (not saved to disk)")

                st.info("View the training report in the 'Training Report' tab")
            else:
                st.error("Model training failed. Check logs for details.")
        except Exception as e:
            st.error(f"Error training model: {str(e)}")
            logger.error(f"Error training model: {str(e)}", exc_info=True)