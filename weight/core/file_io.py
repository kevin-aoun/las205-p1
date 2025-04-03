import os
import streamlit as st
from music.logs import logger

def get_absolute_path(relative_path):
    """Convert relative path to absolute path based on the music directory"""
    music_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    music_dir = os.path.normpath(music_dir)
    return os.path.join(music_dir, relative_path)


def ensure_directories(config_dict):
    """Ensure all required directories exist and update config with absolute paths"""
    if config_dict['model']['save_model']:
        models_dir = get_absolute_path(config_dict['model']['models_dir'])
        os.makedirs(models_dir, exist_ok=True)
        config_dict['model']['models_dir'] = models_dir

    if config_dict['data']['save_uploads']:
        uploads_dir = get_absolute_path(config_dict['data']['uploads_dir'])
        os.makedirs(uploads_dir, exist_ok=True)
        config_dict['data']['uploads_dir'] = uploads_dir

    if config_dict['logging']['file']:
        log_file = get_absolute_path(config_dict['logging']['file'])
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        config_dict['logging']['file'] = log_file


def save_uploaded_file(uploaded_file):
    """Save uploaded file to the configured uploads directory"""
    config = st.session_state.config

    if not config['data']['save_uploads']:
        return None

    uploads_dir = config['data']['uploads_dir']
    file_path = os.path.join(uploads_dir, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    logger.info(f"Saved uploaded file to {file_path}")
    return file_path


def return_latest_model(return_full_paths=False):
    """Check for saved model files and return latest model or path(s)"""
    config = st.session_state.config
    models_dir = config['model']['models_dir']

    if not os.path.exists(models_dir):
        logger.info("Models directory does not exist")
        return (None, None) if return_full_paths else None

    model_files = [
        f for f in os.listdir(models_dir)
        if f.startswith('weight_predictor_model_') and (f.endswith('.joblib') or f.endswith('.pkl'))
    ]

    if not model_files:
        logger.info("No model files found")
        return (None, None) if return_full_paths else None

    latest_model = sorted(model_files)[-1]
    logger.info(f"Found latest model: {latest_model}")

    if not return_full_paths:
        return latest_model

    encoder_files = [f for f in os.listdir(models_dir) if f.startswith('label_encoder_')]

    if encoder_files:
        latest_encoder = sorted(encoder_files)[-1]
        return os.path.join(models_dir, latest_model), os.path.join(models_dir, latest_encoder)

    logger.warning("Found model file but no encoder file")
    return os.path.join(models_dir, latest_model), None