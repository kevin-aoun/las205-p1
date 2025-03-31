import yaml
import logging
import streamlit as st
import os

from music.logs import setup_logging
from music.core import ensure_directories

logger = logging.getLogger('music_preference_predictor')

def create_default_config():
    """
    Create default configuration dictionary.

    Returns:
        dict: The default configuration dictionary
    """
    default_config = {
        'model': {
            'save_model': True,
            'models_dir': 'models',
            'use_saved_model': True,
            'n_estimators': 100,
            'random_state': 42
        },
        'data': {
            'save_uploads': True,
            'uploads_dir': 'uploads',
            'age_bins': [0, 25, 31, 999],
            'age_labels': ['young', 'mid', 'older']
        },
        'app': {
            'title': 'Music Preference Predictor',
            'default_age': 25,
            'age_range': [0, 100]
        },
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file': 'logs/app.log',
            'console': True
        }
    }

    return default_config

def save_config_to_file(config_dict, config_path="music/config.yaml"):
    """
    Save configuration to file.

    Args:
        config_dict (dict): The configuration dictionary to save
        config_path (str): Path to save the configuration file

    Returns:
        bool: True if saved successfully, False otherwise
    """
    try:
        with open(config_path, 'w') as file:
            yaml.dump(config_dict, file, default_flow_style=False)
        logger.info("Configuration saved successfully")
        return True
    except Exception as e:
        logger.error(f"Error saving configuration: {str(e)}", exc_info=True)
        return False


def load_config_from_file(config_path="music/config.yaml"):
    """
    Load configuration from file or create default if file doesn't exist.

    Args:
        config_path (str): Path to the configuration file

    Returns:
        dict: The configuration dictionary
    """
    try:
        with open(config_path, 'r') as file:
            loaded_config = yaml.safe_load(file)
        logger.info(f"Configuration loaded from {config_path}")
        return loaded_config

    except FileNotFoundError:
        logger.info(f"Configuration file {config_path} not found, creating default")
        default_config = create_default_config()

        save_config_to_file(default_config, config_path)
        return default_config

    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}", exc_info=True)
        default_config = create_default_config()
        return default_config


def update_config(new_config):
    """
    Update the configuration in session state and apply changes.
    Makes sure to preserve relative paths in the saved config file.

    Args:
        new_config (dict): The new configuration dictionary

    Returns:
        bool: True if updated successfully, False otherwise
    """
    # Create a copy of the new config that will be modified to use relative paths for saving
    save_config = {
        'model': {**new_config['model']},
        'data': {**new_config['data']},
        'app': {**new_config['app']},
        'logging': {**new_config['logging']}
    }

    # Convert absolute paths back to relative paths for saving to file
    # Models directory
    if 'models_dir' in save_config['model'] and os.path.isabs(save_config['model']['models_dir']):
        save_config['model']['models_dir'] = 'models'

    # Uploads directory
    if 'uploads_dir' in save_config['data'] and os.path.isabs(save_config['data']['uploads_dir']):
        save_config['data']['uploads_dir'] = 'uploads'

    # Log file
    if 'file' in save_config['logging'] and os.path.isabs(save_config['logging']['file']):
        save_config['logging']['file'] = 'logs/app.log'

    # Save the config with relative paths to file
    if save_config_to_file(save_config):
        # Update in session state with the absolute paths for runtime use
        st.session_state.config = new_config
        ensure_directories(new_config)
        setup_logging(new_config)

        return True

    return False

def init_config():
    """
    Initialize configuration in session state if not already present.
    This should be called at the start of the app.
    """
    if 'config' not in st.session_state:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)

        config = load_config_from_file()
        ensure_directories(config)
        setup_logging(config)

        st.session_state.config = config