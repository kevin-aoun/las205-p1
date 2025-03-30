import os
import yaml
import logging
import streamlit as st
from logging.handlers import RotatingFileHandler

# Create a logger instance
logger = logging.getLogger('music_preference_predictor')


def ensure_directories(config_dict):
    """
    Ensure all required directories exist based on configuration.

    Args:
        config_dict (dict): Configuration dictionary
    """
    # Create model directory
    if config_dict['model']['save_model']:
        os.makedirs(config_dict['model']['models_dir'], exist_ok=True)

    # Create uploads directory
    if config_dict['data']['save_uploads']:
        os.makedirs(config_dict['data']['uploads_dir'], exist_ok=True)

    # Create logs directory
    log_dir = os.path.dirname(config_dict['logging']['file'])
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)


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


def setup_logging(config_dict):
    """
    Setup logging configuration.

    Args:
        config_dict (dict): Configuration dictionary
    """
    # Clear existing handlers
    if logger.handlers:
        logger.handlers.clear()

    # Set the log level from config
    log_level = getattr(logging, config_dict['logging']['level'])
    logger.setLevel(log_level)

    # Set the formatter from config
    formatter = logging.Formatter(config_dict['logging']['format'])

    # Add file handler
    if config_dict['logging']['file']:
        log_dir = os.path.dirname(config_dict['logging']['file'])
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        file_handler = RotatingFileHandler(
            config_dict['logging']['file'],
            maxBytes=10 * 1024 * 1024,
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Add console handler
    if config_dict['logging']['console']:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)


def init_config():
    """
    Initialize configuration in session state if not already present.
    This should be called at the start of the app.
    """
    if 'config' not in st.session_state:
        # Set a basic console handler initially
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)

        # Try to load from file, or create default
        st.session_state.config = load_config_from_file()

        # Setup logging with the loaded config
        setup_logging(st.session_state.config)


def load_config_from_file(config_path="config.yaml"):
    """
    Load configuration from file or create default if file doesn't exist.

    Args:
        config_path (str): Path to the configuration file

    Returns:
        dict: The configuration dictionary
    """
    try:
        # Try to load config from file
        with open(config_path, 'r') as file:
            loaded_config = yaml.safe_load(file)
        logger.info(f"Configuration loaded from {config_path}")

        # Ensure necessary directories exist
        ensure_directories(loaded_config)

        return loaded_config

    except FileNotFoundError:
        # Create default configuration if file doesn't exist
        logger.info(f"Configuration file {config_path} not found, creating default")
        default_config = create_default_config()

        # Save the default config to file
        with open(config_path, 'w') as file:
            yaml.dump(default_config, file, default_flow_style=False)

        # Ensure necessary directories exist
        ensure_directories(default_config)

        return default_config

    except Exception as e:
        # Handle any other errors
        logger.error(f"Error loading configuration: {str(e)}", exc_info=True)
        default_config = create_default_config()

        # Ensure necessary directories exist
        ensure_directories(default_config)

        return default_config


def save_config_to_file(config_dict, config_path="config.yaml"):
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


def update_config(new_config):
    """
    Update the configuration in session state and apply changes.

    Args:
        new_config (dict): The new configuration dictionary

    Returns:
        bool: True if updated successfully, False otherwise
    """
    # Save to file first
    if save_config_to_file(new_config):
        # Update in session state
        st.session_state.config = new_config

        # Update logging configuration
        setup_logging(new_config)

        # Ensure directories exist
        ensure_directories(new_config)

        return True

    return False


def save_uploaded_file(uploaded_file):
    """
    Save an uploaded file to the uploads directory.

    Args:
        uploaded_file (UploadedFile): The uploaded file

    Returns:
        str or None: Path to the saved file or None if saving is disabled
    """
    config = st.session_state.config

    if not config['data']['save_uploads']:
        return None

    uploads_dir = config['data']['uploads_dir']
    os.makedirs(uploads_dir, exist_ok=True)

    file_path = os.path.join(uploads_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    logger.info(f"Saved uploaded file to {file_path}")
    return file_path


def check_saved_model_exists():
    """
    Check if a saved model exists.

    Returns:
        str or None: Filename of the most recent model or None if no models found
    """
    config = st.session_state.config
    models_dir = config['model']['models_dir']

    if os.path.exists(models_dir):
        # Look for model files (either .joblib or .pkl)
        model_files = [
            f for f in os.listdir(models_dir)
            if f.startswith('music_preferences_model_') and (f.endswith('.joblib') or f.endswith('.pkl'))
        ]

        if model_files:
            latest_model = sorted(model_files)[-1]
            logger.info(f"Found latest model: {latest_model}")
            return latest_model

    logger.info("No saved models found")
    return None


def config_sidebar():
    """
    Display and handle configuration sidebar.

    Returns:
        bool: True if configuration was updated, False otherwise
    """
    st.sidebar.header("Configuration")

    # Get current config from session state
    current_config = st.session_state.config

    # Create a deep copy to detect changes
    updated_config = {
        'model': {**current_config['model']},
        'data': {**current_config['data']},
        'app': {**current_config['app']},
        'logging': {**current_config['logging']}
    }

    # Model settings section
    st.sidebar.subheader("Model Settings")
    updated_config['model']['save_model'] = st.sidebar.checkbox(
        "Save trained models",
        value=current_config['model']['save_model']
    )
    updated_config['model']['use_saved_model'] = st.sidebar.checkbox(
        "Use saved models",
        value=current_config['model']['use_saved_model']
    )
    updated_config['model']['n_estimators'] = st.sidebar.slider(
        "Number of estimators",
        10, 500,
        value=current_config['model']['n_estimators']
    )

    # Data settings section
    st.sidebar.subheader("Data Settings")
    updated_config['data']['save_uploads'] = st.sidebar.checkbox(
        "Save uploaded files",
        value=current_config['data']['save_uploads']
    )

    # App settings section
    st.sidebar.subheader("App Settings")
    updated_config['app']['default_age'] = st.sidebar.number_input(
        "Default age",
        min_value=0,
        max_value=100,
        value=current_config['app']['default_age']
    )

    # Logging settings section
    st.sidebar.subheader("Logging Settings")
    log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    updated_config['logging']['level'] = st.sidebar.selectbox(
        "Logging level",
        log_levels,
        index=log_levels.index(current_config['logging']['level'])
    )

    # Save button
    if st.sidebar.button("Save Configuration"):
        # Check if anything actually changed
        changes_made = False
        for section in current_config:
            for key in updated_config[section]:
                if updated_config[section][key] != current_config[section][key]:
                    changes_made = True
                    break
            if changes_made:
                break

        if changes_made:
            if update_config(updated_config):
                st.sidebar.success("Configuration saved successfully!")
                return True
            else:
                st.sidebar.error("Failed to save configuration.")
        else:
            st.sidebar.info("No changes detected in configuration.")

    return False