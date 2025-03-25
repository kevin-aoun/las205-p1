import os
import yaml
import logging
from logging.handlers import RotatingFileHandler

config = None
logger = None


def load_config(config_path="config.yaml"):
    """
    Load configuration from YAML file

    Args:
        config_path (str): Path to config YAML file

    Returns:
        dict: Configuration dictionary
    """
    global config

    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        # Create necessary directories
        if config['model']['save_model']:
            os.makedirs(config['model']['models_dir'], exist_ok=True)

        if config['data']['save_uploads']:
            os.makedirs(config['data']['uploads_dir'], exist_ok=True)

        return config
    except Exception as e:
        # If config file doesn't exist or has errors, use default config
        config = {
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
                'file': 'app.log',
                'console': True
            }
        }

        # Create default config file if it doesn't exist
        if not os.path.exists(config_path):
            with open(config_path, 'w') as file:
                yaml.dump(config, file, default_flow_style=False)

        # Create necessary directories
        os.makedirs(config['model']['models_dir'], exist_ok=True)
        os.makedirs(config['data']['uploads_dir'], exist_ok=True)

        return config


def setup_logging():
    """
    Set up logging based on configuration

    Returns:
        logging.Logger: Configured logger
    """
    global logger, config

    if config is None:
        config = load_config()

    # Create logger
    logger = logging.getLogger('music_preference_predictor')

    # Set level
    log_level = getattr(logging, config['logging']['level'])
    logger.setLevel(log_level)

    # Create formatter
    formatter = logging.Formatter(config['logging']['format'])

    # Clear existing handlers to avoid duplicate logs
    if logger.handlers:
        logger.handlers.clear()

    # File handler
    if config['logging']['file']:
        file_handler = RotatingFileHandler(
            config['logging']['file'],
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Console handler
    if config['logging']['console']:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


if config is None:
    config = load_config()

if logger is None:
    logger = setup_logging()