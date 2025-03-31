import logging
from logging.handlers import RotatingFileHandler

# Create a consistent, application-wide logger
logger = logging.getLogger('height_preference_predictor')


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
        try:
            # Directory should already be created by ensure_directories
            file_handler = RotatingFileHandler(
                config_dict['logging']['file'],
                maxBytes=10 * 1024 * 1024,
                backupCount=5
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.info("File logging initialized successfully")
        except Exception as e:
            print(f"Error setting up file logging: {str(e)}")
            # Continue with console logging

    # Add console handler
    if config_dict['logging']['console']:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Log a test message to verify setup
    logger.info("Logging system initialized")