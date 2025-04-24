import logging
import os
from datetime import datetime

def setup_logger(log_to_file=True, log_to_console=True, log_level='DEBUG', log_dir='Script/logs'):
    # Create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Prevent adding duplicate handlers if this function is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Define a formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s - Line:%(lineno)d - %(message)s'
    )

    # Setup file handler if enabled
    if log_to_file:
        os.makedirs(log_dir, exist_ok=True)
        log_filename = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
        log_path = os.path.join(log_dir, log_filename)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Setup console handler if enabled
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

