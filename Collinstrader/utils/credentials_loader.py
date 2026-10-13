# File: src/utils/credentials_loader.py

import os
import yaml
from pathlib import Path
import logging

# Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def load_credentials(filepath: str = "/storage/emulated/0/Android/data/ru.iiec.pydroid3/files/cryptsignal/config/credentials.yaml") -> dict:
    """
    Load API credentials from a YAML file.

    Args:
        filepath (str): Path to the YAML file.

    Returns:
        dict: Dictionary of credentials.
    """
    try:
        config_path = Path(filepath)
        if not config_path.exists():
            raise FileNotFoundError(f"Credentials file not found: {filepath}")

        with open(config_path, "r") as file:
            credentials = yaml.safe_load(file)

        logger.info(f"Credentials loaded successfully from {filepath}")
        return credentials

    except Exception as e:
        logger.exception(f"Failed to load credentials: {e}")
        raise