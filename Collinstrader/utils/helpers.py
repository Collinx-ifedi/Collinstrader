import os
import yaml

def load_config(path=None):
    """
    Load the YAML config file from the specified path.
    Defaults to the standard project config path if not provided.
    """
    if path is None:
        # Default path inside the project directory
        path = "/storage/emulated/0/Android/data/ru.iiec.pydroid3/files/cryptsignal/config/credentials.yaml"

    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found at: {path}")

    with open(path, "r") as f:
        return yaml.safe_load(f)