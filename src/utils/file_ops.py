"""
Utility functions for file operations (pickle, serialization, etc.)
"""

import logging
import pickle
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def save_pickle(obj: Any, filepath: Path, description: str = "") -> None:
    """
    Save an object to a pickle file.

    Args:
        obj: Object to pickle and save
        filepath: Path where to save the pickle file
        description: Optional description for logging

    Returns:
        None

    Raises:
        IOError: If file cannot be written
    """
    try:
        with open(filepath, "wb") as f:
            pickle.dump(obj, f)
        desc = f" ({description})" if description else ""
        logger.info(f"Saved pickle file: {filepath.name}{desc}")
    except IOError as e:
        logger.error(f"Failed to save pickle file {filepath}: {e}")
        raise


def load_pickle(filepath: Path, description: str = "") -> Any:
    """
    Load an object from a pickle file.

    Args:
        filepath: Path to the pickle file
        description: Optional description for logging

    Returns:
        Loaded object

    Raises:
        FileNotFoundError: If file does not exist
        IOError: If file cannot be read
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Pickle file not found: {filepath}")

    try:
        with open(filepath, "rb") as f:
            obj = pickle.load(f)
        desc = f" ({description})" if description else ""
        logger.info(f"Loaded pickle file: {filepath.name}{desc}")
        return obj
    except IOError as e:
        logger.error(f"Failed to load pickle file {filepath}: {e}")
        raise
