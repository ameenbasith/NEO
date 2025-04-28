# File: src/preprocessing/data_loader.py

import pandas as pd
import numpy as np
import logging
from src.utils.constants import NEO_DATA_PATH, SENTRY_DATA_PATH

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_neo_data(filepath=NEO_DATA_PATH):
    """
    Load NEO dataset from CSV file

    Args:
        filepath: Path to the NEO CSV file

    Returns:
        pandas DataFrame with NEO data
    """
    logger.info(f"Loading NEO data from {filepath}")
    try:
        neo_data = pd.read_csv(filepath)
        logger.info(f"Loaded NEO data with shape: {neo_data.shape}")
        return neo_data
    except Exception as e:
        logger.error(f"Error loading NEO data: {e}")
        raise


def load_sentry_data(filepath=SENTRY_DATA_PATH):
    """
    Load Sentry dataset from CSV file

    Args:
        filepath: Path to the Sentry CSV file

    Returns:
        pandas DataFrame with Sentry data
    """
    logger.info(f"Loading Sentry data from {filepath}")
    try:
        sentry_data = pd.read_csv(filepath)
        # Clean column names by removing trailing whitespace
        sentry_data.columns = [col.strip() for col in sentry_data.columns]
        logger.info(f"Loaded Sentry data with shape: {sentry_data.shape}")
        return sentry_data
    except Exception as e:
        logger.error(f"Error loading Sentry data: {e}")
        raise


def load_all_data():
    """
    Load both NEO and Sentry datasets

    Returns:
        Tuple of (neo_data, sentry_data) DataFrames
    """
    neo_data = load_neo_data()
    sentry_data = load_sentry_data()
    return neo_data, sentry_data