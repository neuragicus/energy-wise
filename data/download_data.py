"""
Download public UCI Energy datasets:
1. Energy Efficiency Data Set (features for building design)
2. Appliances Energy Prediction Data Set (hourly appliance consumption)

Note: This script requires an internet connection to download real data from
the UCI Machine Learning Repository.
"""

import logging
import os

import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
ENERGY_DATA_URL = url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv"


def download_data() -> None:
    """Download appliances energy data from UCI ML Repository."""
    # Save as appliances_energy.csv to match expected filename in constants
    filepath = "data/appliances_energy.csv"

    if os.path.exists(filepath):
        logger.info(f"{filepath} already exists.")
        return

    logger.info(f"Downloading from {url}...")

    response = requests.get(url)
    response.raise_for_status()

    with open(filepath, "wb") as f:
        f.write(response.content)

    logger.info(f"Downloaded to {filepath}")


if __name__ == "__main__":
    download_data()
    logger.info("Data download complete!")
