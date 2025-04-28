# File: main.py

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import project modules
from src.preprocessing.data_loader import load_all_data
from src.preprocessing.data_cleaner import NEODataCleaner
from src.features.feature_engineering import NEOFeatureEngineer
from src.utils.constants import RANDOM_SEED, TEST_SIZE


def main():
    """Main function to run the NEO predictor pipeline"""
    logger.info("Starting NEO Predictor pipeline")

    # 1. Load data
    logger.info("Step 1: Loading data")
    neo_data, sentry_data = load_all_data()

    # 2. Clean data
    logger.info("Step 2: Cleaning data")
    cleaner = NEODataCleaner()
    neo_data_clean = cleaner.clean_neo_data(neo_data)
    sentry_data_clean = cleaner.clean_sentry_data(sentry_data)

    # 3. Merge and engineer features
    logger.info("Step 3: Feature engineering")
    engineer = NEOFeatureEngineer()
    merged_data = engineer.merge_datasets(neo_data_clean, sentry_data_clean)
    feature_data = engineer.create_advanced_features(merged_data)

    # 4. Create preprocessing pipeline
    logger.info("Step 4: Creating preprocessing pipeline")
    # Just create the preprocessor - don't try to get feature names yet
    preprocessor = engineer.create_preprocessing_pipeline(feature_data)

    # 5. Save processed data
    logger.info("Step 5: Saving processed data")
    os.makedirs('data/processed', exist_ok=True)
    feature_data.to_csv('data/processed/processed_neo_data.csv', index=False)

    logger.info("NEO Predictor pipeline completed successfully")

    return feature_data


if __name__ == "__main__":
    processed_data = main()

    # Print a summary of the processed data
    print("\nProcessed Data Summary:")
    print(f"Total objects: {len(processed_data)}")
    print(f"Potentially hazardous objects: {processed_data['is_potentially_hazardous_asteroid'].sum()}")

    # Save some sample plots for EDA
    plt.figure(figsize=(10, 6))
    if 'combined_risk_score' in processed_data.columns and 'diameter_km' in processed_data.columns:
        plt.scatter(processed_data['diameter_km'],
                    processed_data['combined_risk_score'],
                    alpha=0.5)
        plt.xlabel('Diameter (km)')
        plt.ylabel('Combined Risk Score')
        plt.title('Risk Score vs Size')
        plt.savefig('risk_vs_size.png')
        plt.close()