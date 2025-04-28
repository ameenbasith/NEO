# File: train.py

import os
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
import pickle
import json

from src.preprocessing.data_loader import load_all_data
from src.preprocessing.data_cleaner import NEODataCleaner
from src.features.feature_engineering import NEOFeatureEngineer
from src.models.hazard_classifier import HazardClassifier
from src.models.impact_predictor import ImpactPredictor
from src.models.ensemble import NEOEnsemble
from src.utils.constants import (
    RANDOM_SEED, TEST_SIZE,
    CLASSIFICATION_TARGET, REGRESSION_TARGETS
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def prepare_data():
    """
    Prepare the data for model training

    Returns:
        Processed DataFrame with features and targets
    """
    # 1. Load data
    neo_data, sentry_data = load_all_data()

    # 2. Clean data
    cleaner = NEODataCleaner()
    neo_data_clean = cleaner.clean_neo_data(neo_data)
    sentry_data_clean = cleaner.clean_sentry_data(sentry_data)

    # 3. Merge and engineer features
    engineer = NEOFeatureEngineer()
    merged_data = engineer.merge_datasets(neo_data_clean, sentry_data_clean)
    feature_data = engineer.create_advanced_features(merged_data)

    return feature_data, engineer


def train_hazard_classifier(data, engineer):
    """
    Train the hazard classification model

    Args:
        data: Processed DataFrame
        engineer: NEOFeatureEngineer instance

    Returns:
        Trained model and evaluation metrics
    """
    logger.info("Training hazard classification model")

    # Ensure target exists
    if CLASSIFICATION_TARGET not in data.columns:
        logger.error(f"Classification target '{CLASSIFICATION_TARGET}' not found in data")
        return None, None

    # Create preprocessing pipeline
    preprocessor = engineer.create_preprocessing_pipeline(data)

    # Filter rows with the target variable
    target_data = data.dropna(subset=[CLASSIFICATION_TARGET])

    # Make sure the target variable is boolean or binary (0/1)
    # This is the key fix for your issue
    target_data = target_data.copy()
    if isinstance(target_data[CLASSIFICATION_TARGET].iloc[0], bool):
        target_data[CLASSIFICATION_TARGET] = target_data[CLASSIFICATION_TARGET].astype(int)
    elif target_data[CLASSIFICATION_TARGET].dtype == object:
        # Convert string 'True'/'False' to int if needed
        target_data[CLASSIFICATION_TARGET] = target_data[CLASSIFICATION_TARGET].map(
            lambda x: 1 if x == True or x == 'True' else 0 if x == False or x == 'False' else None
        )
        # Drop any rows where conversion resulted in None
        target_data = target_data.dropna(subset=[CLASSIFICATION_TARGET])
        # Convert to int
        target_data[CLASSIFICATION_TARGET] = target_data[CLASSIFICATION_TARGET].astype(int)

    # Split features and target
    X = target_data.drop(CLASSIFICATION_TARGET, axis=1)
    y = target_data[CLASSIFICATION_TARGET]

    # Print unique values to debug
    logger.info(f"Target variable unique values: {np.unique(y)}")
    logger.info(f"Target variable dtype: {y.dtype}")

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    # Apply preprocessing
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Create and train classifier
    classifier = HazardClassifier(model_type='random_forest')
    classifier.train(X_train_processed, y_train)

    # Create directory for model artifacts
    os.makedirs('models', exist_ok=True)

    # Save model and preprocessor
    with open('models/hazard_classifier.pkl', 'wb') as f:
        pickle.dump(classifier.model, f)

    with open('models/hazard_preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)

    # Save feature names (after fitting)
    # Use get_feature_names_out method from fitted preprocessor if available
    try:
        feature_names = preprocessor.get_feature_names_out()
        with open('models/feature_names.json', 'w') as f:
            json.dump(feature_names.tolist(), f)
    except AttributeError:
        # For older sklearn versions that don't have get_feature_names_out
        logger.warning("Could not extract feature names from preprocessor")

    return classifier, (X_test_processed, y_test)

def train_impact_predictor(data, engineer, target_column):
    """
    Train the impact prediction model

    Args:
        data: Processed DataFrame
        engineer: NEOFeatureEngineer instance
        target_column: Name of the target column to predict

    Returns:
        Trained model and evaluation metrics
    """
    logger.info(f"Training impact predictor for {target_column}")

    # Ensure target exists
    if target_column not in data.columns:
        logger.error(f"Regression target '{target_column}' not found in data")
        return None, None

    # Create preprocessing pipeline
    preprocessor = engineer.create_preprocessing_pipeline(data)

    # Filter rows with the target variable
    target_data = data.dropna(subset=[target_column])

    # Split features and target
    X = target_data.drop(target_column, axis=1)
    y = target_data[target_column]

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    # Apply preprocessing
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Create and train predictor
    predictor = ImpactPredictor(target=target_column, model_type='gradient_boosting')
    predictor.train(X_train_processed, y_train)

    # Save model and preprocessor
    target_name = target_column.lower().replace(' ', '_').replace('(', '').replace(')', '')
    with open(f'models/impact_predictor_{target_name}.pkl', 'wb') as f:
        pickle.dump(predictor.model, f)

    with open(f'models/impact_preprocessor_{target_name}.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)

    return predictor, (X_test_processed, y_test)


def main():
    """
    Main function to train all models
    """
    logger.info("Starting model training")

    # Prepare data
    data, engineer = prepare_data()

    # Train hazard classifier
    classifier, clf_test_data = train_hazard_classifier(data, engineer)

    if classifier is not None and clf_test_data is not None:
        X_test, y_test = clf_test_data
        metrics = classifier.evaluate(X_test, y_test)
        logger.info(f"Hazard classifier metrics: {metrics}")

    # Train impact predictors for each regression target
    for target in REGRESSION_TARGETS:
        predictor, reg_test_data = train_impact_predictor(data, engineer, target)

        if predictor is not None and reg_test_data is not None:
            X_test, y_test = reg_test_data
            metrics = predictor.evaluate(X_test, y_test)
            logger.info(f"{target} predictor metrics: {metrics}")

    logger.info("Model training completed")


if __name__ == "__main__":
    main()