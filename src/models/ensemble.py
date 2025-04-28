# File: src/models/ensemble.py

import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier, VotingRegressor
import logging
from src.models.hazard_classifier import HazardClassifier
from src.models.impact_predictor import ImpactPredictor
from src.utils.constants import RANDOM_SEED

logger = logging.getLogger(__name__)


class NEOEnsemble:
    """
    Ensemble model combining multiple NEO predictors
    """

    def __init__(self, prediction_type='classification'):
        """
        Initialize the ensemble model

        Args:
            prediction_type: Type of prediction ('classification' for hazard or 'regression' for impact probability)
        """
        self.prediction_type = prediction_type
        self.ensemble = None
        self.base_models = []

    def build_classification_ensemble(self):
        """
        Build an ensemble of classification models for hazard prediction

        Returns:
            VotingClassifier ensemble model
        """
        logger.info("Building classification ensemble for hazard prediction")

        # Create base models
        rf_model = HazardClassifier(model_type='random_forest')
        gb_model = HazardClassifier(model_type='gradient_boosting')
        nn_model = HazardClassifier(model_type='neural_network')

        # Store models
        self.base_models = [
            ('random_forest', rf_model.model),
            ('gradient_boosting', gb_model.model),
            ('neural_network', nn_model.model)
        ]

        # Create voting ensemble
        self.ensemble = VotingClassifier(
            estimators=self.base_models,
            voting='soft',  # Use probability estimates for voting
            n_jobs=-1
        )

        return self.ensemble

    def build_regression_ensemble(self, target='impact_probability'):
        """
        Build an ensemble of regression models for impact prediction

        Args:
            target: Target variable to predict

        Returns:
            VotingRegressor ensemble model
        """
        logger.info(f"Building regression ensemble for {target} prediction")

        # Create base models
        gb_model = ImpactPredictor(target=target, model_type='gradient_boosting')
        nn_model = ImpactPredictor(target=target, model_type='neural_network')

        # Store models
        self.base_models = [
            ('gradient_boosting', gb_model.model),
            ('neural_network', nn_model.model)
        ]

        # Create voting ensemble
        self.ensemble = VotingRegressor(
            estimators=self.base_models,
            n_jobs=-1
        )

        return self.ensemble

    def train(self, X_train, y_train):
        """
        Train the ensemble model

        Args:
            X_train: Training features
            y_train: Training labels/values

        Returns:
            Trained ensemble model
        """
        if self.ensemble is None:
            if self.prediction_type == 'classification':
                self.build_classification_ensemble()
            else:
                self.build_regression_ensemble()

        logger.info(f"Training {self.prediction_type} ensemble")
        self.ensemble.fit(X_train, y_train)

        return self.ensemble

    def predict(self, X):
        """
        Make predictions with the trained ensemble

        Args:
            X: Feature data for prediction

        Returns:
            Array of predictions
        """
        if self.ensemble is None:
            raise ValueError("Ensemble has not been trained yet")

        if self.prediction_type == 'classification' and hasattr(self.ensemble, 'predict_proba'):
            # Return probability of being hazardous
            return self.ensemble.predict_proba(X)[:, 1]
        else:
            # Return regular predictions
            return self.ensemble.predict(X)