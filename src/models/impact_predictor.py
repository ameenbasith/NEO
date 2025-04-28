# File: src/models/impact_predictor.py

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging
from src.utils.constants import RANDOM_SEED, CV_FOLDS

logger = logging.getLogger(__name__)


class ImpactPredictor:
    """
    Regression model for predicting impact probability and risk scales
    """

    def __init__(self, target='impact_probability', model_type='gradient_boosting'):
        """
        Initialize the impact predictor

        Args:
            target: Target variable to predict ('impact_probability', 'palermo_scale', or 'torino_scale')
            model_type: Type of model to use ('gradient_boosting' or 'neural_network')
        """
        self.target = target
        self.model_type = model_type
        self.model = None
        self.feature_importances = None

        if model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=RANDOM_SEED
            )
        elif model_type == 'neural_network':
            self.model = MLPRegressor(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                batch_size='auto',
                learning_rate='adaptive',
                max_iter=500,
                random_state=RANDOM_SEED
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def train(self, X_train, y_train):
        """
        Train the impact predictor

        Args:
            X_train: Training features
            y_train: Training target values

        Returns:
            Trained model
        """
        logger.info(f"Training {self.model_type} for predicting {self.target}")

        # Train the model
        self.model.fit(X_train, y_train)

        # Store feature importances if available
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importances = self.model.feature_importances_

        return self.model

    def predict(self, X):
        """
        Make predictions with the trained model

        Args:
            X: Feature data for prediction

        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model performance

        Args:
            X_test: Test features
            y_test: Test target values

        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        y_pred = self.model.predict(X_test)

        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }

        logger.info(f"Evaluation metrics for {self.target} prediction:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")

        return metrics

    def tune_hyperparameters(self, X_train, y_train, param_grid=None):
        """
        Tune model hyperparameters using GridSearchCV

        Args:
            X_train: Training features
            y_train: Training target values
            param_grid: Dictionary of hyperparameter grids

        Returns:
            Best model after tuning
        """
        logger.info(f"Tuning hyperparameters for {self.model_type} {self.target} predictor")

        if param_grid is None:
            # Default parameter grids for each model type
            if self.model_type == 'gradient_boosting':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            else:  # neural_network
                param_grid = {
                    'hidden_layer_sizes': [(50,), (100,), (100, 50)],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive']
                }

        # Create grid search with cross-validation
        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=CV_FOLDS,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )

        # Train with grid search
        grid_search.fit(X_train, y_train)

        # Set the best model
        self.model = grid_search.best_estimator_

        # Store feature importances if available
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importances = self.model.feature_importances_

        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best RMSE: {np.sqrt(-grid_search.best_score_):.4f}")

        return self.model