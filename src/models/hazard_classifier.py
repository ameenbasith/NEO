# File: src/models/hazard_classifier.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)
import logging
from src.utils.constants import RANDOM_SEED, CV_FOLDS

logger = logging.getLogger(__name__)


class HazardClassifier:
    """
    Classifier for determining if an asteroid is potentially hazardous
    """

    def __init__(self, model_type='random_forest'):
        """
        Initialize the classifier

        Args:
            model_type: Type of model to use ('random_forest', 'gradient_boosting', or 'neural_network')
        """
        self.model_type = model_type
        self.model = None
        self.feature_importances = None

        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=RANDOM_SEED
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=RANDOM_SEED
            )
        elif model_type == 'neural_network':
            self.model = MLPClassifier(
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
        Train the hazard classifier

        Args:
            X_train: Training features
            y_train: Training labels (1 for hazardous, 0 for non-hazardous)

        Returns:
            Trained model
        """
        logger.info(f"Training {self.model_type} hazard classifier")

        # Convert y to 1/0 if it's boolean
        if y_train.dtype == bool:
            y_train = y_train.astype(int)

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
            Array of predictions (1 for hazardous, 0 for non-hazardous)
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Make probability predictions with the trained model

        Args:
            X: Feature data for prediction

        Returns:
            Array of probability predictions [prob_not_hazardous, prob_hazardous]
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        return self.model.predict_proba(X)

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model performance

        Args:
            X_test: Test features
            y_test: Test labels (1 for hazardous, 0 for non-hazardous)

        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        # Convert y to 1/0 if it's boolean
        if y_test.dtype == bool:
            y_test = y_test.astype(int)

        # Make predictions
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]  # Probability of hazardous class

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0)
        }

        # Calculate AUC if we have both classes in test set
        if len(np.unique(y_test)) > 1:
            metrics['roc_auc'] = roc_auc_score(y_test, y_proba)

        # Create confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = conf_matrix

        # Log evaluation results
        logger.info(f"Evaluation metrics for {self.model_type} hazard classifier:")
        for metric, value in metrics.items():
            if metric != 'confusion_matrix':
                logger.info(f"  {metric}: {value:.4f}")

        # Generate classification report
        report = classification_report(y_test, y_pred)
        logger.info(f"Classification report:\n{report}")

        return metrics

    def tune_hyperparameters(self, X_train, y_train, param_grid=None):
        """
        Tune model hyperparameters using GridSearchCV

        Args:
            X_train: Training features
            y_train: Training labels (1 for hazardous, 0 for non-hazardous)
            param_grid: Dictionary of hyperparameter grids

        Returns:
            Best model after tuning
        """
        logger.info(f"Tuning hyperparameters for {self.model_type} hazard classifier")

        if param_grid is None:
            # Default parameter grids for each model type
            if self.model_type == 'random_forest':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            elif self.model_type == 'gradient_boosting':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
            else:  # neural_network
                param_grid = {
                    'hidden_layer_sizes': [(50,), (100,), (100, 50), (200, 100, 50)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive']
                }

        # Convert y to 1/0 if it's boolean
        if y_train.dtype == bool:
            y_train = y_train.astype(int)

        # Create grid search with cross-validation
        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=CV_FOLDS,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )

        # Train with grid search
        grid_search.fit(X_train, y_train)

        # Set the best model
        self.model = grid_search.best_estimator_

        # Store feature importances if available
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importances = self.model.feature_importances_

        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best F1 score: {grid_search.best_score_:.4f}")

        return self.model

    def get_feature_importances(self, feature_names=None):
        """
        Get feature importances from the trained model

        Args:
            feature_names: List of feature names corresponding to the columns in X

        Returns:
            DataFrame with feature importances sorted in descending order
        """
        if self.feature_importances is None:
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importances = self.model.feature_importances_
            else:
                logger.warning(f"Model type {self.model_type} does not provide feature importances")
                return None

        # If feature names are provided, use them
        if feature_names is not None:
            if len(feature_names) != len(self.feature_importances):
                logger.warning(f"Number of feature names ({len(feature_names)}) doesn't match "
                               f"number of features ({len(self.feature_importances)})")
                feature_names = [f"Feature {i}" for i in range(len(self.feature_importances))]
        else:
            feature_names = [f"Feature {i}" for i in range(len(self.feature_importances))]

        # Create DataFrame with feature importances
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': self.feature_importances
        })

        # Sort by importance in descending order
        importance_df = importance_df.sort_values('Importance', ascending=False).reset_index(drop=True)

        return importance_df

    def save_model(self, filepath):
        """
        Save the trained model to a file

        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)

        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath, model_type='random_forest'):
        """
        Load a trained model from a file

        Args:
            filepath: Path to the saved model
            model_type: Type of model that was saved

        Returns:
            HazardClassifier instance with the loaded model
        """
        import pickle
        with open(filepath, 'rb') as f:
            model = pickle.load(f)

        # Create a new instance
        classifier = cls(model_type=model_type)
        classifier.model = model

        # Store feature importances if available
        if hasattr(model, 'feature_importances_'):
            classifier.feature_importances = model.feature_importances_

        logger.info(f"Model loaded from {filepath}")
        return classifier