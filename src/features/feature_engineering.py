# File: src/features/feature_engineering.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import logging

from src.utils.constants import (
    PHYSICAL_FEATURES,
    ORBITAL_FEATURES,
    TEMPORAL_FEATURES,
    CATEGORICAL_FEATURES
)

logger = logging.getLogger(__name__)


class NEOFeatureEngineer:
    """Class for engineering features from NEO and Sentry data"""

    def __init__(self):
        self.preprocessor = None
        self.feature_names = None

    def merge_datasets(self, neo_df, sentry_df):
        """
        Merge the NEO and Sentry datasets on object designation

        Args:
            neo_df: Cleaned NEO DataFrame
            sentry_df: Cleaned Sentry DataFrame

        Returns:
            Merged DataFrame
        """
        logger.info("Merging NEO and Sentry datasets")

        # Create a clean designation column for both datasets to use as a key
        neo_copy = neo_df.copy()
        sentry_copy = sentry_df.copy()

        # Extract designations using a helper function
        neo_copy['clean_designation'] = neo_copy['name'].apply(self._extract_designation)
        sentry_copy['clean_designation'] = sentry_copy['Object Designation'].apply(self._extract_designation)

        # Merge on clean designation
        merged_df = pd.merge(
            neo_copy, sentry_copy,
            on='clean_designation',
            how='outer',
            suffixes=('_neo', '_sentry')
        )

        # Add a source indicator
        merged_df['in_neo'] = ~merged_df['id'].isna()
        merged_df['in_sentry'] = ~merged_df['Object Designation'].isna()

        logger.info(f"Merged data shape: {merged_df.shape}")
        return merged_df

    def _extract_designation(self, text):
        """Extract designation from object name/designation"""
        if not isinstance(text, str):
            return ""

        # Look for patterns like (1950 DA) or (2000 SG344)
        import re
        match = re.search(r'\(([^)]+)\)', text)
        if match:
            return match.group(1).strip()
        return text.strip()

    def create_advanced_features(self, df):
        """
        Create advanced features from the merged dataset

        Args:
            df: Merged DataFrame with NEO and Sentry data

        Returns:
            DataFrame with additional engineered features
        """
        logger.info("Creating advanced features")

        result_df = df.copy()

        # 1. Create orbital eccentricity proxy
        # If we have both velocity and miss distance, we can create a proxy for orbital eccentricity
        if 'max_velocity_kms' in result_df.columns and 'min_miss_distance_au' in result_df.columns:
            # Higher values indicate more eccentric orbits
            result_df['orbit_eccentricity_proxy'] = (
                    result_df['max_velocity_kms'] /
                    (result_df['min_miss_distance_au'] + 0.001)  # Adding small value to avoid division by zero
            )

        # 2. Create MOID (Minimum Orbital Intersection Distance) risk factor
        # MOID < 0.05 AU is considered potentially hazardous
        if 'min_miss_distance_au' in result_df.columns:
            result_df['moid_risk_factor'] = 1 / (result_df['min_miss_distance_au'] + 0.001)

        # 3. Create kinetic energy proxy (combines velocity and mass/size)
        if 'max_velocity_kms' in result_df.columns and 'diameter_km' in result_df.columns:
            # Kinetic energy is proportional to mass * velocity^2
            # Mass is proportional to diameter^3 (assuming similar density)
            result_df['kinetic_energy_proxy'] = (
                    (result_df['diameter_km'] ** 3) *
                    (result_df['max_velocity_kms'] ** 2)
            )

        # 4. Create risk score combining multiple factors
        # We'll create a risk score that combines:
        # - Impact probability (if available)
        # - Size (larger = more risk)
        # - Velocity (faster = more risk)
        # - Miss distance (closer = more risk)

        # Normalize components to 0-1 scale
        components = []

        # Size component (0-1)
        if 'diameter_km' in result_df.columns:
            max_diameter = result_df['diameter_km'].max()
            min_diameter = result_df['diameter_km'].min()
            size_component = (result_df['diameter_km'] - min_diameter) / (max_diameter - min_diameter)
            components.append(size_component)

        # Velocity component (0-1)
        if 'max_velocity_kms' in result_df.columns:
            max_velocity = result_df['max_velocity_kms'].max()
            min_velocity = result_df['max_velocity_kms'].min()
            velocity_component = (result_df['max_velocity_kms'] - min_velocity) / (max_velocity - min_velocity)
            components.append(velocity_component)

        # Miss distance component (0-1, inverted so closer = higher risk)
        if 'min_miss_distance_au' in result_df.columns:
            max_distance = result_df['min_miss_distance_au'].max()
            min_distance = result_df['min_miss_distance_au'].min()
            range_distance = max_distance - min_distance
            if range_distance > 0:
                distance_component = 1 - ((result_df['min_miss_distance_au'] - min_distance) / range_distance)
                components.append(distance_component)

        # Impact probability component (already 0-1)
        if 'Impact Probability (cumulative)' in result_df.columns:
            prob_component = result_df['Impact Probability (cumulative)'].fillna(0)
            components.append(prob_component)

        # Combine components (average of available components)
        if components:
            # Convert to DataFrame for easier manipulation
            components_df = pd.DataFrame(components).T
            # Replace NaNs with 0 for calculation
            components_df = components_df.fillna(0)
            # Calculate average across row
            result_df['combined_risk_score'] = components_df.mean(axis=1)

        logger.info("Advanced feature creation complete")
        return result_df

    def create_preprocessing_pipeline(self, df, numerical_features=None, categorical_features=None):
        """
        Create a preprocessing pipeline for model training

        Args:
            df: DataFrame with features
            numerical_features: List of numerical feature columns (if None, use defaults)
            categorical_features: List of categorical feature columns (if None, use defaults)

        Returns:
            ColumnTransformer preprocessing pipeline
        """
        if numerical_features is None:
            numerical_features = (
                    PHYSICAL_FEATURES +
                    ORBITAL_FEATURES +
                    TEMPORAL_FEATURES +
                    ['orbit_eccentricity_proxy', 'moid_risk_factor',
                     'kinetic_energy_proxy', 'combined_risk_score']
            )
            # Filter to only include columns that exist in the dataframe
            numerical_features = [f for f in numerical_features if f in df.columns]

        if categorical_features is None:
            categorical_features = CATEGORICAL_FEATURES
            # Filter to only include columns that exist in the dataframe
            categorical_features = [f for f in categorical_features if f in df.columns]

        # Create preprocessors for different types of features
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Combine transformers in a column transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )

        # Store for later use
        self.preprocessor = preprocessor
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features

        return preprocessor

    def _get_feature_names(self, column_transformer, numerical_features, categorical_features):
        """
        Get feature names after preprocessing

        Args:
            column_transformer: ColumnTransformer
            numerical_features: List of numerical feature names
            categorical_features: List of categorical feature names

        Returns:
            List of feature names after preprocessing
        """
        feature_names = []

        # Add numerical feature names (these keep their original names)
        feature_names.extend(numerical_features)

        # For categorical features, we need to generate the one-hot encoded feature names
        # without fitting the transformer
        for cat_feature in categorical_features:
            # Get unique values from an example categorical value
            # This is a placeholder approach since we don't have the actual values yet
            feature_names.append(f"{cat_feature}_placeholder")

        return feature_names