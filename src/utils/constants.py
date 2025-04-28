# File: src/utils/constants.py

# Data paths
NEO_DATA_PATH = 'data/neo_data.csv'
SENTRY_DATA_PATH = 'data/cneos_sentry_summary_data.csv'

# Feature columns
PHYSICAL_FEATURES = ['absolute_magnitude_h', 'diameter_km', 'diameter_m']
ORBITAL_FEATURES = ['min_miss_distance_au', 'max_velocity_kms', 'approach_count']
TEMPORAL_FEATURES = ['years_until_impact', 'impact_time_span']
CATEGORICAL_FEATURES = ['size_category', 'risk_category']

# Target variables
CLASSIFICATION_TARGET = 'is_potentially_hazardous_asteroid'
REGRESSION_TARGETS = ['Impact Probability (cumulative)', 'Palermo Scale (cum.)', 'Torino Scale (max.)']

# Model parameters
RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5