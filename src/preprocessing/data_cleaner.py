# File: src/preprocessing/data_cleaner.py

import pandas as pd
import numpy as np
import ast
import re
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class NEODataCleaner:
    """Class for cleaning and preprocessing NASA NEO data"""

    def __init__(self):
        self.current_year = datetime.now().year

    def _parse_string_to_dict(self, text):
        """Parse string representation of dictionary to actual dict"""
        if not isinstance(text, str):
            return text

        try:
            # Replace single quotes with double quotes for JSON parsing
            text = text.replace("'", '"')
            return ast.literal_eval(text)
        except:
            try:
                # Try using json parser as fallback
                import json
                return json.loads(text)
            except:
                logger.warning(f"Could not parse text as dictionary: {text[:50]}...")
                return {}

    def _extract_designation(self, text):
        """Extract designation from object name/designation"""
        if not isinstance(text, str):
            return ""

        # Look for patterns like (1950 DA) or (2000 SG344)
        match = re.search(r'\(([^)]+)\)', text)
        if match:
            return match.group(1).strip()
        return text.strip()

    def _extract_diameter(self, diameter_dict, unit='kilometers'):
        """Extract diameter from nested dictionary"""
        if not isinstance(diameter_dict, dict):
            return np.nan

        try:
            if unit in diameter_dict:
                min_val = diameter_dict[unit]['estimated_diameter_min']
                max_val = diameter_dict[unit]['estimated_diameter_max']
                return (min_val + max_val) / 2
        except:
            pass

        return np.nan

    def _extract_most_recent_approach_date(self, approaches):
        """Extract the most recent close approach date"""
        if not isinstance(approaches, list) or len(approaches) == 0:
            return np.nan

        try:
            dates = [approach.get('close_approach_date', '') for approach in approaches
                     if isinstance(approach, dict)]
            dates = [d for d in dates if d]  # Remove empty strings
            if dates:
                return max(dates)
        except:
            pass

        return np.nan

    def _extract_min_miss_distance(self, approaches, unit='astronomical'):
        """Extract minimum miss distance from close approaches"""
        if not isinstance(approaches, list) or len(approaches) == 0:
            return np.nan

        try:
            distances = []
            for approach in approaches:
                if isinstance(approach, dict) and 'miss_distance' in approach:
                    miss_dist = approach['miss_distance']
                    if isinstance(miss_dist, dict) and unit in miss_dist:
                        try:
                            distances.append(float(miss_dist[unit]))
                        except:
                            pass

            if distances:
                return min(distances)
        except:
            pass

        return np.nan

    def _extract_max_velocity(self, approaches):
        """Extract maximum relative velocity from close approaches in km/s"""
        if not isinstance(approaches, list) or len(approaches) == 0:
            return np.nan

        try:
            velocities = []
            for approach in approaches:
                if isinstance(approach, dict) and 'relative_velocity' in approach:
                    rel_vel = approach['relative_velocity']
                    if isinstance(rel_vel, dict) and 'kilometers_per_second' in rel_vel:
                        try:
                            velocities.append(float(rel_vel['kilometers_per_second']))
                        except:
                            pass

            if velocities:
                return max(velocities)
        except:
            pass

        return np.nan

    def _classify_size(self, diameter_km):
        """Classify NEO by size based on diameter in km"""
        if pd.isna(diameter_km):
            return 'unknown'
        elif diameter_km < 0.05:
            return 'small'
        elif diameter_km < 0.5:
            return 'medium'
        else:
            return 'large'

    def _classify_risk(self, palermo_scale):
        """Classify risk based on Palermo Scale"""
        if pd.isna(palermo_scale):
            return 'unknown'
        elif palermo_scale < -8:
            return 'negligible'
        elif palermo_scale < -2:
            return 'very_low'
        elif palermo_scale < 0:
            return 'low'
        elif palermo_scale < 1:
            return 'moderate'
        else:
            return 'high'

    def clean_neo_data(self, neo_df):
        """Clean the NEO dataset"""
        logger.info("Cleaning NEO data")

        # Make a copy to avoid modifying the original
        df = neo_df.copy()

        # Convert string representations of dictionaries to actual dictionaries
        for col in ['estimated_diameter', 'close_approach_data']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: self._parse_string_to_dict(x) if isinstance(x, str) else x)

        # Convert boolean strings to actual booleans
        for col in ['is_potentially_hazardous_asteroid', 'is_sentry_object']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: True if x == 'True' else
                (False if x == 'False' else x))

        # Extract diameter information from the nested dictionary
        if 'estimated_diameter' in df.columns:
            # Extract diameter in kilometers (average of min and max)
            df['diameter_km'] = df['estimated_diameter'].apply(
                lambda x: self._extract_diameter(x, 'kilometers')
                if isinstance(x, dict) else np.nan
            )

            # Extract diameter in meters
            df['diameter_m'] = df['estimated_diameter'].apply(
                lambda x: self._extract_diameter(x, 'meters')
                if isinstance(x, dict) else np.nan
            )

        # Extract close approach information
        if 'close_approach_data' in df.columns:
            # Most recent close approach date
            df['most_recent_approach'] = df['close_approach_data'].apply(
                lambda x: self._extract_most_recent_approach_date(x)
                if isinstance(x, list) and len(x) > 0 else np.nan
            )

            # Minimum miss distance (in astronomical units)
            df['min_miss_distance_au'] = df['close_approach_data'].apply(
                lambda x: self._extract_min_miss_distance(x, 'astronomical')
                if isinstance(x, list) and len(x) > 0 else np.nan
            )

            # Maximum relative velocity (in km/s)
            df['max_velocity_kms'] = df['close_approach_data'].apply(
                lambda x: self._extract_max_velocity(x)
                if isinstance(x, list) and len(x) > 0 else np.nan
            )

            # Number of close approaches
            df['approach_count'] = df['close_approach_data'].apply(
                lambda x: len(x) if isinstance(x, list) else 0
            )

        # Create size classification based on diameter
        df['size_category'] = df['diameter_km'].apply(self._classify_size)

        logger.info("NEO data cleaning complete")
        return df

    def clean_sentry_data(self, sentry_df):
        """Clean the Sentry dataset"""
        logger.info("Cleaning Sentry data")

        # Make a copy to avoid modifying the original
        df = sentry_df.copy()

        # Extract year information
        if 'Year Range' in df.columns:
            # Start year
            df['impact_start_year'] = df['Year Range'].apply(
                lambda x: int(x.split('-')[0]) if isinstance(x, str) and '-' in x else np.nan
            )

            # End year
            df['impact_end_year'] = df['Year Range'].apply(
                lambda x: int(x.split('-')[1]) if isinstance(x, str) and '-' in x else np.nan
            )

            # Time span
            df['impact_time_span'] = df['impact_end_year'] - df['impact_start_year'] + 1

        # Create risk categories based on Palermo Scale
        if 'Palermo Scale (cum.)' in df.columns:
            df['risk_category'] = df['Palermo Scale (cum.)'].apply(self._classify_risk)

        # Calculate years until potential impact
        df['years_until_impact'] = df['impact_start_year'].apply(
            lambda x: x - self.current_year if not pd.isna(x) else np.nan
        )

        logger.info("Sentry data cleaning complete")
        return df