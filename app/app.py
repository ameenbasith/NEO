# File: app/app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import math

# Get the absolute path to the project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add parent directory to path to import from src
sys.path.append(BASE_DIR)
from src.preprocessing.data_cleaner import NEODataCleaner
from src.features.feature_engineering import NEOFeatureEngineer

# Page configuration
st.set_page_config(
    page_title="NEO Hazard Predictor",
    page_icon="☄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("☄️ Near-Earth Object Hazard Predictor")
st.markdown("""
This application predicts the hazard potential and impact probability of Near-Earth Objects (NEOs) 
using machine learning models trained on NASA data. You can either:
1. Enter the parameters of a known or hypothetical NEO
2. Upload a CSV file with multiple NEOs for batch prediction
""")


# Load models and preprocessors
@st.cache_resource
def load_models():
    # Dictionary to store loaded models and preprocessors
    models_dict = {}

    try:
        # Use absolute paths
        models_dir = os.path.join(BASE_DIR, 'models')

        # Check if directory exists
        if not os.path.exists(models_dir):
            st.warning(f"Models directory not found at: {models_dir}")
            return {}, False

        # Debug: List files in the directory
        st.write(f"Looking for models in: {models_dir}")
        st.write(f"Files in models directory: {os.listdir(models_dir)}")

        # Load hazard classifier
        hazard_model_path = os.path.join(models_dir, 'hazard_classifier.pkl')
        with open(hazard_model_path, 'rb') as f:
            models_dict['hazard_model'] = pickle.load(f)

        hazard_preprocessor_path = os.path.join(models_dir, 'hazard_preprocessor.pkl')
        with open(hazard_preprocessor_path, 'rb') as f:
            models_dict['hazard_preprocessor'] = pickle.load(f)

        # Load impact probability predictor
        impact_prob_model_path = os.path.join(models_dir, 'impact_predictor_impact_probability_cumulative.pkl')
        with open(impact_prob_model_path, 'rb') as f:
            models_dict['impact_prob_model'] = pickle.load(f)

        impact_prob_preprocessor_path = os.path.join(models_dir,
                                                     'impact_preprocessor_impact_probability_cumulative.pkl')
        with open(impact_prob_preprocessor_path, 'rb') as f:
            models_dict['impact_prob_preprocessor'] = pickle.load(f)

        # Load feature names
        feature_names_path = os.path.join(models_dir, 'feature_names.json')
        with open(feature_names_path, 'r') as f:
            models_dict['feature_names'] = json.load(f)

        # Load sample data for statistics
        processed_data_path = os.path.join(BASE_DIR, 'data', 'processed', 'processed_neo_data.csv')
        try:
            models_dict['sample_data'] = pd.read_csv(processed_data_path)
        except Exception as e:
            st.warning(f"Could not load sample data: {e}")
            models_dict['sample_data'] = None

        return models_dict, True
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return {}, False


# Function to make predictions
def predict_hazard(input_data, models_dict):
    """
    Make hazard predictions for input data

    Args:
        input_data: DataFrame with features
        models_dict: Dictionary containing models and preprocessors

    Returns:
        DataFrame with predictions
    """
    # Create a copy of the input data
    result_df = input_data.copy()

    # Get the list of required columns from the preprocessor
    required_columns = []
    for name, _, columns in models_dict['hazard_preprocessor'].transformers:
        if isinstance(columns, list):
            required_columns.extend(columns)
        else:
            required_columns.append(columns)

    # Add any missing columns with default values
    for column in required_columns:
        if column not in result_df.columns:
            st.warning(f"Adding missing column '{column}' with default value 0")
            result_df[column] = 0

    # If 'diameter_m' is specifically missing but 'diameter_km' is present, calculate it
    if 'diameter_m' not in result_df.columns and 'diameter_km' in result_df.columns:
        result_df['diameter_m'] = result_df['diameter_km'] * 1000
        st.info("Calculated diameter_m from diameter_km")

    # Preprocess data
    X_processed = models_dict['hazard_preprocessor'].transform(result_df)

    # Make predictions
    hazard_proba = models_dict['hazard_model'].predict_proba(X_processed)[:, 1]
    result_df['hazard_probability'] = hazard_proba

    # Classify as hazardous if probability > 0.5
    result_df['is_hazardous'] = hazard_proba > 0.5

    # Predict impact probability if model exists
    if 'impact_prob_model' in models_dict and 'impact_prob_preprocessor' in models_dict:
        try:
            # Add missing columns for impact probability prediction too
            impact_required_columns = []
            for name, _, columns in models_dict['impact_prob_preprocessor'].transformers:
                if isinstance(columns, list):
                    impact_required_columns.extend(columns)
                else:
                    impact_required_columns.append(columns)

            for column in impact_required_columns:
                if column not in result_df.columns:
                    result_df[column] = 0

            X_impact_processed = models_dict['impact_prob_preprocessor'].transform(result_df)
            impact_prob = models_dict['impact_prob_model'].predict(X_impact_processed)
            result_df['impact_probability'] = impact_prob
        except Exception as e:
            st.warning(f"Could not predict impact probability: {e}")

    return result_df


# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Single NEO Prediction", "Batch Prediction", "About"])

# Load models
models_dict, models_loaded = load_models()

if not models_loaded:
    st.error("Failed to load models. Please make sure you've trained the models first by running train.py.")
    st.stop()

# Create data cleaner and feature engineer for processing inputs
cleaner = NEODataCleaner()
engineer = NEOFeatureEngineer()

# Single NEO prediction page
if page == "Single NEO Prediction":
    st.header("Single NEO Prediction")
    st.markdown("Enter the parameters of a Near-Earth Object to predict its hazard potential.")

    # Create tabs for different input methods
    tabs = st.tabs(["Basic Parameters", "Advanced Parameters"])

    with tabs[0]:
        # Basic parameters input form
        st.subheader("Basic NEO Parameters")

        col1, col2 = st.columns(2)

        with col1:
            abs_magnitude = st.slider("Absolute Magnitude (H)", 10.0, 30.0, 20.0,
                                      help="Lower values indicate larger objects")
            diameter_km = st.number_input("Estimated Diameter (km)", 0.01, 10.0, 0.5,
                                          help="Estimated diameter in kilometers")

        with col2:
            min_distance = st.number_input("Minimum Approach Distance (AU)", 0.0001, 1.0, 0.05,
                                           help="Closest approach distance in Astronomical Units")
            rel_velocity = st.slider("Relative Velocity (km/s)", 5.0, 40.0, 20.0,
                                     help="Relative velocity at approach in km/s")

    with tabs[1]:
        # Advanced parameters input form
        st.subheader("Advanced NEO Parameters")

        col1, col2 = st.columns(2)

        with col1:
            approach_count = st.number_input("Number of Close Approaches", 1, 100, 5,
                                             help="Number of times the object approaches Earth")
            years_until_impact = st.slider("Years Until Potential Impact", 0, 200, 50,
                                           help="Number of years until the potential impact")

        with col2:
            impact_time_span = st.slider("Impact Time Span (years)", 1, 50, 10,
                                         help="Time span of potential impact period")
            size_category = st.selectbox("Size Category",
                                         ["small", "medium", "large"],
                                         index=1,
                                         help="Size classification of the object")

    # Button to make prediction
    if st.button("Predict Hazard Potential"):
        # Create a DataFrame with input values
        input_data = pd.DataFrame({
            'absolute_magnitude_h': [abs_magnitude],
            'diameter_km': [diameter_km],
            'diameter_m': [diameter_km * 1000],  # Add diameter in meters
            'min_miss_distance_au': [min_distance],
            'max_velocity_kms': [rel_velocity],
            'approach_count': [approach_count],
            'years_until_impact': [years_until_impact],
            'impact_time_span': [impact_time_span],
            'size_category': [size_category]
        })

        # Engineer additional features
        try:
            input_data = engineer.create_advanced_features(input_data)

            # Make prediction
            predictions = predict_hazard(input_data, models_dict)

            # Display results
            st.header("Prediction Results")

            # Create columns for different metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                hazard_prob = predictions['hazard_probability'].iloc[0]
                st.metric("Hazard Probability", f"{hazard_prob:.2%}")

                # Gauge chart for hazard probability
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=hazard_prob * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Hazard Level"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 25], 'color': "lightgreen"},
                            {'range': [25, 50], 'color': "yellow"},
                            {'range': [50, 75], 'color': "orange"},
                            {'range': [75, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': hazard_prob * 100
                        }
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                if 'impact_probability' in predictions.columns:
                    impact_prob = predictions['impact_probability'].iloc[0]
                    # Ensure probability is in reasonable range
                    impact_prob = max(0, min(impact_prob, 1))
                    st.metric("Impact Probability", f"{impact_prob:.6%}")

                    # Create visualization for impact probability
                    fig = go.Figure(go.Indicator(
                        mode="number+gauge",
                        value=impact_prob * 1000000,  # Scale up for visibility
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Impact Probability (per million)"},
                        gauge={
                            'axis': {'range': [0, 1000]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 10], 'color': "lightgreen"},
                                {'range': [10, 100], 'color': "yellow"},
                                {'range': [100, 1000], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': impact_prob * 1000000
                            }
                        }
                    ))
                    st.plotly_chart(fig, use_container_width=True)

            with col3:
                # Risk assessment based on both hazard and impact probability
                if 'impact_probability' in predictions.columns:
                    combined_risk = hazard_prob * math.sqrt(impact_prob)  # Combined risk metric
                    risk_level = "Low"
                    if combined_risk > 0.01:
                        risk_level = "Medium"
                    if combined_risk > 0.1:
                        risk_level = "High"
                    if combined_risk > 0.5:
                        risk_level = "Very High"

                    st.metric("Overall Risk Assessment", risk_level)

                    # Display additional information
                    st.markdown(f"""
                    **Object Classification:**  
                    - Size: {size_category.capitalize()}
                    - Diameter: {diameter_km:.2f} km
                    - Velocity: {rel_velocity:.2f} km/s

                    **Potential Energy Impact:**  
                    {diameter_km ** 3 * rel_velocity ** 2 / 2:.2e} MT TNT equivalent
                    """)

            # Show comparison with known objects
            if models_dict['sample_data'] is not None:
                st.subheader("Comparison with Known NEOs")

                # Create scatter plot of diameter vs hazard
                fig = px.scatter(
                    models_dict['sample_data'].sample(min(500, len(models_dict['sample_data']))),
                    x='diameter_km',
                    y='absolute_magnitude_h',
                    color='is_potentially_hazardous_asteroid',
                    opacity=0.7,
                    title="Your Object Compared to Known NEOs",
                    labels={
                        'diameter_km': 'Diameter (km)',
                        'absolute_magnitude_h': 'Absolute Magnitude',
                        'is_potentially_hazardous_asteroid': 'Hazardous'
                    }
                )

                # Add the current object as a large marker
                fig.add_trace(
                    go.Scatter(
                        x=[diameter_km],
                        y=[abs_magnitude],
                        mode='markers',
                        marker=dict(
                            color='yellow',
                            size=15,
                            line=dict(color='black', width=2)
                        ),
                        name='Your Object'
                    )
                )

                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error making prediction: {e}")
            st.exception(e)

# Batch prediction page
elif page == "Batch Prediction":
    st.header("Batch NEO Prediction")
    st.markdown("""
    Upload a CSV file with multiple NEO objects to make batch predictions.

    **Required columns:**
    - `absolute_magnitude_h`: Absolute magnitude (H)
    - `diameter_km`: Diameter in kilometers
    - Additional parameters will improve prediction accuracy
    """)

    # File uploader
    uploaded_file = st.file_uploader("Upload NEO data CSV", type="csv")

    if uploaded_file is not None:
        try:
            # Load the data
            input_df = pd.read_csv(uploaded_file)

            # Show data preview
            st.subheader("Data Preview")
            st.dataframe(input_df.head())

            # Check for required columns
            required_cols = ['absolute_magnitude_h', 'diameter_km']
            missing_cols = [col for col in required_cols if col not in input_df.columns]

            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
                st.stop()

            # Add diameter_m if not present
            if 'diameter_m' not in input_df.columns and 'diameter_km' in input_df.columns:
                input_df['diameter_m'] = input_df['diameter_km'] * 1000
                st.info("Added 'diameter_m' column calculated from 'diameter_km'")

            # Process button
            if st.button("Process Batch"):
                # Engineer features
                processed_df = engineer.create_advanced_features(input_df)

                # Make predictions
                results = predict_hazard(processed_df, models_dict)

                # Display results
                st.subheader("Prediction Results")
                st.dataframe(results)

                # Visualization of results
                st.subheader("Results Visualization")

                col1, col2 = st.columns(2)

                with col1:
                    # Distribution of hazard probabilities
                    fig = px.histogram(
                        results,
                        x='hazard_probability',
                        nbins=20,
                        title="Distribution of Hazard Probabilities",
                        labels={'hazard_probability': 'Hazard Probability'}
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    # Scatter plot of size vs hazard probability
                    fig = px.scatter(
                        results,
                        x='diameter_km',
                        y='hazard_probability',
                        color='is_hazardous',
                        title="Diameter vs Hazard Probability",
                        labels={
                            'diameter_km': 'Diameter (km)',
                            'hazard_probability': 'Hazard Probability',
                            'is_hazardous': 'Predicted Hazardous'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Download button for results
                csv = results.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name="neo_predictions.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.exception(e)

# About page
else:
    st.header("About the NEO Hazard Predictor")

    st.markdown("""
    ## Project Overview

    This application uses machine learning to predict the potential hazard of Near-Earth Objects (NEOs) based on their physical and orbital characteristics. The predictive models were trained on NASA data from the Center for Near Earth Object Studies (CNEOS) and the NASA/JPL Small-Body Database.

    ### Data Sources

    - **CNEOS Sentry System**: Monitors potential future Earth impact events
    - **NASA NEO API**: Provides near-Earth object close approach data

    ### Machine Learning Models

    The application employs several predictive models:

    1. **Hazard Classification**: Determines if an object should be classified as potentially hazardous
    2. **Impact Probability Prediction**: Estimates the probability of Earth impact
    3. **Risk Assessment**: Evaluates the overall risk posed by the object

    ### Key Features

    - Single object prediction
    - Batch processing for multiple objects
    - Interactive visualizations
    - Comparison with known NEOs

    ### Technical Details

    The models were developed using:
    - Python
    - Scikit-learn
    - Pandas
    - Streamlit
    - Plotly

    ## How to Use

    1. Navigate to "Single NEO Prediction" to analyze one object
    2. Enter the object's parameters (at minimum: absolute magnitude and diameter)
    3. Click "Predict Hazard Potential" to see results
    4. For multiple objects, use the "Batch Prediction" page and upload a CSV file

    ## Interpretation of Results

    - **Hazard Probability**: Likelihood the object is potentially hazardous
    - **Impact Probability**: Estimated chance of Earth impact
    - **Overall Risk**: Combined assessment based on size, velocity, and probabilities

    ## Limitations

    This tool provides estimates based on available data and should not be used for official hazard assessment. For official information, refer to NASA's Center for Near Earth Object Studies.
    """)

    # Add references and links
    st.subheader("References and Links")
    st.markdown("""
    - [NASA Center for Near Earth Object Studies](https://cneos.jpl.nasa.gov/)
    - [NASA/JPL Small-Body Database](https://ssd.jpl.nasa.gov/sbdb.cgi)
    - [NEO Earth Close Approaches](https://cneos.jpl.nasa.gov/ca/)
    - [Torino Impact Hazard Scale](https://cneos.jpl.nasa.gov/sentry/torino_scale.html)
    - [Palermo Technical Impact Hazard Scale](https://cneos.jpl.nasa.gov/sentry/palermo_scale.html)
    """)

# Footer
st.markdown("---")
st.markdown("NEO Hazard Predictor | Portfolio Project | Developed with ❤️ using Python and Streamlit")