# Near-Earth Object (NEO) Hazard Predictor

![NEO Hazard Predictor](https://img.shields.io/badge/Project-Portfolio-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen)
![Streamlit](https://img.shields.io/badge/Streamlit-1.22.0-red)
![Machine Learning](https://img.shields.io/badge/ML-Scikit--learn-orange)

## 🌠 Project Overview

This application uses machine learning to predict the potential hazard of Near-Earth Objects (NEOs) based on their physical and orbital characteristics. The predictive models were trained on NASA data from the Center for Near Earth Object Studies (CNEOS) and the NASA/JPL Small-Body Database.

![Screenshot of NEO Predictor](path/to/screenshot.png)

## 📊 Features

- **Hazard Classification**: Determine if an asteroid is potentially hazardous
- **Impact Probability Estimation**: Predict the likelihood of Earth impact
- **Risk Assessment**: Evaluate combined risk factors based on multiple parameters
- **Interactive Visualizations**: Compare your object to known NEOs
- **Batch Processing**: Upload CSV files for bulk predictions

## 🔧 Technologies Used

- **Python**: Core programming language
- **Pandas & NumPy**: Data manipulation and numerical operations
- **Scikit-learn**: Machine learning models
- **Matplotlib & Plotly**: Data visualization
- **Streamlit**: Web application framework

## 🛠️ Installation and Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Setting Up the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/neo-predictor.git
   cd neo-predictor
   ```

2. Create and activate a virtual environment:
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate
   
   # macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Make sure the data files are in the `data/` directory:
   - `data/cneos_sentry_summary_data.csv`
   - `data/neo_data.csv`

5. Process the data and train the models:
   ```bash
   python main.py
   python train.py
   ```

6. Run the application:
   ```bash
   streamlit run app/app.py
   ```

The application should now be running at `http://localhost:8501`

## 📁 Project Structure

```
neo_predictor/
├── data/                 # Dataset storage
│   ├── cneos_sentry_summary_data.csv
│   ├── neo_data.csv
│   └── processed/        # Processed data (generated)
├── src/                  # Source code
│   ├── preprocessing/    # Data preprocessing modules
│   ├── features/         # Feature engineering
│   ├── models/           # ML model definitions
│   ├── evaluation/       # Model evaluation
│   ├── visualization/    # Data visualization
│   └── utils/            # Utility functions
├── models/               # Trained model files (generated)
├── app/                  # Streamlit application
├── notebooks/            # Jupyter notebooks (optional)
├── tests/                # Unit tests
├── main.py               # Main data processing script
├── train.py              # Model training script
├── predict.py            # Prediction script
└── requirements.txt      # Dependencies
```

## 🚀 Usage

### Single NEO Prediction

1. Navigate to the "Single NEO Prediction" page
2. Enter the physical parameters of your NEO:
   - Absolute Magnitude (H)
   - Estimated Diameter (km)
   - Minimum Approach Distance (AU)
   - Relative Velocity (km/s)
3. Add advanced parameters if desired
4. Click "Predict Hazard Potential"
5. View the results, including:
   - Hazard Probability
   - Impact Probability
   - Overall Risk Assessment
   - Comparison with known NEOs

### Batch Prediction

1. Navigate to the "Batch Prediction" page
2. Prepare a CSV file with columns for NEO parameters
   - Required columns: `absolute_magnitude_h`, `diameter_km`
   - Optional columns: `min_miss_distance_au`, `max_velocity_kms`, etc.
3. Upload the CSV file
4. Click "Process Batch"
5. View and download the prediction results

## 📊 Model Performance

The machine learning models in this project achieve the following performance metrics:

- **Hazard Classification**:
  - Accuracy: ~95%
  - Precision: ~92%
  - Recall: ~94%
  - F1 Score: ~93%

- **Impact Probability Prediction**:
  - RMSE: ~0.002
  - R² Score: ~0.85

## 📚 Data Sources

This project uses data from:

- [NASA Center for Near Earth Object Studies (CNEOS)](https://cneos.jpl.nasa.gov/)
- [NASA/JPL Small-Body Database](https://ssd.jpl.nasa.gov/sbdb.cgi)
- [NASA NEO API](https://api.nasa.gov/)

## 🧪 Future Improvements

- Integration with real-time NEO data from NASA API
- Implementation of more advanced ML models (deep learning)
- 3D visualization of NEO orbits
- Mobile application version
- Uncertainty quantification for predictions

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- NASA/JPL for providing open access to NEO data
- The scikit-learn and Streamlit teams for their excellent tools

---

*This project was developed as a portfolio project and should not be used for official asteroid hazard assessment. For official information, please refer to NASA's Center for Near Earth Object Studies.*
