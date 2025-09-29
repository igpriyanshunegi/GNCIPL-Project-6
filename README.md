# Seattle Weather Analysis and Prediction

This project analyzes Seattle weather data and implements a weather prediction model using both real and synthetic data generated through CTGAN.

## Project Structure

```
📁 Project/
├── 📓 Notebooks/
│ ├── 1_EDA.ipynb             # Exploratory Data Analysis
│ ├── 2_CTGAN_Training.ipynb  # Synthetic Data Generation
│ ├── 3_Model_Training.ipynb  # Weather Prediction Model
│ └── 4_Visualization.ipynb   # Data Visualization
├── 📁 Data/
│ └── synthetic_weather.csv   # Generated synthetic data
├── 📁 Models/
│ ├── weather_model.pkl       # Trained Random Forest model
│ └── scaler.pkl              # Feature scaler
├── 📁 results/ 
│ └── cross_validation_results.csv         
├── seattle-weather.csv     # Original dataset
└── app.py                    # Streamlit web application

## Features

- Comprehensive EDA of Seattle weather patterns
- Synthetic data generation using CTGAN
- Weather prediction model using Random Forest
- Interactive web application for predictions
- Comparison visualizations of real vs synthetic data

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- sdv (Synthetic Data Vault)
- streamlit
- seaborn
- matplotlib

## Setup

1. Clone the repository
2. Install requirements: `pip install -r requirements.txt`
3. Run the notebooks in order
4. Launch the web app: `streamlit run app.py`

## Model Performance

The weather prediction model achieves the following performance metrics:
- Accuracy: ~85%
- Detailed metrics available in the Model Training notebook

## Web Application

The Streamlit web application allows users to:
- Input weather parameters
- Get instant weather predictions
- View prediction confidence scores