# Network Attack Detection using Synthetic Data Generation

This project implements an advanced anomaly detection system for cybersecurity using CTGAN (Conditional Tabular GAN) to generate synthetic attack logs and machine learning models for detection.

## Project Structure

```
📁 Project/
├── 📓 Notebooks/
│ ├── 1_EDA.ipynb             # Exploratory Data Analysis
│ ├── 2_CTGAN_Training.ipynb  # Synthetic Attack Log Generation
│ ├── 3_Model_Training.ipynb  # Anomaly Detection Models
│ └── 4_Visualization.ipynb   # Attack Pattern Analysis
├── 📁 Data/
│ ├── Test_data.csv     # Original dataset
│ └── Train_data.csv  #Trained dataset
├── 📁 models/
│ ├── best_cybersecurity_model.pkl    # Trained detection model
│ ├── cybersecurity_ctgan_model.pkl    # CTGAN model
│ ├── cybersecurity_scaler.pkl        # Scaler model
│ ├── lightgbm_cybersecurity.pkl        # Lightgbm model
│ ├── preprocessing_objects.pkl        # Preprocessing Object model
│ ├── random_forest_cybersecurity.pkl     # Random Forest model
│ ├── svm_cybersecurity.pkl              # SVM model
│ ├── training_metadata.pkl               # Metadata model
│ └── xgboost_cybersecurity.pkl             # XGBoost model
├── 📁 results/
│ ├── model_performance_comparison.csv 
│ └── synthetic_cybersecurity_data.csv
├── app.py                    # Streamlit web application
├── GNCIPL Project ppt.pptx   # Presentation file of project
├── GNCIPL Project report file.pdf  # Report file of project
└── README.md


## Features

- Comprehensive EDA of network attack patterns
- Synthetic attack log generation using CTGAN
- Multiple anomaly detection models:
  - Random Forest
  - XGBoost
  - LightGBM
  - SVM
- Interactive visualization dashboard
- Real-time attack detection web interface

## Dataset Description

The project uses the Network Attack Dataset containing:
- Network traffic features
- Attack types and patterns
- Temporal information
- Protocol information
- Attack severity levels

## Models Implemented

1. **CTGAN for Synthetic Data**
   - Generates realistic attack patterns
   - Preserves attack distributions
   - Maintains feature correlations

2. **Anomaly Detection Models**
   - Random Forest Classifier
   - XGBoost
   - LightGBM
   - SVM
   - Evaluation metrics for security context

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- sdv (Synthetic Data Vault)
- streamlit
- plotly
- xgboost
- lightgbm

## Setup

1. Clone the repository
2. Install requirements: `pip install -r requirements.txt`
3. Run notebooks in sequence
4. Launch detection app: `streamlit run app.py`

## Web Application Features

- Real-time network traffic analysis
- Attack probability scoring
- Attack type classification
- Confidence metrics
- Interactive visualizations

## Security Considerations

- Model interpretability for security teams
- False positive/negative analysis
- Attack pattern evolution tracking
- Model retraining capabilities

## Deployment Guidelines

1. Regular model updates
2. Performance monitoring
3. Alert threshold configuration
4. Integration with security systems

## Performance Metrics

- Detection Accuracy
- False Positive Rate
- False Negative Rate
- Detection Latency
- Model Confidence Scores

## Contributors

- [Priyanshu]