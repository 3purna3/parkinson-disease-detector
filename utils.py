"""
Utility functions for the Parkinson's Disease Detection app.
Handles model loading, preprocessing, and predictions.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'saved')


def load_scaler():
    """Load the fitted StandardScaler."""
    return joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))


def load_feature_columns():
    """Load the list of feature column names."""
    return joblib.load(os.path.join(MODEL_DIR, 'feature_columns.pkl'))


def load_model(model_name):
    """
    Load a trained model by name.
    model_name: 'SVM', 'Random Forest', or 'CNN'
    """
    if model_name == 'SVM':
        return joblib.load(os.path.join(MODEL_DIR, 'svm_model.pkl'))
    elif model_name == 'Random Forest':
        return joblib.load(os.path.join(MODEL_DIR, 'rf_model.pkl'))
    elif model_name == 'CNN':
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        from tensorflow.keras.models import load_model as keras_load
        return keras_load(os.path.join(MODEL_DIR, 'cnn_model.h5'), compile=False)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def load_metrics():
    """Load saved model metrics."""
    with open(os.path.join(MODEL_DIR, 'metrics.json'), 'r') as f:
        return json.load(f)


def preprocess_input(features_dict):
    """
    Preprocess a dictionary of feature values.
    Returns scaled numpy array ready for prediction.
    """
    feature_cols = load_feature_columns()
    scaler = load_scaler()

    values = np.array([[features_dict[col] for col in feature_cols]])
    scaled = scaler.transform(values)
    return scaled


def predict(model_name, features_dict):
    """
    Make a prediction using the specified model.
    Returns (prediction, confidence).
    """
    scaled_input = preprocess_input(features_dict)
    model = load_model(model_name)

    if model_name == 'CNN':
        cnn_input = scaled_input.reshape(-1, scaled_input.shape[1], 1)
        prob = model.predict(cnn_input, verbose=0).flatten()[0]
        prediction = 1 if prob >= 0.5 else 0
        confidence = prob if prediction == 1 else 1 - prob
    elif model_name == 'SVM':
        prediction = model.predict(scaled_input)[0]
        prob = model.predict_proba(scaled_input)[0]
        confidence = prob[prediction]
    else:  # Random Forest
        prediction = model.predict(scaled_input)[0]
        prob = model.predict_proba(scaled_input)[0]
        confidence = prob[prediction]

    return int(prediction), float(confidence)


def predict_batch(model_name, df):
    """
    Make predictions for a DataFrame of samples.
    Returns DataFrame with predictions and confidence.
    """
    feature_cols = load_feature_columns()
    scaler = load_scaler()
    model = load_model(model_name)

    # Ensure all required columns exist
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    X = df[feature_cols].values
    X_scaled = scaler.transform(X)

    results = df.copy()

    if model_name == 'CNN':
        X_cnn = X_scaled.reshape(-1, X_scaled.shape[1], 1)
        probs = model.predict(X_cnn, verbose=0).flatten()
        preds = (probs >= 0.5).astype(int)
        confs = np.where(preds == 1, probs, 1 - probs)
    elif model_name == 'SVM':
        preds = model.predict(X_scaled)
        prob_all = model.predict_proba(X_scaled)
        confs = np.array([prob_all[i, p] for i, p in enumerate(preds)])
    else:  # Random Forest
        preds = model.predict(X_scaled)
        prob_all = model.predict_proba(X_scaled)
        confs = np.array([prob_all[i, p] for i, p in enumerate(preds)])

    results['Prediction'] = preds
    results['Diagnosis'] = results['Prediction'].map({0: 'Healthy', 1: "Parkinson's"})
    results['Confidence'] = np.round(confs * 100, 2)

    return results


def get_dataset_info():
    """Get basic info about the training dataset."""
    data_path = os.path.join(BASE_DIR, 'data', 'parkinsons.data')
    df = pd.read_csv(data_path)
    feature_cols = [c for c in df.columns if c not in ['name', 'status']]

    return {
        'total_samples': len(df),
        'pd_count': int((df['status'] == 1).sum()),
        'healthy_count': int((df['status'] == 0).sum()),
        'num_features': len(feature_cols),
        'feature_names': feature_cols,
        'dataframe': df
    }


# Feature descriptions for the UI
FEATURE_INFO = {
    'MDVP:Fo(Hz)': {'desc': 'Average vocal fundamental frequency', 'min': 80.0, 'max': 270.0, 'default': 150.0},
    'MDVP:Fhi(Hz)': {'desc': 'Maximum vocal fundamental frequency', 'min': 100.0, 'max': 600.0, 'default': 200.0},
    'MDVP:Flo(Hz)': {'desc': 'Minimum vocal fundamental frequency', 'min': 60.0, 'max': 240.0, 'default': 120.0},
    'MDVP:Jitter(%)': {'desc': 'Jitter percentage', 'min': 0.001, 'max': 0.035, 'default': 0.006},
    'MDVP:Jitter(Abs)': {'desc': 'Absolute jitter (microseconds)', 'min': 0.000005, 'max': 0.00026, 'default': 0.00004},
    'MDVP:RAP': {'desc': 'Relative amplitude perturbation', 'min': 0.0005, 'max': 0.022, 'default': 0.003},
    'MDVP:PPQ': {'desc': 'Five-point period perturbation quotient', 'min': 0.0005, 'max': 0.02, 'default': 0.003},
    'Jitter:DDP': {'desc': 'Average absolute difference of differences', 'min': 0.001, 'max': 0.065, 'default': 0.009},
    'MDVP:Shimmer': {'desc': 'Local shimmer', 'min': 0.009, 'max': 0.12, 'default': 0.03},
    'MDVP:Shimmer(dB)': {'desc': 'Shimmer in decibels', 'min': 0.08, 'max': 1.3, 'default': 0.28},
    'Shimmer:APQ3': {'desc': 'Three-point amplitude perturbation quotient', 'min': 0.004, 'max': 0.06, 'default': 0.015},
    'Shimmer:APQ5': {'desc': 'Five-point amplitude perturbation quotient', 'min': 0.005, 'max': 0.08, 'default': 0.02},
    'MDVP:APQ': {'desc': 'Amplitude perturbation quotient', 'min': 0.007, 'max': 0.14, 'default': 0.024},
    'Shimmer:DDA': {'desc': 'Average absolute difference of consecutive differences', 'min': 0.01, 'max': 0.17, 'default': 0.046},
    'NHR': {'desc': 'Noise-to-harmonics ratio', 'min': 0.0006, 'max': 0.32, 'default': 0.025},
    'HNR': {'desc': 'Harmonics-to-noise ratio', 'min': 8.0, 'max': 34.0, 'default': 22.0},
    'RPDE': {'desc': 'Recurrence period density entropy', 'min': 0.25, 'max': 0.69, 'default': 0.5},
    'DFA': {'desc': 'Signal fractal scaling exponent', 'min': 0.57, 'max': 0.83, 'default': 0.72},
    'spread1': {'desc': 'Nonlinear dynamical complexity measure 1', 'min': -7.96, 'max': -2.43, 'default': -5.68},
    'spread2': {'desc': 'Nonlinear dynamical complexity measure 2', 'min': 0.006, 'max': 0.45, 'default': 0.23},
    'D2': {'desc': 'Correlation dimension', 'min': 1.42, 'max': 3.67, 'default': 2.38},
    'PPE': {'desc': 'Pitch period entropy', 'min': 0.04, 'max': 0.53, 'default': 0.21},
}
