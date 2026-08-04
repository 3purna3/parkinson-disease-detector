"""
Parkinson's Disease Detection — Model Training Pipeline
Trains SVM, Random Forest, and 1D-CNN models on the UCI Parkinsons dataset.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ─── Paths ───────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'parkinsons.data')
SAVE_DIR = os.path.join(BASE_DIR, 'saved')
os.makedirs(SAVE_DIR, exist_ok=True)

# ─── 1. Load & Preprocess Data ──────────────────────
print("=" * 60)
print("  PARKINSON'S DISEASE DETECTION — MODEL TRAINING")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"\n📊 Dataset loaded: {df.shape[0]} samples, {df.shape[1]} columns")
print(f"   PD patients: {(df['status'] == 1).sum()}, Healthy: {(df['status'] == 0).sum()}")

# Feature columns (drop 'name' and 'status')
FEATURE_COLS = [c for c in df.columns if c not in ['name', 'status']]
X = df[FEATURE_COLS].values
y = df['status'].values

# Save feature column names
joblib.dump(FEATURE_COLS, os.path.join(SAVE_DIR, 'feature_columns.pkl'))

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, os.path.join(SAVE_DIR, 'scaler.pkl'))
print("   ✅ Scaler fitted and saved")

# ─── Helper: evaluate model ─────────────────────────
def evaluate_model(name, model, X_test_s, y_test):
    y_pred = model.predict(X_test_s)
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test_s)[:, 1]
    elif hasattr(model, 'decision_function'):
        y_prob = model.decision_function(X_test_s)
        # Normalize to 0-1 for ROC
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-8)
    else:
        y_prob = y_pred.astype(float)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    print(f"\n{'─' * 40}")
    print(f"  {name}")
    print(f"{'─' * 40}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  ROC AUC  : {roc_auc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Healthy', 'Parkinson'])}")

    return {
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1_score': round(f1, 4),
        'roc_auc': round(roc_auc, 4),
        'confusion_matrix': cm,
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist()
    }

# --- 2. Train SVM with GridSearchCV -------------------
print("\n Training SVM with GridSearch (RBF kernel)...")
svm_param_grid = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': ['scale', 'auto', 0.01, 0.001, 0.0001],
    'kernel': ['rbf']
}
svm_grid = GridSearchCV(
    SVC(probability=True, random_state=42),
    svm_param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='accuracy',
    n_jobs=-1,
    verbose=0
)
svm_grid.fit(X_train_scaled, y_train)
svm_model = svm_grid.best_estimator_
print(f"  Best SVM params: {svm_grid.best_params_}")
print(f"  Best CV Accuracy: {svm_grid.best_score_:.4f}")
svm_metrics = evaluate_model("Support Vector Machine (SVM)", svm_model, X_test_scaled, y_test)
joblib.dump(svm_model, os.path.join(SAVE_DIR, 'svm_model.pkl'))
print("  ✅ SVM model saved")

# --- 3. Train Random Forest with GridSearchCV ---------
print("\n Training Random Forest with GridSearch...")
rf_param_grid = {
    'n_estimators': [200, 500, 800],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    rf_param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='accuracy',
    n_jobs=-1,
    verbose=0
)
rf_grid.fit(X_train_scaled, y_train)
rf_model = rf_grid.best_estimator_
print(f"  Best RF params: {rf_grid.best_params_}")
print(f"  Best CV Accuracy: {rf_grid.best_score_:.4f}")
rf_metrics = evaluate_model("Random Forest", rf_model, X_test_scaled, y_test)
joblib.dump(rf_model, os.path.join(SAVE_DIR, 'rf_model.pkl'))

# Feature importance
feature_importance = dict(zip(FEATURE_COLS, rf_model.feature_importances_.tolist()))
rf_metrics['feature_importance'] = feature_importance
print("  ✅ Random Forest model saved")

# --- 4. Train 1D-CNN (Enhanced Architecture) ----------
print("\n Training 1D-CNN (Enhanced Deep Architecture)...")
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

tf.random.set_seed(42)

# Reshape for Conv1D: (samples, features, 1)
X_train_cnn = X_train_scaled.reshape(-1, X_train_scaled.shape[1], 1)
X_test_cnn = X_test_scaled.reshape(-1, X_test_scaled.shape[1], 1)

cnn_model = keras.Sequential([
    layers.Conv1D(64, kernel_size=3, activation='relu', padding='same',
                  input_shape=(X_train_cnn.shape[1], 1)),
    layers.BatchNormalization(),
    layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv1D(256, kernel_size=3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.GlobalAveragePooling1D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

cnn_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Callbacks for smarter training
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)

history = cnn_model.fit(
    X_train_cnn, y_train,
    epochs=200,
    batch_size=16,
    validation_split=0.15,
    callbacks=[early_stop, reduce_lr],
    verbose=0
)

# Evaluate CNN
y_pred_prob = cnn_model.predict(X_test_cnn, verbose=0).flatten()
y_pred_cnn = (y_pred_prob >= 0.5).astype(int)

acc = accuracy_score(y_test, y_pred_cnn)
prec = precision_score(y_test, y_pred_cnn, zero_division=0)
rec = recall_score(y_test, y_pred_cnn, zero_division=0)
f1 = f1_score(y_test, y_pred_cnn, zero_division=0)
cm = confusion_matrix(y_test, y_pred_cnn).tolist()
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

cnn_metrics = {
    'accuracy': round(acc, 4),
    'precision': round(prec, 4),
    'recall': round(rec, 4),
    'f1_score': round(f1, 4),
    'roc_auc': round(roc_auc, 4),
    'confusion_matrix': cm,
    'fpr': fpr.tolist(),
    'tpr': tpr.tolist()
}

print(f"\n{'─' * 40}")
print(f"  1D Convolutional Neural Network (CNN)")
print(f"{'─' * 40}")
print(f"  Accuracy : {acc:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall   : {rec:.4f}")
print(f"  F1 Score : {f1:.4f}")
print(f"  ROC AUC  : {roc_auc:.4f}")
print(f"\n{classification_report(y_test, y_pred_cnn, target_names=['Healthy', 'Parkinson'])}")

cnn_model.save(os.path.join(SAVE_DIR, 'cnn_model.h5'))
print("  ✅ CNN model saved")

# Save training history
train_history = {
    'loss': [float(v) for v in history.history['loss']],
    'accuracy': [float(v) for v in history.history['accuracy']],
    'val_loss': [float(v) for v in history.history['val_loss']],
    'val_accuracy': [float(v) for v in history.history['val_accuracy']],
}

# ─── 5. Save All Metrics ────────────────────────────
all_metrics = {
    'SVM': svm_metrics,
    'Random Forest': rf_metrics,
    'CNN': cnn_metrics,
    'training_history': train_history
}

with open(os.path.join(SAVE_DIR, 'metrics.json'), 'w') as f:
    json.dump(all_metrics, f, indent=2)

print("\n" + "=" * 60)
print("  ✅ ALL MODELS TRAINED AND SAVED SUCCESSFULLY!")
print("=" * 60)
print(f"\n  📁 Saved to: {SAVE_DIR}")
print(f"     • svm_model.pkl")
print(f"     • rf_model.pkl")
print(f"     • cnn_model.h5")
print(f"     • scaler.pkl")
print(f"     • feature_columns.pkl")
print(f"     • metrics.json")
print()
