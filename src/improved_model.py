#!/usr/bin/env python3
"""
Improved model that uses the 5 better approaches for gesture recognition
"""
import numpy as np
import os
import pickle
import json
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from .utils import get_output_dir
from .improved_features import extract_all_improved_features, analyze_feature_importance
import glob

# GPU Configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU acceleration enabled: {len(gpus)} GPU(s) available")
    except RuntimeError as e:
        print(f"⚠️ GPU configuration error: {e}")
else:
    print("ℹ️ No GPU detected, using CPU")

def prepare_improved_features(emg_data_list, quaternion_data_list, labels):
    """
    Prepare improved features from raw EMG and quaternion data
    """
    print("Extracting improved features...")
    
    features_list = []
    processed_labels = []
    
    for i, (emg_data, quaternion_data) in enumerate(zip(emg_data_list, quaternion_data_list)):
        try:
            # Handle different data formats
            if isinstance(emg_data, (int, float)):
                print(f"⚠️ Skipping sample {i} - emg_data is scalar: {type(emg_data)}")
                continue
                
            if isinstance(quaternion_data, (int, float)):
                print(f"⚠️ Skipping sample {i} - quaternion_data is scalar: {type(quaternion_data)}")
                continue
            
            # Ensure data is numpy arrays with correct shape
            if not hasattr(emg_data, 'shape'):
                print(f"⚠️ Skipping sample {i} - emg_data has no shape attribute")
                continue
                
            if not hasattr(quaternion_data, 'shape'):
                print(f"⚠️ Skipping sample {i} - quaternion_data has no shape attribute")
                continue
            
            # Check expected shapes
            if len(emg_data.shape) != 2 or emg_data.shape[1] != 8:
                print(f"⚠️ Skipping sample {i} - emg_data shape {emg_data.shape} not (N, 8)")
                continue
                
            if len(quaternion_data.shape) != 2 or quaternion_data.shape[1] != 4:
                print(f"⚠️ Skipping sample {i} - quaternion_data shape {quaternion_data.shape} not (N, 4)")
                continue
            
            # Extract improved features
            features = extract_all_improved_features(emg_data, quaternion_data)
            
            # Check for invalid features
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                print(f"⚠️ Skipping sample {i} due to invalid features")
                continue
                
            features_list.append(features)
            processed_labels.append(labels[i])
            
        except Exception as e:
            print(f"⚠️ Error processing sample {i}: {e}")
            continue
    
    if not features_list:
        raise ValueError("No valid features extracted!")
    
    X = np.array(features_list)
    y = np.array(processed_labels)
    
    print(f"✅ Extracted {len(X)} feature vectors with {X.shape[1]} features each")
    print(f"Feature statistics:")
    print(f"  Mean: {np.mean(X):.4f}")
    print(f"  Std: {np.std(X):.4f}")
    print(f"  Range: [{np.min(X):.4f}, {np.max(X):.4f}]")
    
    return X, y

def train_improved_model(emg_data_list, quaternion_data_list, labels, model_type='ensemble'):
    """
    Train an improved model using the 5 better approaches
    """
    output_dir = get_output_dir()
    
    # Prepare improved features
    X, y = prepare_improved_features(emg_data_list, quaternion_data_list, labels)
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nTraining {model_type} model...")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Classes: {le.classes_}")
    
    if model_type == 'ensemble':
        # Ensemble of multiple models
        models = {
            'random_forest': RandomForestClassifier(n_estimators=200, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=200, random_state=42),
            'svm': SVC(kernel='rbf', probability=True, random_state=42)
        }
        
        trained_models = {}
        for name, model in models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"{name} accuracy: {accuracy:.4f}")
            trained_models[name] = model
        
        # Use the best model
        best_model_name = max(trained_models.keys(), 
                            key=lambda x: accuracy_score(y_test, trained_models[x].predict(X_test_scaled)))
        best_model = trained_models[best_model_name]
        print(f"\nBest model: {best_model_name}")
        
    elif model_type == 'neural_network':
        # Neural network approach
        model = Sequential([
            Input(shape=(X_train.shape[1],)),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(len(le.classes_), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7)
        ]
        
        history = model.fit(
            X_train_scaled, y_train,
            validation_data=(X_test_scaled, y_test),
            epochs=200,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        best_model = model
    
    # Evaluate the model
    if model_type == 'ensemble':
        y_pred = best_model.predict(X_test_scaled)
        y_pred_proba = best_model.predict_proba(X_test_scaled)
    else:
        y_pred = np.argmax(best_model.predict(X_test_scaled), axis=1)
        y_pred_proba = best_model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nFinal accuracy: {accuracy:.4f}")
    
    # Print detailed results
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Feature importance analysis
    if model_type == 'ensemble':
        print("\nFeature Importance Analysis:")
        importance, mi_scores = analyze_feature_importance(X_train_scaled, y_train)
        print(f"Top 10 most important features:")
        top_indices = np.argsort(importance)[-10:][::-1]
        for i, idx in enumerate(top_indices):
            print(f"  {i+1}. Feature {idx}: {importance[idx]:.4f}")
    
    # Save the model and components
    model_path = os.path.join(output_dir, "improved_handwriting_model.pkl")
    scaler_path = os.path.join(output_dir, "improved_scaler.pkl")
    le_path = os.path.join(output_dir, "improved_label_encoder.pkl")
    
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    with open(le_path, 'wb') as f:
        pickle.dump(le, f)
    
    print(f"\n✅ Model saved to {output_dir}")
    print(f"  Model: {model_path}")
    print(f"  Scaler: {scaler_path}")
    print(f"  Label Encoder: {le_path}")
    
    return best_model, scaler, le

def load_improved_model():
    """
    Load the improved model and components
    """
    output_dir = get_output_dir()
    
    model_path = os.path.join(output_dir, "improved_handwriting_model.pkl")
    scaler_path = os.path.join(output_dir, "improved_scaler.pkl")
    le_path = os.path.join(output_dir, "improved_label_encoder.pkl")
    
    if not all(os.path.exists(p) for p in [model_path, scaler_path, le_path]):
        raise FileNotFoundError("Improved model files not found. Please train the model first.")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(le_path, 'rb') as f:
        le = pickle.load(f)
    
    print(f"✅ Loaded improved model from {output_dir}")
    return model, scaler, le

def predict_with_improved_model(emg_data, quaternion_data, model, scaler, le):
    """
    Make prediction using the improved model
    """
    # Extract improved features
    features = extract_all_improved_features(emg_data, quaternion_data)
    
    # Check for invalid features
    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
        raise ValueError("Invalid features detected")
    
    # Scale features
    features_scaled = scaler.transform(features.reshape(1, -1))
    
    # Make prediction
    if hasattr(model, 'predict_proba'):
        # Ensemble model
        prediction_proba = model.predict_proba(features_scaled)[0]
        predicted_class = model.predict(features_scaled)[0]
    else:
        # Neural network
        prediction_proba = model.predict(features_scaled)[0]
        predicted_class = np.argmax(prediction_proba)
    
    # Decode prediction
    predicted_label = le.inverse_transform([predicted_class])[0]
    confidence = np.max(prediction_proba)
    
    return predicted_label, confidence, prediction_proba

def test_improved_model():
    """
    Test the improved model with sample data
    """
    print("=== Testing Improved Model ===")
    
    try:
        model, scaler, le = load_improved_model()
        print(f"✅ Model loaded successfully")
        print(f"Classes: {le.classes_}")
        
        # Test with sample data
        print("\nTesting with sample data...")
        
        # Generate test data
        test_emg = np.random.randn(100, 8) * 10  # Some variance
        test_quaternion = np.random.randn(100, 4) * 0.1
        
        predicted_label, confidence, probabilities = predict_with_improved_model(
            test_emg, test_quaternion, model, scaler, le
        )
        
        print(f"Prediction: {predicted_label}")
        print(f"Confidence: {confidence:.4f}")
        print(f"All probabilities:")
        for i, (class_name, prob) in enumerate(zip(le.classes_, probabilities)):
            print(f"  {class_name}: {prob:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        return False

if __name__ == "__main__":
    test_improved_model() 