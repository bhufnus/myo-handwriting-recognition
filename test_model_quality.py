#!/usr/bin/env python3
"""
Test script to check model quality and identify prediction issues
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import tensorflow as tf
import pickle

def test_model_predictions():
    """Test model predictions with synthetic data"""
    print("🧪 Testing Model Prediction Quality")
    print("=" * 50)
    
    # Try to load a model
    try:
        # Look for model files
        model_files = [f for f in os.listdir('.') if f.endswith('.h5')]
        if not model_files:
            print("❌ No model files found (.h5)")
            print("   Please train a model first or specify the model path")
            return
        
        model_file = model_files[0]  # Use first found model
        print(f"📁 Found model: {model_file}")
        
        # Load model
        model = tf.keras.models.load_model(model_file)
        print(f"✅ Model loaded successfully")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        
        # Load label encoder
        le_file = model_file.replace('.h5', '_labels.pkl')
        if os.path.exists(le_file):
            with open(le_file, 'rb') as f:
                le = pickle.load(f)
            print(f"✅ Label encoder loaded: {list(le.classes_)}")
        else:
            print(f"⚠️  No label encoder found: {le_file}")
            return
        
        # Test with synthetic data for each class
        print(f"\n🎯 Testing predictions with synthetic data:")
        
        window_size = model.input_shape[1]  # Get from model
        num_classes = len(le.classes_)
        
        for i, class_name in enumerate(le.classes_):
            # Create synthetic data that should be different for each class
            # Add class-specific patterns
            base_data = np.random.randn(window_size, 12) * 0.1
            class_pattern = np.sin(np.linspace(0, 2*np.pi*i, window_size)).reshape(-1, 1) * 0.5
            synthetic_data = base_data + class_pattern
            
            # Reshape for model input
            X_test = synthetic_data.reshape(1, window_size, 12)
            
            # Make prediction
            pred = model.predict(X_test, verbose=0)
            predicted_class = np.argmax(pred)
            confidence = np.max(pred)
            predicted_label = le.inverse_transform([predicted_class])[0]
            
            print(f"   Class {class_name} test data:")
            print(f"     Predicted: {predicted_label}")
            print(f"     Confidence: {confidence:.3f}")
            print(f"     Raw predictions: {pred.flatten()}")
            
        # Test with random data
        print(f"\n🎲 Testing with random data:")
        for i in range(5):
            random_data = np.random.randn(window_size, 12)
            X_test = random_data.reshape(1, window_size, 12)
            
            pred = model.predict(X_test, verbose=0)
            predicted_class = np.argmax(pred)
            confidence = np.max(pred)
            predicted_label = le.inverse_transform([predicted_class])[0]
            
            print(f"   Random test {i+1}: {predicted_label} (confidence: {confidence:.3f})")
        
        # Check if model is biased
        print(f"\n🔍 Bias Analysis:")
        all_predictions = []
        for i in range(100):
            random_data = np.random.randn(window_size, 12)
            X_test = random_data.reshape(1, window_size, 12)
            pred = model.predict(X_test, verbose=0)
            predicted_class = np.argmax(pred)
            all_predictions.append(predicted_class)
        
        unique_preds, pred_counts = np.unique(all_predictions, return_counts=True)
        print(f"   Predictions from 100 random inputs:")
        for pred_class, count in zip(unique_preds, pred_counts):
            pred_label = le.inverse_transform([pred_class])[0]
            print(f"     {pred_label}: {count} times ({count/100*100:.1f}%)")
        
        # Check for bias
        max_pred_count = max(pred_counts)
        if max_pred_count > 80:  # More than 80% predictions are the same
            print(f"   ⚠️  MODEL BIAS DETECTED: {max_pred_count}% predictions are the same class!")
            print(f"   This suggests the model needs retraining with better data")
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        import traceback
        traceback.print_exc()

def check_training_data():
    """Check if training data exists and its quality"""
    print(f"\n📊 Checking Training Data")
    print("=" * 30)
    
    # Look for data files
    data_files = [f for f in os.listdir('.') if f.endswith('.npz')]
    if not data_files:
        print("❌ No training data files found (.npz)")
        return
    
    for data_file in data_files[:3]:  # Check first 3 files
        print(f"📁 Checking: {data_file}")
        try:
            data = np.load(data_file, allow_pickle=True)
            
            # Check what's in the file
            print(f"   Keys: {list(data.keys())}")
            
            # Check class distribution
            for key in data.keys():
                if key.endswith('_emg'):
                    class_name = key.replace('_emg', '')
                    emg_data = data[key]
                    quaternion_key = f"{class_name}_quaternion"
                    
                    if quaternion_key in data:
                        quaternion_data = data[quaternion_key]
                        print(f"   {class_name}: {len(emg_data)} samples")
                        
                        # Check data quality
                        if len(emg_data) > 0:
                            sample_emg = emg_data[0]
                            sample_quat = quaternion_data[0]
                            print(f"     EMG shape: {sample_emg.shape}")
                            print(f"     Quaternion shape: {sample_quat.shape}")
                            
        except Exception as e:
            print(f"   ❌ Error reading {data_file}: {e}")

if __name__ == "__main__":
    test_model_predictions()
    check_training_data() 