#!/usr/bin/env python3
"""
Test Position-Only Prediction
Tests the position-only prediction with existing data.
"""

import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import extract_quaternion_only_features

def test_position_prediction():
    """Test position-only prediction with sample data."""
    print("🧪 Testing Position-Only Prediction")
    print("=" * 40)
    
    # Load existing data
    data_file = "data/new_fixed_data.npz"
    if not os.path.exists(data_file):
        print(f"❌ Data file {data_file} not found!")
        return
    
    print(f"📂 Loading data from: {data_file}")
    data = np.load(data_file)
    
    # Get available classes
    classes = []
    for key in data.keys():
        if key.endswith('_quaternion'):
            class_name = key.replace('_quaternion', '')
            classes.append(class_name)
    
    print(f"📊 Available classes: {classes}")
    
    # Test with a few samples from each class
    window_size = 100
    
    for class_name in classes:
        print(f"\n🔍 Testing class: {class_name}")
        
        # Get quaternion data for this class
        quaternion_key = f"{class_name}_quaternion"
        if quaternion_key not in data:
            print(f"  No quaternion data found for {class_name}")
            continue
        
        quaternion_data = data[quaternion_key]
        print(f"  Quaternion data shape: {quaternion_data.shape}")
        
        if len(quaternion_data) == 0:
            print(f"  No quaternion data for {class_name}")
            continue
        
        # Test first sample
        if len(quaternion_data) >= window_size:
            quaternion_window = quaternion_data[:window_size]
            
            # Extract position-only features
            features = extract_quaternion_only_features(quaternion_window)
            
            print(f"  Sample shape: {quaternion_window.shape}")
            print(f"  Feature shape: {features.shape}")
            print(f"  Feature range: [{np.min(features):.4f}, {np.max(features):.4f}]")
            print(f"  Feature mean: {np.mean(features):.4f}")
            print(f"  Feature std: {np.std(features):.4f}")
            
            # Calculate quaternion variance (movement indicator)
            quaternion_variance = np.var(quaternion_window, axis=0)
            print(f"  Quaternion variance: {np.mean(quaternion_variance):.4f}")
            
            if class_name == 'IDLE':
                print(f"  → Expected: Low variance (idle state)")
            else:
                print(f"  → Expected: Higher variance (gesture)")
        else:
            print(f"  Not enough data for window size {window_size}")
    
    print("\n✅ Position-only feature extraction test complete!")
    print("📊 Features look good for position-only model training.")

def test_position_model_loading():
    """Test loading the position-only model."""
    print("\n🔧 Testing Position-Only Model Loading")
    print("=" * 40)
    
    model_path = "models/position_only_model.h5"
    le_path = "models/position_only_label_encoder.pkl"
    
    if not os.path.exists(model_path):
        print(f"❌ Position-only model not found: {model_path}")
        print("💡 Run train_position_only.py first to create the model")
        return False
    
    if not os.path.exists(le_path):
        print(f"❌ Position-only label encoder not found: {le_path}")
        return False
    
    try:
        from tensorflow import keras
        import pickle
        
        # Load model
        model = keras.models.load_model(model_path)
        with open(le_path, 'rb') as f:
            le = pickle.load(f)
        
        print(f"✅ Model loaded successfully")
        print(f"📋 Model input shape: {model.input_shape}")
        print(f"📋 Model output shape: {model.output_shape}")
        print(f"🏷️  Classes: {le.classes_}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def test_position_prediction_with_model():
    """Test actual prediction with the position-only model."""
    print("\n🎯 Testing Position-Only Prediction with Model")
    print("=" * 50)
    
    # Load model
    model_path = "models/position_only_model.h5"
    le_path = "models/position_only_label_encoder.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(le_path):
        print("❌ Position-only model not found. Train it first.")
        return
    
    try:
        from tensorflow import keras
        import pickle
        
        model = keras.models.load_model(model_path)
        with open(le_path, 'rb') as f:
            le = pickle.load(f)
        
        # Load test data
        data_file = "data/new_fixed_data.npz"
        data = np.load(data_file)
        
        # Get available classes
        classes = []
        for key in data.keys():
            if key.endswith('_quaternion'):
                class_name = key.replace('_quaternion', '')
                classes.append(class_name)
        
        # Test with one sample from each class
        window_size = 100
        
        for class_name in classes:
            quaternion_key = f"{class_name}_quaternion"
            if quaternion_key not in data:
                continue
            
            quaternion_data = data[quaternion_key]
            if len(quaternion_data) < window_size:
                continue
            
            quaternion_window = quaternion_data[:window_size]
            
            # Make prediction
            features = extract_quaternion_only_features(quaternion_window)
            features_batch = features[np.newaxis, ...]
            
            pred = model.predict(features_batch, verbose=0)
            predicted_class = np.argmax(pred)
            confidence = np.max(pred)
            predicted_label = le.inverse_transform([predicted_class])[0]
            
            print(f"🔍 {class_name}:")
            print(f"  True label: {class_name}")
            print(f"  Predicted: {predicted_label}")
            print(f"  Confidence: {confidence:.3f}")
            print(f"  Correct: {'✅' if predicted_label == class_name else '❌'}")
            
            # Show all probabilities
            print(f"  All probabilities:")
            for i, (label, prob) in enumerate(zip(le.classes_, pred[0])):
                marker = "★" if label == predicted_label else " "
                print(f"    {marker} {label}: {prob:.3f}")
            print()
        
        print("✅ Position-only prediction test complete!")
        
    except Exception as e:
        print(f"❌ Error during prediction test: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run all tests."""
    print("🎯 Position-Only Model Testing Suite")
    print("=" * 50)
    
    # Test 1: Feature extraction
    test_position_prediction()
    
    # Test 2: Model loading
    model_loaded = test_position_model_loading()
    
    # Test 3: Actual prediction (if model exists)
    if model_loaded:
        test_position_prediction_with_model()
    else:
        print("\n💡 To test prediction, first run:")
        print("   python train_position_only.py")
    
    print("\n🎉 All tests complete!")

if __name__ == "__main__":
    main() 