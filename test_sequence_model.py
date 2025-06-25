#!/usr/bin/env python3
"""
Test script to verify sequence model training and prediction
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from src.preprocessing import preprocess_emg
from src.model import train_model
import tensorflow as tf

def create_test_data():
    """Create synthetic test data for sequence model"""
    print("Creating synthetic test data...")
    
    # Create synthetic EMG and quaternion data
    num_samples = 10
    window_size = 100
    num_classes = 3
    labels = ['A', 'B', 'C']
    
    all_windows = []
    all_labels = []
    
    for label_idx, label in enumerate(labels):
        print(f"Generating data for class {label}...")
        
        for sample in range(num_samples):
            # Create synthetic EMG data (8 channels)
            emg_data = np.random.randn(window_size, 8) * 10 + label_idx * 5
            
            # Create synthetic quaternion data (4 components)
            quaternion_data = np.random.randn(window_size, 4)
            # Normalize quaternions
            quaternion_data = quaternion_data / np.linalg.norm(quaternion_data, axis=1, keepdims=True)
            
            # Preprocess EMG
            emg_proc = preprocess_emg(emg_data)
            
            # Concatenate for sequence input: (window_size, 12)
            X_win = np.concatenate([emg_proc, quaternion_data], axis=1)
            
            all_windows.append(X_win)
            all_labels.append(label)
    
    X = np.array(all_windows)
    y = np.array(all_labels)
    
    print(f"Test data created:")
    print(f"  X shape: {X.shape} (samples, window_size, features)")
    print(f"  y shape: {y.shape}")
    print(f"  Classes: {np.unique(y)}")
    
    return X, y, labels

def test_model_training():
    """Test the sequence model training"""
    print("\n" + "="*50)
    print("Testing sequence model training...")
    
    # Create test data
    X, y, labels = create_test_data()
    
    try:
        # Train model
        print("Training LSTM sequence model...")
        model, le = train_model(X, y, labels)
        
        print("✅ Model training successful!")
        print(f"Model summary:")
        model.summary()
        
        # Test prediction
        print("\nTesting prediction...")
        test_sample = X[0:1]  # Take first sample
        prediction = model.predict(test_sample, verbose=0)
        predicted_class = np.argmax(prediction)
        predicted_label = le.inverse_transform([predicted_class])[0]
        confidence = np.max(prediction)
        
        print(f"✅ Prediction successful!")
        print(f"  Input shape: {test_sample.shape}")
        print(f"  Predicted: {predicted_label}")
        print(f"  Confidence: {confidence:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gui_compatibility():
    """Test that the data format is compatible with the GUI"""
    print("\n" + "="*50)
    print("Testing GUI compatibility...")
    
    # Simulate GUI data collection format
    window_size = 100
    
    # Simulate collected data (like in GUI)
    collected_emg = [np.random.randn(200, 8) for _ in range(3)]  # 3 samples, 200 timesteps each
    collected_quaternion = [np.random.randn(200, 4) for _ in range(3)]
    
    # Process like the GUI does
    all_windows = []
    all_labels = []
    
    for i, (emg_arr, quaternion_arr) in enumerate(zip(collected_emg, collected_quaternion)):
        min_len = min(len(emg_arr), len(quaternion_arr))
        if min_len >= window_size:
            for j in range(0, min_len - window_size + 1, window_size // 2):
                emg_win = emg_arr[j:j+window_size]
                quaternion_win = quaternion_arr[j:j+window_size]
                
                # Preprocess EMG
                emg_proc = preprocess_emg(emg_win)
                
                # Concatenate for sequence input
                X_win = np.concatenate([emg_proc, quaternion_win], axis=1)
                
                all_windows.append(X_win)
                all_labels.append('A')  # Test label
    
    if all_windows:
        X = np.array(all_windows)
        print(f"✅ GUI data processing successful!")
        print(f"  Generated {len(all_windows)} windows")
        print(f"  X shape: {X.shape}")
        print(f"  Expected shape: (n_samples, {window_size}, 12)")
        
        # Test prediction format
        test_sequence = X[0:1]  # (1, window_size, 12)
        print(f"  Test prediction input shape: {test_sequence.shape}")
        
        return True
    else:
        print("❌ GUI data processing failed!")
        return False

if __name__ == "__main__":
    print("🧪 Testing Sequence Model Implementation")
    print("="*50)
    
    # Test 1: Model training
    training_success = test_model_training()
    
    # Test 2: GUI compatibility
    gui_success = test_gui_compatibility()
    
    print("\n" + "="*50)
    print("Test Results:")
    print(f"  Model Training: {'✅ PASS' if training_success else '❌ FAIL'}")
    print(f"  GUI Compatibility: {'✅ PASS' if gui_success else '❌ FAIL'}")
    
    if training_success and gui_success:
        print("\n🎉 All tests passed! Your sequence model is ready to use.")
        print("You can now retrain your model with the updated GUI.")
    else:
        print("\n⚠️ Some tests failed. Please check the implementation.") 