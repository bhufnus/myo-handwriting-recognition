#!/usr/bin/env python3
"""
Test the preprocessing fix for static data
"""
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import preprocess_emg
from src.model import load_trained_model

def test_preprocessing_fix():
    """Test that preprocessing handles static data correctly"""
    print("=== Testing Preprocessing Fix ===")
    
    # Test 1: Static/zero data
    print("Test 1: Static EMG data")
    static_emg = np.zeros((100, 8))
    try:
        processed = preprocess_emg(static_emg)
        print(f"  Preprocessing successful: {processed.shape}")
        print(f"  Contains NaN: {np.any(np.isnan(processed))}")
        print(f"  All zeros: {np.all(processed == 0)}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Test 2: Very low variance data
    print("\nTest 2: Low variance EMG data")
    low_var_emg = np.random.randn(100, 8) * 0.001  # Very small variance
    try:
        processed = preprocess_emg(low_var_emg)
        print(f"  Preprocessing successful: {processed.shape}")
        print(f"  Contains NaN: {np.any(np.isnan(processed))}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Test 3: Normal variance data
    print("\nTest 3: Normal variance EMG data")
    normal_emg = np.random.randn(100, 8) * 50
    try:
        processed = preprocess_emg(normal_emg)
        print(f"  Preprocessing successful: {processed.shape}")
        print(f"  Contains NaN: {np.any(np.isnan(processed))}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Test 4: Model prediction with static data
    print("\nTest 4: Model prediction with static data")
    try:
        model, le = load_trained_model()
        
        # Test with static data
        static_emg = np.zeros((100, 8))
        static_quaternion = np.array([[0, 0, 0, 1]] * 100)
        
        processed_emg = preprocess_emg(static_emg)
        X_test = np.concatenate([processed_emg, static_quaternion], axis=1)
        X_test_batch = X_test[np.newaxis, ...]
        
        prediction = model.predict(X_test_batch, verbose=0)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)
        predicted_label = le.inverse_transform([predicted_class])[0]
        
        print(f"  Static data prediction: {predicted_label} (confidence: {confidence:.3f})")
        print(f"  All probabilities:")
        for i, (label, prob) in enumerate(zip(le.classes_, prediction[0])):
            print(f"    {label}: {prob:.3f}")
            
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    test_preprocessing_fix() 