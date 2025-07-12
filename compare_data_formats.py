#!/usr/bin/env python3
"""
Compare data formats between diagnostic tests and real-time prediction
"""
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model import load_trained_model
from src.preprocessing import preprocess_emg

def compare_formats():
    """Compare diagnostic vs real-time data formats"""
    print("=== Data Format Comparison ===")
    
    try:
        model, le = load_trained_model()
        
        # Test 1: Diagnostic format (static data)
        print("Test 1: Diagnostic format (static data)")
        static_emg = np.zeros((100, 8))
        static_quaternion = np.array([[0, 0, 0, 1]] * 100)
        
        processed_emg = preprocess_emg(static_emg)
        X_test = np.concatenate([processed_emg, static_quaternion], axis=1)
        X_test_batch = X_test[np.newaxis, ...]
        
        prediction = model.predict(X_test_batch, verbose=0)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)
        predicted_label = le.inverse_transform([predicted_class])[0]
        
        print(f"  Diagnostic result: {predicted_label} (confidence: {confidence:.3f})")
        print(f"  All probabilities: A={prediction[0][0]:.3f}, B={prediction[0][1]:.3f}, C={prediction[0][2]:.3f}, IDLE={prediction[0][3]:.3f}, NOISE={prediction[0][4]:.3f}")
        
        # Test 2: Simulate real-time data (what the Myo actually sends)
        print("\nTest 2: Real-time format (simulated Myo data)")
        
        # Simulate what the Myo might actually send during idle
        # Real Myo data might have small noise even when "idle"
        real_time_emg = np.random.randn(100, 8) * 0.1  # Small noise
        real_time_quaternion = np.random.randn(100, 4) * 0.01  # Small noise
        
        processed_emg_rt = preprocess_emg(real_time_emg)
        X_test_rt = np.concatenate([processed_emg_rt, real_time_quaternion], axis=1)
        X_test_batch_rt = X_test_rt[np.newaxis, ...]
        
        prediction_rt = model.predict(X_test_batch_rt, verbose=0)
        predicted_class_rt = np.argmax(prediction_rt)
        confidence_rt = np.max(prediction_rt)
        predicted_label_rt = le.inverse_transform([predicted_class_rt])[0]
        
        print(f"  Real-time result: {predicted_label_rt} (confidence: {confidence_rt:.3f})")
        print(f"  All probabilities: A={prediction_rt[0][0]:.3f}, B={prediction_rt[0][1]:.3f}, C={prediction_rt[0][2]:.3f}, IDLE={prediction_rt[0][3]:.3f}, NOISE={prediction_rt[0][4]:.3f}")
        
        # Test 3: Check data characteristics
        print("\nTest 3: Data characteristics comparison")
        print(f"  Diagnostic EMG - mean: {np.mean(static_emg):.4f}, std: {np.std(static_emg):.4f}")
        print(f"  Real-time EMG - mean: {np.mean(real_time_emg):.4f}, std: {np.std(real_time_emg):.4f}")
        print(f"  Diagnostic quaternion - mean: {np.mean(static_quaternion):.4f}, std: {np.std(static_quaternion):.4f}")
        print(f"  Real-time quaternion - mean: {np.mean(real_time_quaternion):.4f}, std: {np.std(real_time_quaternion):.4f}")
        
        # Test 4: Check if the issue is in preprocessing
        print("\nTest 4: Preprocessing comparison")
        print(f"  Diagnostic processed EMG - mean: {np.mean(processed_emg):.4f}, std: {np.std(processed_emg):.4f}")
        print(f"  Real-time processed EMG - mean: {np.mean(processed_emg_rt):.4f}, std: {np.std(processed_emg_rt):.4f}")
        
        # Test 5: Test with actual Myo-like data ranges
        print("\nTest 5: Myo-like data ranges")
        # Myo EMG typically ranges from -128 to 127
        myo_like_emg = np.random.randint(-128, 128, (100, 8)).astype(np.float32)
        myo_like_quaternion = np.random.randn(100, 4) * 0.1
        
        processed_emg_myo = preprocess_emg(myo_like_emg)
        X_test_myo = np.concatenate([processed_emg_myo, myo_like_quaternion], axis=1)
        X_test_batch_myo = X_test_myo[np.newaxis, ...]
        
        prediction_myo = model.predict(X_test_batch_myo, verbose=0)
        predicted_class_myo = np.argmax(prediction_myo)
        confidence_myo = np.max(prediction_myo)
        predicted_label_myo = le.inverse_transform([predicted_class_myo])[0]
        
        print(f"  Myo-like result: {predicted_label_myo} (confidence: {confidence_myo:.3f})")
        print(f"  All probabilities: A={prediction_myo[0][0]:.3f}, B={prediction_myo[0][1]:.3f}, C={prediction_myo[0][2]:.3f}, IDLE={prediction_myo[0][3]:.3f}, NOISE={prediction_myo[0][4]:.3f}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    compare_formats() 