#!/usr/bin/env python3
"""
Debug real Myo data to understand the idle vs A distinction
"""
import sys
import os
import numpy as np
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.myo_interface import init_myo, collect_data
from src.preprocessing import preprocess_emg
from src.model import load_trained_model

def capture_real_data():
    """Capture real Myo data for analysis"""
    print("=== Real Myo Data Analysis ===")
    print("This will capture 2 seconds of data for analysis.")
    print("Keep your arm completely still (idle) for the first test.")
    print()
    
    try:
        init_myo()
        model, le = load_trained_model()
        
        # Test 1: Capture idle data
        print("Test 1: Capturing IDLE data (keep arm still)")
        input("Press Enter when ready to capture idle data...")
        
        idle_emg, idle_accel, idle_gyro, idle_labels = collect_data("IDLE", duration_ms=2000)
        
        if len(idle_emg) > 0:
            print(f"Captured {len(idle_emg)} EMG samples for idle")
            
            # Analyze idle data
            idle_emg_array = np.array(idle_emg)
            print(f"Idle EMG - mean: {np.mean(idle_emg_array):.2f}, std: {np.std(idle_emg_array):.2f}")
            print(f"Idle EMG - min: {np.min(idle_emg_array):.2f}, max: {np.max(idle_emg_array):.2f}")
            
            # Test prediction on idle data
            if len(idle_emg) >= 100:
                idle_window = idle_emg_array[:100]  # Take first 100 samples
                idle_quaternion = np.array([[0, 0, 0, 1]] * 100)  # Assume no rotation
                
                processed_idle = preprocess_emg(idle_window)
                X_idle = np.concatenate([processed_idle, idle_quaternion], axis=1)
                X_idle_batch = X_idle[np.newaxis, ...]
                
                prediction = model.predict(X_idle_batch, verbose=0)
                predicted_class = np.argmax(prediction)
                confidence = np.max(prediction)
                predicted_label = le.inverse_transform([predicted_class])[0]
                
                print(f"Model prediction for idle data: {predicted_label} (confidence: {confidence:.3f})")
                print(f"All probabilities: A={prediction[0][0]:.3f}, B={prediction[0][1]:.3f}, C={prediction[0][2]:.3f}, IDLE={prediction[0][3]:.3f}, NOISE={prediction[0][4]:.3f}")
        
        # Test 2: Capture A data
        print("\nTest 2: Capturing A data (write letter A)")
        input("Press Enter when ready to capture A data...")
        
        a_emg, a_accel, a_gyro, a_labels = collect_data("A", duration_ms=2000)
        
        if len(a_emg) > 0:
            print(f"Captured {len(a_emg)} EMG samples for A")
            
            # Analyze A data
            a_emg_array = np.array(a_emg)
            print(f"A EMG - mean: {np.mean(a_emg_array):.2f}, std: {np.std(a_emg_array):.2f}")
            print(f"A EMG - min: {np.min(a_emg_array):.2f}, max: {np.max(a_emg_array):.2f}")
            
            # Test prediction on A data
            if len(a_emg) >= 100:
                a_window = a_emg_array[:100]  # Take first 100 samples
                a_quaternion = np.array([[0, 0, 0, 1]] * 100)  # Assume no rotation
                
                processed_a = preprocess_emg(a_window)
                X_a = np.concatenate([processed_a, a_quaternion], axis=1)
                X_a_batch = X_a[np.newaxis, ...]
                
                prediction = model.predict(X_a_batch, verbose=0)
                predicted_class = np.argmax(prediction)
                confidence = np.max(prediction)
                predicted_label = le.inverse_transform([predicted_class])[0]
                
                print(f"Model prediction for A data: {predicted_label} (confidence: {confidence:.3f})")
                print(f"All probabilities: A={prediction[0][0]:.3f}, B={prediction[0][1]:.3f}, C={prediction[0][2]:.3f}, IDLE={prediction[0][3]:.3f}, NOISE={prediction[0][4]:.3f}")
        
        # Compare the data
        if len(idle_emg) > 0 and len(a_emg) > 0:
            print("\n=== Data Comparison ===")
            idle_std = np.std(idle_emg_array)
            a_std = np.std(a_emg_array)
            
            print(f"Idle EMG std: {idle_std:.2f}")
            print(f"A EMG std: {a_std:.2f}")
            print(f"Ratio (A/Idle): {a_std/idle_std:.2f}")
            
            if a_std > idle_std * 1.5:
                print("✅ A data has higher variance than idle (as expected)")
            else:
                print("⚠️ A data variance is similar to idle (unexpected)")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    capture_real_data() 