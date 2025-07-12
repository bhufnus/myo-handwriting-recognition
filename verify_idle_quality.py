#!/usr/bin/env python3
"""
Verify IDLE data quality during collection
"""
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def analyze_sample_quality(emg_data, quaternion_data, class_name):
    """Analyze the quality of a single sample"""
    print(f"\n=== {class_name} Sample Analysis ===")
    
    # Convert to numpy arrays if needed
    if not isinstance(emg_data, np.ndarray):
        emg_data = np.array(emg_data)
    if not isinstance(quaternion_data, np.ndarray):
        quaternion_data = np.array(quaternion_data)
    
    # EMG analysis
    emg_std = np.std(emg_data, axis=0)
    emg_mean = np.mean(emg_data, axis=0)
    emg_range = np.ptp(emg_data, axis=0)
    
    print(f"EMG Statistics:")
    print(f"  Mean std across channels: {np.mean(emg_std):.2f}")
    print(f"  Max std across channels: {np.max(emg_std):.2f}")
    print(f"  Min std across channels: {np.min(emg_std):.2f}")
    print(f"  Overall range: {np.min(emg_data):.2f} to {np.max(emg_data):.2f}")
    
    # Quaternion analysis
    quat_std = np.std(quaternion_data, axis=0)
    quat_mean = np.mean(quaternion_data, axis=0)
    quat_range = np.ptp(quaternion_data, axis=0)
    
    print(f"Quaternion Statistics:")
    print(f"  Mean std across channels: {np.mean(quat_std):.4f}")
    print(f"  Max std across channels: {np.max(quat_std):.4f}")
    print(f"  Min std across channels: {np.min(quat_std):.4f}")
    print(f"  Overall range: {np.min(quaternion_data):.4f} to {np.max(quaternion_data):.4f}")
    
    # Quality assessment
    print(f"\nQuality Assessment:")
    
    if class_name == "IDLE":
        # For IDLE, we want low variance
        if np.mean(emg_std) < 2.0:
            print(f"  ✅ Good IDLE sample (EMG std: {np.mean(emg_std):.2f} < 2.0)")
        else:
            print(f"  ❌ Poor IDLE sample (EMG std: {np.mean(emg_std):.2f} >= 2.0)")
            print(f"     Keep your arm COMPLETELY still!")
        
        if np.mean(quat_std) < 0.01:
            print(f"  ✅ Good IDLE sample (Quat std: {np.mean(quat_std):.4f} < 0.01)")
        else:
            print(f"  ❌ Poor IDLE sample (Quat std: {np.mean(quat_std):.4f} >= 0.01)")
            print(f"     Don't move your arm at all!")
    else:
        # For gestures, we want high variance
        if np.mean(emg_std) > 5.0:
            print(f"  ✅ Good {class_name} sample (EMG std: {np.mean(emg_std):.2f} > 5.0)")
        else:
            print(f"  ❌ Poor {class_name} sample (EMG std: {np.mean(emg_std):.2f} <= 5.0)")
            print(f"     Make more movement!")
        
        if np.mean(quat_std) > 0.01:
            print(f"  ✅ Good {class_name} sample (Quat std: {np.mean(quat_std):.4f} > 0.01)")
        else:
            print(f"  ❌ Poor {class_name} sample (Quat std: {np.mean(quat_std):.4f} <= 0.01)")
            print(f"     Move your arm more!")

def main():
    print("=== IDLE Data Quality Verifier ===")
    print()
    print("This script helps verify that your IDLE data has low variance.")
    print("Use this during training to ensure proper data quality.")
    print()
    print("Target values:")
    print("- IDLE EMG std: < 2.0")
    print("- IDLE Quaternion std: < 0.01")
    print("- Gesture EMG std: > 5.0")
    print("- Gesture Quaternion std: > 0.01")
    print()
    print("To use this script:")
    print("1. Run your training GUI")
    print("2. After collecting a sample, call this function:")
    print("   from verify_idle_quality import analyze_sample_quality")
    print("   analyze_sample_quality(emg_data, quaternion_data, 'IDLE')")
    print()

if __name__ == "__main__":
    main() 