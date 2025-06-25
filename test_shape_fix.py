#!/usr/bin/env python3
"""
Test script to verify the shape mismatch fix
"""

import numpy as np

def test_shape_fix():
    """Test the shape mismatch fix logic"""
    print("🧪 Testing Shape Mismatch Fix")
    print("=" * 40)
    
    # Simulate the problem: different lengths
    emg_data = np.random.randn(101, 8)  # 101 timesteps
    quaternion_data = np.random.randn(100, 4)  # 100 timesteps
    
    print(f"Original shapes:")
    print(f"  EMG: {emg_data.shape}")
    print(f"  Quaternion: {quaternion_data.shape}")
    
    # Apply the fix
    min_len = min(len(emg_data), len(quaternion_data))
    emg_fixed = emg_data[:min_len]
    quaternion_fixed = quaternion_data[:min_len]
    
    print(f"\nAfter fix:")
    print(f"  EMG: {emg_fixed.shape}")
    print(f"  Quaternion: {quaternion_fixed.shape}")
    
    # Test concatenation
    try:
        combined = np.concatenate([emg_fixed, quaternion_fixed], axis=1)
        print(f"  Combined shape: {combined.shape}")
        print(f"  ✅ Concatenation successful!")
    except Exception as e:
        print(f"  ❌ Concatenation failed: {e}")
    
    # Test with window size
    window_size = 100
    if min_len >= window_size:
        emg_win = emg_fixed[:window_size]
        quaternion_win = quaternion_fixed[:window_size]
        
        print(f"\nWindow extraction:")
        print(f"  EMG window: {emg_win.shape}")
        print(f"  Quaternion window: {quaternion_win.shape}")
        
        # Test final sequence creation
        sequence = np.concatenate([emg_win, quaternion_win], axis=1)
        print(f"  Final sequence: {sequence.shape}")
        
        # Reshape for model
        model_input = sequence.reshape(1, sequence.shape[0], sequence.shape[1])
        print(f"  Model input: {model_input.shape}")
        print(f"  ✅ Ready for LSTM model!")
    else:
        print(f"  ⚠️  Data too short for window size {window_size}")

if __name__ == "__main__":
    test_shape_fix() 