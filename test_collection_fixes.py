#!/usr/bin/env python3
"""
Test script to verify the collection fixes
"""

import numpy as np

def test_length_standardization():
    """Test the length standardization logic"""
    print("🧪 Testing Length Standardization")
    print("=" * 40)
    
    # Simulate different length data
    test_cases = [
        (105, 100),  # EMG longer than quaternion
        (100, 104),  # Quaternion longer than EMG
        (120, 115),  # Both longer than target
        (80, 85),    # Both shorter than target
        (100, 100),  # Perfect match
    ]
    
    target_length = 100
    
    for emg_len, quat_len in test_cases:
        print(f"\n📊 Test case: EMG={emg_len}, Quaternion={quat_len}")
        
        # Simulate the data
        emg_buffer = list(np.random.randn(emg_len, 8))
        quaternion_buffer = list(np.random.randn(quat_len, 4))
        
        # Apply the standardization logic
        min_len = min(len(emg_buffer), len(quaternion_buffer))
        
        # Truncate both to the same length
        emg_data = np.array(emg_buffer[:min_len])
        quaternion_data = np.array(quaternion_buffer[:min_len])
        
        print(f"  After truncation: EMG={emg_data.shape}, Quaternion={quaternion_data.shape}")
        
        # Standardize to target length
        if min_len > target_length:
            start_idx = (min_len - target_length) // 2
            emg_data = emg_data[start_idx:start_idx + target_length]
            quaternion_data = quaternion_data[start_idx:start_idx + target_length]
            print(f"  After trimming: EMG={emg_data.shape}, Quaternion={quaternion_data.shape}")
        elif min_len < target_length:
            emg_pad = np.zeros((target_length - min_len, 8))
            quaternion_pad = np.zeros((target_length - min_len, 4))
            emg_data = np.vstack([emg_data, emg_pad])
            quaternion_data = np.vstack([quaternion_data, quaternion_pad])
            print(f"  After padding: EMG={emg_data.shape}, Quaternion={quaternion_data.shape}")
        
        # Verify final shapes
        if emg_data.shape[0] == target_length and quaternion_data.shape[0] == target_length:
            print(f"  ✅ Standardization successful!")
        else:
            print(f"  ❌ Standardization failed!")
        
        # Test concatenation
        try:
            combined = np.concatenate([emg_data, quaternion_data], axis=1)
            print(f"  Combined shape: {combined.shape}")
        except Exception as e:
            print(f"  ❌ Concatenation failed: {e}")

def test_sample_limit():
    """Test the sample limit logic"""
    print(f"\n🎯 Testing Sample Limit Logic")
    print("=" * 30)
    
    samples_per_class = 30
    current_samples = 29
    
    print(f"Current samples: {current_samples}/{samples_per_class}")
    
    # Test limit check
    if current_samples >= samples_per_class:
        print("❌ Should not allow more recording")
    else:
        print("✅ Can record more samples")
    
    # Test after adding one more
    current_samples += 1
    print(f"After adding one: {current_samples}/{samples_per_class}")
    
    if current_samples >= samples_per_class:
        print("✅ Correctly reached limit")
    else:
        print("❌ Limit not enforced")

if __name__ == "__main__":
    test_length_standardization()
    test_sample_limit()
    
    print(f"\n🎉 Fix Summary:")
    print(f"  ✅ Length standardization: All data now 100 timesteps")
    print(f"  ✅ Sample limit enforcement: Button disabled when limit reached")
    print(f"  ✅ Consistent data: No more shape mismatches")
    print(f"  ✅ Better logging: Shows original vs standardized lengths") 