#!/usr/bin/env python3
"""
Check the actual training data to understand class distribution
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np

def analyze_training_data():
    """Analyze the training data in the data/ folder"""
    print("📊 Analyzing Training Data")
    print("=" * 40)
    
    data_path = os.path.join("data", "data.npz")
    if not os.path.exists(data_path):
        print(f"❌ No training data found at {data_path}")
        return
    
    try:
        data = np.load(data_path, allow_pickle=True)
        print(f"✅ Loaded training data: {data_path}")
        print(f"   Keys: {list(data.keys())}")
        
        # Analyze each class
        total_samples = 0
        class_info = {}
        
        for key in data.keys():
            if key.endswith('_emg'):
                class_name = key.replace('_emg', '')
                emg_data = data[key]
                quaternion_key = f"{class_name}_quaternion"
                
                if quaternion_key in data:
                    quaternion_data = data[quaternion_key]
                    num_samples = len(emg_data)
                    total_samples += num_samples
                    
                    class_info[class_name] = {
                        'samples': num_samples,
                        'emg_data': emg_data,
                        'quaternion_data': quaternion_data
                    }
                    
                    print(f"\n📝 Class {class_name}:")
                    print(f"   Samples: {num_samples}")
                    
                    if num_samples > 0:
                        # Check data quality
                        sample_emg = emg_data[0]
                        sample_quat = quaternion_data[0]
                        print(f"   EMG shape: {sample_emg.shape}")
                        print(f"   Quaternion shape: {sample_quat.shape}")
                        
                        # Check data ranges
                        if len(emg_data) > 0:
                            all_emg = np.concatenate(emg_data)
                            all_quat = np.concatenate(quaternion_data)
                            print(f"   EMG range: {np.min(all_emg):.2f} to {np.max(all_emg):.2f}")
                            print(f"   Quaternion range: {np.min(all_quat):.2f} to {np.max(all_quat):.2f}")
        
        # Show class distribution
        print(f"\n📈 Class Distribution:")
        for class_name, info in class_info.items():
            percentage = (info['samples'] / total_samples) * 100
            print(f"   {class_name}: {info['samples']} samples ({percentage:.1f}%)")
        
        # Check for imbalance
        if len(class_info) > 1:
            samples = [info['samples'] for info in class_info.values()]
            min_samples = min(samples)
            max_samples = max(samples)
            imbalance_ratio = max_samples / min_samples if min_samples > 0 else float('inf')
            
            print(f"\n⚖️  Balance Analysis:")
            print(f"   Min samples: {min_samples}")
            print(f"   Max samples: {max_samples}")
            print(f"   Imbalance ratio: {imbalance_ratio:.2f}x")
            
            if imbalance_ratio > 2.0:
                print(f"   ⚠️  SIGNIFICANT IMBALANCE DETECTED!")
                print(f"   Consider collecting more samples for underrepresented classes")
            else:
                print(f"   ✅ Data is reasonably balanced")
        
        # Check data quality
        print(f"\n🔍 Data Quality Check:")
        for class_name, info in class_info.items():
            if info['samples'] > 0:
                # Check for NaN or infinite values
                all_emg = np.concatenate(info['emg_data'])
                all_quat = np.concatenate(info['quaternion_data'])
                
                has_nan_emg = np.any(np.isnan(all_emg))
                has_nan_quat = np.any(np.isnan(all_quat))
                has_inf_emg = np.any(np.isinf(all_emg))
                has_inf_quat = np.any(np.isinf(all_quat))
                
                print(f"   {class_name}:")
                print(f"     NaN in EMG: {has_nan_emg}")
                print(f"     NaN in Quaternion: {has_nan_quat}")
                print(f"     Inf in EMG: {has_inf_emg}")
                print(f"     Inf in Quaternion: {has_inf_quat}")
        
    except Exception as e:
        print(f"❌ Error analyzing data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_training_data() 