#!/usr/bin/env python3
"""
Debug script to understand the actual data format
"""
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def debug_data_format():
    """
    Debug the actual data format to understand the structure
    """
    print("=== Debugging Data Format ===")
    
    # Load existing data
    data_path = os.path.join("data", "data.npz")
    if not os.path.exists(data_path):
        print(f"❌ No training data found at {data_path}")
        return
    
    try:
        data = np.load(data_path, allow_pickle=True)
        print(f"✅ Loaded data: {data_path}")
        print(f"Keys: {list(data.keys())}")
        
        # Analyze each class
        for key in data.keys():
            if key.endswith('_emg'):
                class_name = key.replace('_emg', '')
                emg_data = data[key]
                quaternion_key = f"{class_name}_quaternion"
                
                print(f"\n📝 Class {class_name}:")
                print(f"  EMG data type: {type(emg_data)}")
                print(f"  EMG data length: {len(emg_data)}")
                
                if len(emg_data) > 0:
                    print(f"  First EMG sample type: {type(emg_data[0])}")
                    print(f"  First EMG sample: {emg_data[0]}")
                    
                    if hasattr(emg_data[0], 'shape'):
                        print(f"  First EMG sample shape: {emg_data[0].shape}")
                    else:
                        print(f"  First EMG sample has no shape attribute")
                        print(f"  First EMG sample value: {emg_data[0]}")
                
                if quaternion_key in data:
                    quaternion_data = data[quaternion_key]
                    print(f"  Quaternion data type: {type(quaternion_data)}")
                    print(f"  Quaternion data length: {len(quaternion_data)}")
                    
                    if len(quaternion_data) > 0:
                        print(f"  First quaternion sample type: {type(quaternion_data[0])}")
                        print(f"  First quaternion sample: {quaternion_data[0]}")
                        
                        if hasattr(quaternion_data[0], 'shape'):
                            print(f"  First quaternion sample shape: {quaternion_data[0].shape}")
                        else:
                            print(f"  First quaternion sample has no shape attribute")
                            print(f"  First quaternion sample value: {quaternion_data[0]}")
        
        # Try to understand the data structure better
        print(f"\n🔍 Detailed Analysis:")
        for key in data.keys():
            if key.endswith('_emg'):
                class_name = key.replace('_emg', '')
                emg_data = data[key]
                quaternion_key = f"{class_name}_quaternion"
                
                print(f"\n{class_name} EMG data:")
                print(f"  Type: {type(emg_data)}")
                print(f"  Length: {len(emg_data)}")
                
                # Check first few samples
                for i in range(min(3, len(emg_data))):
                    sample = emg_data[i]
                    print(f"  Sample {i}: type={type(sample)}, value={sample}")
                    
                    if isinstance(sample, np.ndarray):
                        print(f"    Shape: {sample.shape}")
                        print(f"    Data type: {sample.dtype}")
                    elif isinstance(sample, (list, tuple)):
                        print(f"    Length: {len(sample)}")
                        print(f"    First few values: {sample[:5]}")
                
                if quaternion_key in data:
                    quaternion_data = data[quaternion_key]
                    print(f"\n{class_name} Quaternion data:")
                    print(f"  Type: {type(quaternion_data)}")
                    print(f"  Length: {len(quaternion_data)}")
                    
                    # Check first few samples
                    for i in range(min(3, len(quaternion_data))):
                        sample = quaternion_data[i]
                        print(f"  Sample {i}: type={type(sample)}, value={sample}")
                        
                        if isinstance(sample, np.ndarray):
                            print(f"    Shape: {sample.shape}")
                            print(f"    Data type: {sample.dtype}")
                        elif isinstance(sample, (list, tuple)):
                            print(f"    Length: {len(sample)}")
                            print(f"    First few values: {sample[:5]}")
        
    except Exception as e:
        print(f"❌ Error analyzing data: {e}")
        import traceback
        traceback.print_exc()

def fix_data_format():
    """
    Try to fix the data format if it's corrupted
    """
    print("\n=== Attempting to Fix Data Format ===")
    
    data_path = os.path.join("data", "data.npz")
    if not os.path.exists(data_path):
        print(f"❌ No training data found at {data_path}")
        return
    
    try:
        data = np.load(data_path, allow_pickle=True)
        
        # Create fixed data
        fixed_data = {}
        
        for key in data.keys():
            if key.endswith('_emg'):
                class_name = key.replace('_emg', '')
                emg_data = data[key]
                quaternion_key = f"{class_name}_quaternion"
                
                print(f"\nFixing {class_name} data...")
                
                # Try to fix EMG data
                fixed_emg = []
                for i, sample in enumerate(emg_data):
                    if isinstance(sample, (int, float)):
                        # This is a scalar - we need to create a proper array
                        print(f"  Sample {i} is scalar: {sample}")
                        # Create a dummy array (this won't work for training)
                        dummy_array = np.zeros((100, 8))
                        fixed_emg.append(dummy_array)
                    elif isinstance(sample, np.ndarray):
                        fixed_emg.append(sample)
                    else:
                        print(f"  Sample {i} has unknown type: {type(sample)}")
                        # Skip this sample
                        continue
                
                if fixed_emg:
                    fixed_data[key] = fixed_emg
                    print(f"  Fixed {len(fixed_emg)} EMG samples")
                
                # Try to fix quaternion data
                if quaternion_key in data:
                    quaternion_data = data[quaternion_key]
                    fixed_quaternion = []
                    
                    for i, sample in enumerate(quaternion_data):
                        if isinstance(sample, (int, float)):
                            print(f"  Quaternion sample {i} is scalar: {sample}")
                            # Create a dummy array
                            dummy_array = np.zeros((100, 4))
                            fixed_quaternion.append(dummy_array)
                        elif isinstance(sample, np.ndarray):
                            fixed_quaternion.append(sample)
                        else:
                            print(f"  Quaternion sample {i} has unknown type: {type(sample)}")
                            continue
                    
                    if fixed_quaternion:
                        fixed_data[quaternion_key] = fixed_quaternion
                        print(f"  Fixed {len(fixed_quaternion)} quaternion samples")
        
        if fixed_data:
            # Save fixed data
            fixed_path = os.path.join("data", "fixed_data.npz")
            np.savez(fixed_path, **fixed_data)
            print(f"\n✅ Saved fixed data to {fixed_path}")
        else:
            print("\n❌ No data could be fixed")
        
    except Exception as e:
        print(f"❌ Error fixing data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_data_format()
    fix_data_format() 