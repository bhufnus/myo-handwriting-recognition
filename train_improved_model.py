#!/usr/bin/env python3
"""
Train the improved model using the 5 better approaches for gesture recognition
"""
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.improved_model import train_improved_model, test_improved_model
from scripts.train_gui_simple import SimpleMyoGUI

def main():
    print("=== Improved Gesture Recognition Model Training ===")
    print()
    print("This script trains a model using the 5 better approaches:")
    print("1. Temporal patterns (how signals change over time)")
    print("2. Frequency domain analysis")
    print("3. Cross-correlation between channels")
    print("4. Gesture-specific movement patterns")
    print("5. Multiple criteria for idle detection")
    print()
    print("This approach doesn't rely solely on EMG variance!")
    print()
    
    # Create GUI for data collection
    app = SimpleMyoGUI(
        labels=['A', 'B', 'C', 'IDLE', 'NOISE'], 
        samples_per_class=100,  # Reduced for faster collection
        duration_ms=2000
    )
    
    print("GUI launched! Follow these steps:")
    print("1. Switch to Training tab")
    print("2. Collect 100 samples each for A, B, C (write the letters)")
    print("3. Collect 100 samples for IDLE (keep arm still)")
    print("4. Collect 100 samples for NOISE (random movements)")
    print("5. Train the improved model")
    print("6. Test predictions")
    print()
    print("The improved model will:")
    print("- Use temporal patterns instead of just variance")
    print("- Analyze frequency content of signals")
    print("- Look at correlations between channels")
    print("- Identify gesture-specific movement patterns")
    print("- Use multiple criteria for idle detection")
    print()
    print("This should solve the variance-only problem!")
    
    app.mainloop()

def train_from_existing_data():
    """
    Train the improved model using existing data
    """
    print("=== Training Improved Model from Existing Data ===")
    
    # Try to load fixed data first, then fall back to original data
    data_paths = [
        os.path.join("data", "fixed_data.npz"),
        os.path.join("data", "data.npz")
    ]
    
    data = None
    used_path = None
    
    for path in data_paths:
        if os.path.exists(path):
            try:
                data = np.load(path, allow_pickle=True)
                used_path = path
                print(f"✅ Loaded data: {path}")
                break
            except Exception as e:
                print(f"⚠️ Error loading {path}: {e}")
                continue
    
    if data is None:
        print(f"❌ No training data found")
        return
    
    try:
        # Prepare data for improved model
        emg_data_list = []
        quaternion_data_list = []
        labels = []
        
        for key in data.keys():
            if key.endswith('_emg'):
                class_name = key.replace('_emg', '')
                emg_data = data[key]
                quaternion_key = f"{class_name}_quaternion"
                
                if quaternion_key in data:
                    quaternion_data = data[quaternion_key]
                    
                    print(f"\nProcessing {class_name}:")
                    print(f"  EMG samples: {len(emg_data)}")
                    print(f"  Quaternion samples: {len(quaternion_data)}")
                    
                    # Add each sample
                    for i in range(len(emg_data)):
                        emg_sample = emg_data[i]
                        quat_sample = quaternion_data[i]
                        
                        # Ensure samples are numpy arrays with correct shape
                        if isinstance(emg_sample, np.ndarray) and emg_sample.shape == (100, 8):
                            if isinstance(quat_sample, np.ndarray) and quat_sample.shape == (100, 4):
                                emg_data_list.append(emg_sample)
                                quaternion_data_list.append(quat_sample)
                                labels.append(class_name)
                            else:
                                print(f"    Skipping sample {i} - quaternion shape: {quat_sample.shape if hasattr(quat_sample, 'shape') else 'no shape'}")
                        else:
                            print(f"    Skipping sample {i} - EMG shape: {emg_sample.shape if hasattr(emg_sample, 'shape') else 'no shape'}")
        
        print(f"\n✅ Prepared {len(emg_data_list)} samples for training")
        print(f"Classes: {set(labels)}")
        
        if len(emg_data_list) == 0:
            print("❌ No valid samples found!")
            return
        
        # Train the improved model
        print("\nTraining improved model...")
        model, scaler, le = train_improved_model(
            emg_data_list, quaternion_data_list, labels, model_type='ensemble'
        )
        
        print("\n✅ Training complete!")
        print("Testing the model...")
        test_improved_model()
        
    except Exception as e:
        print(f"❌ Error training model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--existing':
        train_from_existing_data()
    else:
        main() 