#!/usr/bin/env python3
"""
Quick test of position-only model using existing data
"""
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_position_only():
    """Quick test of position-only model"""
    print("🎯 Testing Position-Only Model")
    print("=" * 40)
    
    try:
        # Import the position-only model functions
        from src.position_only_model import train_position_only_model, predict_position_only
        
        # Look for existing data in data/ directory
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        if os.path.exists(data_dir):
            data_files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
        else:
            data_files = [f for f in os.listdir('.') if f.endswith('.npz')]
            
        if not data_files:
            print("❌ No training data found. Please collect some data first.")
            print("   Looking in:", data_dir if os.path.exists(data_dir) else "current directory")
            return
        
        # Use the most recent data file
        if os.path.exists(data_dir):
            data_file = os.path.join(data_dir, sorted(data_files, reverse=True)[0])
        else:
            data_file = sorted(data_files, reverse=True)[0]
        print(f"📂 Using data file: {data_file}")
        
        # Load data
        data = np.load(data_file, allow_pickle=True)
        
        # Extract quaternion data
        quaternion_samples = []
        labels = []
        
        for key in data.keys():
            if key.endswith('_quaternion'):
                class_name = key.replace('_quaternion', '')
                quaternion_data = data[key]
                
                print(f"📊 {class_name}: {len(quaternion_data)} samples")
                
                for quat_sample in quaternion_data:
                    if isinstance(quat_sample, np.ndarray) and quat_sample.shape == (100, 4):
                        quaternion_samples.append(quat_sample)
                        labels.append(class_name)
        
        if len(quaternion_samples) < 10:
            print("❌ Not enough samples for training")
            return
        
        print(f"✅ Total samples: {len(quaternion_samples)}")
        print(f"Classes: {set(labels)}")
        
        # Train position-only model
        print("\n🚀 Training position-only model...")
        model, le, history = train_position_only_model(quaternion_samples, labels, list(set(labels)))
        
        print("\n✅ Training completed!")
        
        # Test with some sample data
        print("\n🧪 Testing predictions...")
        
        # Test with idle-like data (small quaternion changes)
        idle_test = np.random.randn(100, 4) * 0.1
        predicted_label, confidence, probabilities = predict_position_only(idle_test, model, le)
        print(f"Idle test: {predicted_label} (confidence: {confidence:.3f})")
        
        # Test with gesture-like data (larger quaternion changes)
        gesture_test = np.random.randn(100, 4) * 0.5
        predicted_label, confidence, probabilities = predict_position_only(gesture_test, model, le)
        print(f"Gesture test: {predicted_label} (confidence: {confidence:.3f})")
        
        print("\n🎉 Position-only model is ready!")
        print("You can now use it in the GUI by enabling 'Position-Only Model'")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_position_only() 