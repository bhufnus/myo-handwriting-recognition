#!/usr/bin/env python3
"""
Diagnostic script to analyze class A prediction issues
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import tensorflow as tf
import pickle

def analyze_class_a_data():
    """Analyze the training data for class A specifically"""
    print("🔍 Analyzing Class A Data")
    print("=" * 40)
    
    # Load training data
    data_path = os.path.join("data", "data.npz")
    if not os.path.exists(data_path):
        print(f"❌ No training data found at {data_path}")
        return
    
    try:
        data = np.load(data_path, allow_pickle=True)
        
        # Analyze class A specifically
        if 'A_emg' in data and 'A_quaternion' in data:
            emg_data = data['A_emg']
            quaternion_data = data['A_quaternion']
            
            print(f"📊 Class A Data Analysis:")
            print(f"  Number of samples: {len(emg_data)}")
            
            if len(emg_data) > 0:
                # Analyze sample lengths
                emg_lengths = [len(sample) for sample in emg_data]
                quat_lengths = [len(sample) for sample in quaternion_data]
                
                print(f"  EMG lengths: min={min(emg_lengths)}, max={max(emg_lengths)}, avg={np.mean(emg_lengths):.1f}")
                print(f"  Quaternion lengths: min={min(quat_lengths)}, max={max(quat_lengths)}, avg={np.mean(quat_lengths):.1f}")
                
                # Analyze data ranges
                all_emg = np.concatenate(emg_data)
                all_quat = np.concatenate(quaternion_data)
                
                print(f"  EMG range: {np.min(all_emg):.2f} to {np.max(all_emg):.2f}")
                print(f"  Quaternion range: {np.min(all_quat):.2f} to {np.max(all_quat):.2f}")
                
                # Check for data quality issues
                print(f"  EMG has NaN: {np.any(np.isnan(all_emg))}")
                print(f"  EMG has Inf: {np.any(np.isinf(all_emg))}")
                print(f"  Quaternion has NaN: {np.any(np.isnan(all_quat))}")
                print(f"  Quaternion has Inf: {np.any(np.isinf(all_quat))}")
                
                # Analyze variation in the data
                emg_std = np.std(all_emg, axis=0)
                quat_std = np.std(all_quat, axis=0)
                
                print(f"  EMG channel std dev: {emg_std}")
                print(f"  Quaternion std dev: {quat_std}")
                
                # Check if data is too similar (low variance)
                if np.mean(emg_std) < 10:
                    print(f"  ⚠️  Low EMG variance - data might be too similar")
                if np.mean(quat_std) < 0.1:
                    print(f"  ⚠️  Low quaternion variance - data might be too similar")
                    
        else:
            print(f"❌ No class A data found in training data")
            
    except Exception as e:
        print(f"❌ Error analyzing data: {e}")

def test_class_a_predictions():
    """Test model predictions specifically for class A patterns"""
    print(f"\n🎯 Testing Class A Predictions")
    print("=" * 40)
    
    # Look for model files
    model_files = [f for f in os.listdir('.') if f.endswith('.h5')]
    if not model_files:
        print("❌ No model files found")
        return
    
    model_file = model_files[0]
    print(f"📁 Using model: {model_file}")
    
    try:
        # Load model
        model = tf.keras.models.load_model(model_file)
        
        # Load label encoder
        le_file = model_file.replace('.h5', '_labels.pkl')
        with open(le_file, 'rb') as f:
            le = pickle.load(f)
        
        # Test with different patterns that should be class A
        test_patterns = [
            ("Upward movement", np.random.randn(100, 12) + np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
            ("Downward movement", np.random.randn(100, 12) + np.array([0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
            ("High EMG activity", np.random.randn(100, 12) + np.array([50, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
            ("Random noise", np.random.randn(100, 12)),
        ]
        
        for pattern_name, test_data in test_patterns:
            # Reshape for model
            X_test = test_data.reshape(1, 100, 12)
            
            # Make prediction
            pred = model.predict(X_test, verbose=0)
            predicted_class = np.argmax(pred)
            confidence = np.max(pred)
            predicted_label = le.inverse_transform([predicted_class])[0]
            
            print(f"  {pattern_name}: {predicted_label} (confidence: {confidence:.3f})")
            
        # Test with actual class A data if available
        data_path = os.path.join("data", "data.npz")
        if os.path.exists(data_path):
            data = np.load(data_path, allow_pickle=True)
            if 'A_emg' in data and 'A_quaternion' in data:
                print(f"\n📊 Testing with actual Class A data:")
                
                emg_data = data['A_emg']
                quaternion_data = data['A_quaternion']
                
                # Test first few samples
                for i in range(min(3, len(emg_data))):
                    emg_sample = emg_data[i]
                    quat_sample = quaternion_data[i]
                    
                    # Convert to numpy arrays if they aren't already
                    emg_sample = np.array(emg_sample, dtype=np.float32)
                    quat_sample = np.array(quat_sample, dtype=np.float32)
                    
                    # Standardize to 100 timesteps
                    min_len = min(len(emg_sample), len(quat_sample))
                    emg_sample = emg_sample[:min_len]
                    quat_sample = quat_sample[:min_len]
                    
                    if min_len >= 100:
                        # Take first 100 timesteps
                        combined = np.concatenate([emg_sample[:100], quat_sample[:100]], axis=1)
                        X_test = combined.reshape(1, 100, 12).astype(np.float32)
                        
                        pred = model.predict(X_test, verbose=0)
                        predicted_class = np.argmax(pred)
                        confidence = np.max(pred)
                        predicted_label = le.inverse_transform([predicted_class])[0]
                        
                        print(f"  Sample {i+1}: {predicted_label} (confidence: {confidence:.3f})")
                        
    except Exception as e:
        print(f"❌ Error testing predictions: {e}")

def suggest_improvements():
    """Suggest specific improvements for class A accuracy"""
    print(f"\n💡 Improvement Suggestions for Class A")
    print("=" * 50)
    
    print("1. 📊 Data Quality Issues:")
    print("   • Check if class A samples are too similar")
    print("   • Ensure sufficient variation in movement patterns")
    print("   • Verify EMG and quaternion data quality")
    
    print("\n2. 🎯 Movement Pattern Issues:")
    print("   • Class A might be confused with IDLE (similar arm position)")
    print("   • Try more distinct movements for class A")
    print("   • Consider different gesture for class A")
    
    print("\n3. 🔧 Technical Solutions:")
    print("   • Collect more class A samples with variations")
    print("   • Use data augmentation techniques")
    print("   • Adjust model architecture")
    print("   • Try different preprocessing")
    
    print("\n4. 🚀 Immediate Actions:")
    print("   • Collect 10-15 new class A samples with clear movements")
    print("   • Use the variation system (Fast, Slow, High, Low)")
    print("   • Make class A movement very distinct from IDLE")
    print("   • Retrain model with new data")

if __name__ == "__main__":
    analyze_class_a_data()
    test_class_a_predictions()
    suggest_improvements() 