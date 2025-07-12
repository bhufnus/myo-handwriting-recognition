#!/usr/bin/env python3
"""
Test and compare the old variance-only approach vs the new improved approach
"""
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.improved_features import extract_all_improved_features, print_feature_analysis
from src.improved_model import predict_with_improved_model, load_improved_model
from src.model import load_trained_model
from src.preprocessing import preprocess_emg

def compare_approaches():
    """
    Compare old vs new approach on the same data
    """
    print("=== Comparing Old vs New Approach ===")
    print()
    
    # Generate test data
    print("Generating test data...")
    
    # Test 1: High variance data (like your A gesture)
    high_var_emg = np.random.randn(100, 8) * 10  # High variance
    high_var_quat = np.random.randn(100, 4) * 0.1
    
    # Test 2: Low variance data (like true idle)
    low_var_emg = np.random.randn(100, 8) * 0.5  # Low variance
    low_var_quat = np.random.randn(100, 4) * 0.01
    
    # Test 3: Medium variance data
    med_var_emg = np.random.randn(100, 8) * 5  # Medium variance
    med_var_quat = np.random.randn(100, 4) * 0.05
    
    test_cases = [
        ("High Variance (A-like)", high_var_emg, high_var_quat),
        ("Low Variance (Idle-like)", low_var_emg, low_var_quat),
        ("Medium Variance", med_var_emg, med_var_quat)
    ]
    
    print("Test cases:")
    for name, emg, quat in test_cases:
        emg_std = np.std(emg)
        quat_std = np.std(quat)
        print(f"  {name}: EMG std={emg_std:.2f}, Quat std={quat_std:.4f}")
    
    print()
    
    # Test old approach (if available)
    try:
        print("=== Testing Old Approach (Variance-Only) ===")
        old_model, old_le = load_trained_model()
        
        for name, emg, quat in test_cases:
            # Old approach: just concatenate EMG and quaternion
            processed_emg = preprocess_emg(emg)
            X_old = np.concatenate([processed_emg, quat], axis=1)
            X_old_batch = X_old[np.newaxis, ...]
            
            prediction = old_model.predict(X_old_batch, verbose=0)
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction)
            predicted_label = old_le.inverse_transform([predicted_class])[0]
            
            print(f"  {name}: {predicted_label} (confidence: {confidence:.3f})")
            print(f"    All probabilities: A={prediction[0][0]:.3f}, B={prediction[0][1]:.3f}, C={prediction[0][2]:.3f}, IDLE={prediction[0][3]:.3f}, NOISE={prediction[0][4]:.3f}")
        
    except Exception as e:
        print(f"  ❌ Old model not available: {e}")
    
    print()
    
    # Test new approach
    try:
        print("=== Testing New Approach (Improved Features) ===")
        new_model, new_scaler, new_le = load_improved_model()
        
        for name, emg, quat in test_cases:
            predicted_label, confidence, probabilities = predict_with_improved_model(
                emg, quat, new_model, new_scaler, new_le
            )
            
            print(f"  {name}: {predicted_label} (confidence: {confidence:.3f})")
            print(f"    All probabilities:")
            for i, (class_name, prob) in enumerate(zip(new_le.classes_, probabilities)):
                print(f"      {class_name}: {prob:.3f}")
        
    except Exception as e:
        print(f"  ❌ New model not available: {e}")
        print("  Training new model from existing data...")
        from train_improved_model import train_from_existing_data
        train_from_existing_data()

def analyze_feature_differences():
    """
    Analyze how the new features differ between classes
    """
    print("\n=== Feature Analysis ===")
    
    # Generate different types of data
    data_types = {
        "A-like": (np.random.randn(100, 8) * 10, np.random.randn(100, 4) * 0.1),
        "B-like": (np.random.randn(100, 8) * 8, np.random.randn(100, 4) * 0.08),
        "C-like": (np.random.randn(100, 8) * 12, np.random.randn(100, 4) * 0.12),
        "Idle-like": (np.random.randn(100, 8) * 0.5, np.random.randn(100, 4) * 0.01),
        "Noise-like": (np.random.randn(100, 8) * 15, np.random.randn(100, 4) * 0.15)
    }
    
    print("Feature analysis for different data types:")
    for name, (emg, quat) in data_types.items():
        print(f"\n{name}:")
        features = extract_all_improved_features(emg, quat)
        print(f"  Total features: {len(features)}")
        print(f"  Feature mean: {np.mean(features):.4f}")
        print(f"  Feature std: {np.std(features):.4f}")
        print(f"  Feature range: [{np.min(features):.4f}, {np.max(features):.4f}]")

def test_real_data():
    """
    Test with real data from your debug script
    """
    print("\n=== Testing with Real Data ===")
    
    # Simulate the data from your debug script
    # A gesture with high variance
    real_a_emg = np.random.randn(100, 8) * 10.57  # Your A std
    real_a_quat = np.random.randn(100, 4) * 0.1
    
    # Idle with low variance
    real_idle_emg = np.random.randn(100, 8) * 0.95  # Your idle std
    real_idle_quat = np.random.randn(100, 4) * 0.01
    
    test_data = [
        ("Real A Data", real_a_emg, real_a_quat),
        ("Real Idle Data", real_idle_emg, real_idle_quat)
    ]
    
    try:
        new_model, new_scaler, new_le = load_improved_model()
        
        for name, emg, quat in test_data:
            print(f"\n{name}:")
            predicted_label, confidence, probabilities = predict_with_improved_model(
                emg, quat, new_model, new_scaler, new_le
            )
            
            print(f"  Prediction: {predicted_label}")
            print(f"  Confidence: {confidence:.3f}")
            print(f"  All probabilities:")
            for class_name, prob in zip(new_le.classes_, probabilities):
                print(f"    {class_name}: {prob:.3f}")
        
    except Exception as e:
        print(f"❌ Error testing with real data: {e}")

def main():
    print("=== Improved Approach Testing ===")
    print()
    print("This script compares the old variance-only approach")
    print("with the new improved approach using 5 better methods.")
    print()
    
    compare_approaches()
    analyze_feature_differences()
    test_real_data()
    
    print("\n=== Summary ===")
    print("The improved approach should:")
    print("✅ Not rely solely on EMG variance")
    print("✅ Use temporal patterns and frequency analysis")
    print("✅ Consider correlations between channels")
    print("✅ Identify gesture-specific movement patterns")
    print("✅ Use multiple criteria for idle detection")
    print()
    print("This should solve the problem where high variance")
    print("was incorrectly classified as IDLE!")

if __name__ == "__main__":
    main() 