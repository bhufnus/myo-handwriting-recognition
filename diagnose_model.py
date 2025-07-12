#!/usr/bin/env python3
"""
Diagnostic script to analyze the model's behavior and identify issues
"""
import sys
import os
import numpy as np
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model import load_trained_model
from src.preprocessing import preprocess_emg

def test_model_predictions():
    """Test the model with different types of input data"""
    print("=== Model Diagnostics ===")
    
    try:
        # Load the model
        print("Loading model...")
        model, le = load_trained_model()
        print(f"Model loaded successfully!")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
        print(f"Label encoder classes: {le.classes_}")
        print()
        
        # Test 1: Random noise data
        print("Test 1: Random noise data")
        random_emg = np.random.randn(100, 8) * 50  # Random EMG data
        random_quaternion = np.random.randn(100, 4)  # Random quaternion data
        test_prediction(model, le, random_emg, random_quaternion, "Random noise")
        
        # Test 2: Zero/static data (should be IDLE)
        print("\nTest 2: Static/zero data (should be IDLE)")
        static_emg = np.zeros((100, 8))  # No EMG activity
        static_quaternion = np.array([[0, 0, 0, 1]] * 100)  # No rotation
        test_prediction(model, le, static_emg, static_quaternion, "Static data")
        
        # Test 3: High variance data (should be gesture)
        print("\nTest 3: High variance data (should be gesture)")
        high_var_emg = np.random.randn(100, 8) * 200  # High variance EMG
        high_var_quaternion = np.random.randn(100, 4) * 0.5  # High variance quaternion
        test_prediction(model, le, high_var_emg, high_var_quaternion, "High variance data")
        
        # Test 4: Check model's class distribution
        print("\nTest 4: Model class distribution analysis")
        analyze_model_distribution(model, le)
        
        # Test 5: Test with actual training data format
        print("\nTest 5: Training data format test")
        test_training_format(model, le)
        
    except Exception as e:
        print(f"Error during diagnostics: {e}")
        import traceback
        print(traceback.format_exc())

def test_prediction(model, le, emg_data, quaternion_data, test_name):
    """Test prediction with given data"""
    try:
        # Preprocess EMG
        emg_proc = preprocess_emg(emg_data)
        
        # Concatenate features
        X_test = np.concatenate([emg_proc, quaternion_data], axis=1)
        X_test_batch = X_test[np.newaxis, ...]
        
        # Make prediction
        prediction = model.predict(X_test_batch, verbose=0)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)
        predicted_label = le.inverse_transform([predicted_class])[0]
        
        # Show all class probabilities
        print(f"  {test_name}:")
        print(f"    Predicted: {predicted_label} (confidence: {confidence:.3f})")
        print(f"    All probabilities:")
        for i, (label, prob) in enumerate(zip(le.classes_, prediction[0])):
            print(f"      {label}: {prob:.3f}")
        
        return predicted_label, confidence
        
    except Exception as e:
        print(f"  Error testing {test_name}: {e}")
        return None, 0

def analyze_model_distribution(model, le):
    """Analyze the model's prediction distribution"""
    print("  Testing model with 100 random inputs...")
    
    predictions = []
    confidences = []
    
    for i in range(100):
        # Generate random test data
        random_emg = np.random.randn(100, 8) * 100
        random_quaternion = np.random.randn(100, 4) * 0.1
        
        try:
            emg_proc = preprocess_emg(random_emg)
            X_test = np.concatenate([emg_proc, random_quaternion], axis=1)
            X_test_batch = X_test[np.newaxis, ...]
            
            prediction = model.predict(X_test_batch, verbose=0)
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction)
            predicted_label = le.inverse_transform([predicted_class])[0]
            
            predictions.append(predicted_label)
            confidences.append(confidence)
            
        except Exception as e:
            print(f"    Error in test {i}: {e}")
    
    # Analyze results
    from collections import Counter
    pred_counts = Counter(predictions)
    avg_confidence = np.mean(confidences)
    
    print(f"    Prediction distribution:")
    for label, count in pred_counts.items():
        print(f"      {label}: {count} times ({count/len(predictions)*100:.1f}%)")
    print(f"    Average confidence: {avg_confidence:.3f}")
    
    # Check if model is biased
    most_common = pred_counts.most_common(1)[0]
    if most_common[1] > len(predictions) * 0.8:  # If one class dominates
        print(f"    ⚠️  MODEL BIAS DETECTED: {most_common[0]} dominates predictions")
        print(f"    This suggests the model may be overfitting or has training issues")

def test_training_format(model, le):
    """Test if the model expects the same format as training data"""
    print("  Testing training data format compatibility...")
    
    # Simulate what the training data might look like
    # (This is a guess - we need to see the actual training data format)
    
    # Test with different input shapes
    test_shapes = [
        (1, 100, 12),  # Expected shape
        (1, 50, 12),   # Shorter sequence
        (1, 200, 12),  # Longer sequence
    ]
    
    for shape in test_shapes:
        try:
            test_data = np.random.randn(*shape)
            prediction = model.predict(test_data, verbose=0)
            print(f"    Shape {shape}: Prediction successful")
        except Exception as e:
            print(f"    Shape {shape}: Error - {e}")

if __name__ == "__main__":
    test_model_predictions() 