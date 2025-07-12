#!/usr/bin/env python3
"""
Test script to verify model compatibility with prediction code
"""
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model import load_trained_model
from src.preprocessing import preprocess_emg

def test_model_compatibility():
    """Test if the model can be loaded and used for prediction"""
    try:
        print("Testing model compatibility...")
        
        # Load the model
        model, le = load_trained_model()
        print(f"✅ Model loaded successfully!")
        print(f"   Model input shape: {model.input_shape}")
        print(f"   Model output shape: {model.output_shape}")
        print(f"   Label encoder classes: {le.classes_}")
        
        # Create dummy test data in the format the prediction code uses
        window_size = 100
        dummy_emg = np.random.randn(window_size, 8)  # 8 EMG channels
        dummy_quaternion = np.random.randn(window_size, 4)  # 4 quaternion components
        
        # Preprocess EMG like the prediction code does
        emg_proc = preprocess_emg(dummy_emg)
        
        # Concatenate like the prediction code does
        X_test = np.concatenate([emg_proc, dummy_quaternion], axis=1)  # shape (window_size, 12)
        
        # Add batch dimension like the prediction code does
        X_test_batch = X_test[np.newaxis, ...]  # shape (1, window_size, 12)
        
        print(f"   Test input shape: {X_test_batch.shape}")
        print(f"   Expected input shape: {model.input_shape}")
        
        # Test prediction
        prediction = model.predict(X_test_batch, verbose=0)
        predicted_class = np.argmax(prediction)
        predicted_label = le.inverse_transform([predicted_class])[0]
        confidence = np.max(prediction)
        
        print(f"✅ Prediction successful!")
        print(f"   Predicted: {predicted_label}")
        print(f"   Confidence: {confidence:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model compatibility test failed: {e}")
        import traceback
        print(f"Full traceback:\n{traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = test_model_compatibility()
    if success:
        print("\n✅ Model compatibility test passed!")
        print("The model should work with the prediction GUI.")
    else:
        print("\n❌ Model compatibility test failed!")
        print("There may be a format mismatch between training and prediction.") 