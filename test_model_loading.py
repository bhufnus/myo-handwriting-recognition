#!/usr/bin/env python3
"""
Test script to verify model loading works correctly
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model import load_trained_model

def test_model_loading():
    """Test if the model can be loaded successfully"""
    try:
        print("Testing model loading...")
        model, le = load_trained_model()
        print(f"✅ Model loaded successfully!")
        print(f"   Model input shape: {model.input_shape}")
        print(f"   Model output shape: {model.output_shape}")
        print(f"   Label encoder classes: {le.classes_}")
        return True
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

if __name__ == "__main__":
    success = test_model_loading()
    if success:
        print("\n✅ Model loading test passed!")
    else:
        print("\n❌ Model loading test failed!") 