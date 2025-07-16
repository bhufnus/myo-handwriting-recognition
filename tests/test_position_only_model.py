#!/usr/bin/env python3
"""
Tests for position-only model (no training)
"""

import unittest
import numpy as np
import sys
import os
import tempfile
import pickle

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.position_only_model import (
    create_position_only_model, 
    preprocess_quaternion_only,
    predict_position_only
)


class TestPositionOnlyModel(unittest.TestCase):
    """Test cases for position-only model functions (no training)"""
    
    def setUp(self):
        """Set up test data"""
        # Create sample quaternion data
        self.sample_quaternions = [np.random.randn(100, 4) for _ in range(5)]
        self.sample_labels = ['A', 'B', 'C', 'IDLE', 'NOISE']
        
        # Create test quaternion data for prediction
        self.test_quaternion = np.random.randn(100, 4)
        
    def test_create_position_only_model(self):
        """Test position-only model creation"""
        model = create_position_only_model(input_shape=(100, 4), num_classes=5)
        
        # Check model structure (updated layer count)
        self.assertGreaterEqual(len(model.layers), 7)  # At least 7 layers
        
        # Check input shape
        self.assertEqual(model.input_shape, (None, 100, 4))
        
        # Check output shape
        self.assertEqual(model.output_shape, (None, 5))
        
    def test_preprocess_quaternion_only_basic(self):
        """Test quaternion preprocessing"""
        processed = preprocess_quaternion_only(self.test_quaternion)
        
        # Check output shape
        self.assertEqual(processed.shape, (100, 4))
        
        # Check data is finite
        self.assertTrue(np.all(np.isfinite(processed)))
        
    def test_preprocess_quaternion_only_empty(self):
        """Test quaternion preprocessing with empty data"""
        processed = preprocess_quaternion_only([])
        
        # Should return default shape
        self.assertEqual(processed.shape, (100, 4))
        self.assertTrue(np.allclose(processed, 0))
        
    def test_preprocess_quaternion_only_none(self):
        """Test quaternion preprocessing with None"""
        processed = preprocess_quaternion_only(None)
        
        # Should return default shape
        self.assertEqual(processed.shape, (100, 4))
        self.assertTrue(np.allclose(processed, 0))
        
    def test_predict_position_only_untrained(self):
        """Test position-only prediction with untrained model"""
        # Create a simple model for testing
        model = create_position_only_model(input_shape=(100, 4), num_classes=5)
        
        # Create a simple label encoder
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.fit(['A', 'B', 'C', 'IDLE', 'NOISE'])
        
        # Test prediction with untrained model
        try:
            label, confidence, probabilities = predict_position_only(
                self.test_quaternion, model, le
            )
            
            # Check output types (including numpy types)
            self.assertIsInstance(label, str)
            self.assertIsInstance(confidence, (int, float, np.floating))
            self.assertIsInstance(probabilities, np.ndarray)
            
            # Check confidence is reasonable
            self.assertGreaterEqual(confidence, 0)
            self.assertLessEqual(confidence, 1)
            
            # Check probabilities sum to 1
            self.assertAlmostEqual(np.sum(probabilities), 1.0, places=5)
            
            # Check label is valid
            self.assertIn(label, ['A', 'B', 'C', 'IDLE', 'NOISE'])
            
        except Exception as e:
            self.fail(f"Prediction should work with untrained model: {e}")
            
    def test_model_save_load(self):
        """Test model saving and loading"""
        # Create model
        model = create_position_only_model(input_shape=(100, 4), num_classes=5)
        
        # Create label encoder
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.fit(['A', 'B', 'C', 'IDLE', 'NOISE'])
        
        # Save model and labels
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            model_path = f.name
        with tempfile.NamedTemporaryFile(suffix='_labels.pkl', delete=False) as f:
            le_path = f.name
            
        try:
            # Save
            model.save(model_path)
            with open(le_path, 'wb') as f:
                pickle.dump(le, f)
            
            # Load
            from tensorflow.keras.models import load_model
            loaded_model = load_model(model_path)
            with open(le_path, 'rb') as f:
                loaded_le = pickle.load(f)
            
            # Check they're equivalent
            self.assertEqual(model.input_shape, loaded_model.input_shape)
            self.assertEqual(model.output_shape, loaded_model.output_shape)
            self.assertEqual(list(le.classes_), list(loaded_le.classes_))
            
        finally:
            # Cleanup
            if os.path.exists(model_path):
                os.unlink(model_path)
            if os.path.exists(le_path):
                os.unlink(le_path)
                
    def test_model_performance_characteristics(self):
        """Test model performance characteristics"""
        # Test model creation speed
        import time
        start_time = time.time()
        
        model = create_position_only_model(input_shape=(100, 4), num_classes=5)
        
        creation_time = time.time() - start_time
        
        # Should create model quickly (< 1 second)
        self.assertLess(creation_time, 1.0)
        
        # Test prediction speed
        start_time = time.time()
        
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.fit(['A', 'B', 'C', 'IDLE', 'NOISE'])
        
        # Make multiple predictions
        for _ in range(10):
            predict_position_only(self.test_quaternion, model, le)
            
        prediction_time = time.time() - start_time
        
        # Should make predictions quickly (< 2 seconds for 10 predictions)
        self.assertLess(prediction_time, 2.0)
        
    def test_error_handling(self):
        """Test error handling in model functions"""
        # Test with invalid input shapes
        try:
            model = create_position_only_model(input_shape=(50, 3), num_classes=3)
            self.assertIsNotNone(model)
        except Exception as e:
            self.fail(f"Should handle different input shapes: {e}")
            
        # Test preprocessing with invalid data (should handle gracefully)
        invalid_data = np.random.randn(50, 3)  # Wrong shape
        try:
            processed = preprocess_quaternion_only(invalid_data)
            self.assertIsNotNone(processed)
            # Should return default shape even for invalid input
            self.assertEqual(processed.shape, (100, 4))
        except Exception as e:
            self.fail(f"Should handle invalid data gracefully: {e}")
            
    def test_model_configuration(self):
        """Test different model configurations"""
        # Test different input shapes
        shapes = [(50, 4), (100, 4), (200, 4)]
        classes = [3, 5, 10]
        
        for shape in shapes:
            for num_classes in classes:
                try:
                    model = create_position_only_model(input_shape=shape, num_classes=num_classes)
                    self.assertEqual(model.input_shape, (None,) + shape)
                    self.assertEqual(model.output_shape, (None, num_classes))
                except Exception as e:
                    self.fail(f"Should handle shape {shape} and {num_classes} classes: {e}")


if __name__ == '__main__':
    unittest.main() 