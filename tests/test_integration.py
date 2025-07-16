#!/usr/bin/env python3
"""
Integration tests for UI elements and functions (no model training)
"""

import unittest
import sys
import os
import tempfile
import numpy as np
import time
from unittest.mock import Mock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import preprocess_emg, create_position_focused_sequence
from src.position_only_model import create_position_only_model, predict_position_only


class TestUIIntegration(unittest.TestCase):
    """Integration tests for UI elements and functions"""
    
    def setUp(self):
        """Set up test data"""
        # Create sample data for each class
        self.emg_data = {
            'A': [np.random.randn(100, 8) for _ in range(5)],
            'B': [np.random.randn(100, 8) for _ in range(5)],
            'C': [np.random.randn(100, 8) for _ in range(5)]
        }
        
        self.quaternion_data = {
            'A': [np.random.randn(100, 4) for _ in range(5)],
            'B': [np.random.randn(100, 4) for _ in range(5)],
            'C': [np.random.randn(100, 4) for _ in range(5)]
        }
        
        self.labels = ['A', 'B', 'C']
        
    def test_data_preprocessing_pipeline(self):
        """Test data preprocessing pipeline without training"""
        # Step 1: Preprocess data
        processed_data = []
        processed_labels = []
        
        for label in self.labels:
            for emg, quat in zip(self.emg_data[label], self.quaternion_data[label]):
                # Preprocess EMG
                processed_emg = preprocess_emg(emg)
                
                # Create combined sequence
                sequence = create_position_focused_sequence(processed_emg, quat)
                
                processed_data.append(sequence)
                processed_labels.append(label)
        
        # Verify preprocessing results
        self.assertEqual(len(processed_data), len(processed_labels))
        self.assertEqual(len(processed_data), 15)  # 5 samples per class * 3 classes
        
        # Check data shapes
        for sequence in processed_data:
            self.assertEqual(len(sequence.shape), 2)  # Should be 2D
            self.assertGreater(sequence.shape[0], 0)  # Should have data
            
    def test_data_save_load_pipeline(self):
        """Test data saving and loading pipeline"""
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            temp_path = f.name
            
        try:
            # Save data
            save_dict = {}
            for label in self.labels:
                save_dict[f'{label}_emg'] = self.emg_data[label]
                save_dict[f'{label}_quaternion'] = self.quaternion_data[label]
            
            np.savez(temp_path, **save_dict)
            
            # Load data
            loaded_data = np.load(temp_path, allow_pickle=True)
            
            # Verify loaded data
            for label in self.labels:
                self.assertIn(f'{label}_emg', loaded_data)
                self.assertIn(f'{label}_quaternion', loaded_data)
                
                # Check data integrity
                self.assertEqual(len(loaded_data[f'{label}_emg']), len(self.emg_data[label]))
                self.assertEqual(len(loaded_data[f'{label}_quaternion']), len(self.quaternion_data[label]))
                
        finally:
            # Handle Windows file permission issues
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except PermissionError:
                time.sleep(0.1)
                try:
                    os.unlink(temp_path)
                except PermissionError:
                    pass  # Give up if still locked
                
    def test_model_creation_and_save_load(self):
        """Test model creation and save/load without training"""
        # Create a simple model
        model = create_position_only_model(input_shape=(100, 4), num_classes=3)
        
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            model_path = f.name
            
        try:
            # Save model
            model.save(model_path)
            
            # Load model
            from tensorflow.keras.models import load_model
            loaded_model = load_model(model_path)
            
            # Test prediction with both models (untrained, but should work)
            test_data = np.random.randn(1, 100, 4)
            
            original_pred = model.predict(test_data, verbose=0)
            loaded_pred = loaded_model.predict(test_data, verbose=0)
            
            # Verify predictions are identical
            np.testing.assert_array_almost_equal(original_pred, loaded_pred)
            
        finally:
            # Handle Windows file permission issues
            try:
                if os.path.exists(model_path):
                    os.unlink(model_path)
            except PermissionError:
                time.sleep(0.1)
                try:
                    os.unlink(model_path)
                except PermissionError:
                    pass
                
    def test_error_handling_pipeline(self):
        """Test error handling throughout the pipeline"""
        # Test with invalid data
        invalid_emg = np.random.randn(50, 4)  # Wrong shape
        invalid_quat = np.random.randn(100, 3)  # Wrong shape
        
        # Should handle gracefully
        try:
            sequence = create_position_focused_sequence(invalid_emg, invalid_quat)
            self.assertIsNotNone(sequence)
        except Exception as e:
            self.fail(f"Should handle invalid data gracefully: {e}")
            
        # Test with empty data
        try:
            sequence = create_position_focused_sequence([], [])
            self.assertIsNotNone(sequence)
        except Exception as e:
            self.fail(f"Should handle empty data gracefully: {e}")
            
    def test_performance_pipeline(self):
        """Test performance characteristics without training"""
        # Test with larger dataset
        large_emg_data = [np.random.randn(100, 8) for _ in range(50)]
        large_quat_data = [np.random.randn(100, 4) for _ in range(50)]
        
        # Time the preprocessing
        import time
        start_time = time.time()
        
        processed_data = []
        for emg, quat in zip(large_emg_data, large_quat_data):
            processed_emg = preprocess_emg(emg)
            sequence = create_position_focused_sequence(processed_emg, quat)
            processed_data.append(sequence)
            
        preprocessing_time = time.time() - start_time
        
        # Should complete in reasonable time (< 3 seconds for 50 samples)
        self.assertLess(preprocessing_time, 3.0)
        
        # Test model creation performance
        start_time = time.time()
        model = create_position_only_model(input_shape=(100, 4), num_classes=3)
        model_creation_time = time.time() - start_time
        
        # Should create model quickly (< 1 second)
        self.assertLess(model_creation_time, 1.0)
        
    def test_prediction_without_training(self):
        """Test prediction functionality with untrained model"""
        # Create model and label encoder
        model = create_position_only_model(input_shape=(100, 4), num_classes=3)
        
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.fit(['A', 'B', 'C'])
        
        # Test prediction with untrained model
        test_quaternion = np.random.randn(100, 4)
        
        try:
            predicted_label, confidence, probabilities = predict_position_only(
                test_quaternion, model, le
            )
            
            # Verify prediction structure (not accuracy since model is untrained)
            self.assertIn(predicted_label, ['A', 'B', 'C'])
            self.assertGreaterEqual(confidence, 0)
            self.assertLessEqual(confidence, 1)
            self.assertEqual(len(probabilities), 3)
            self.assertAlmostEqual(np.sum(probabilities), 1.0, places=5)
            
        except Exception as e:
            self.fail(f"Prediction should work with untrained model: {e}")
            
    def test_ui_data_validation(self):
        """Test UI data validation functions"""
        # Test valid data
        valid_emg = np.random.randn(100, 8)
        valid_quat = np.random.randn(100, 4)
        
        self.assertTrue(self._validate_emg_data(valid_emg))
        self.assertTrue(self._validate_quaternion_data(valid_quat))
        
        # Test invalid data
        invalid_emg = np.random.randn(50, 4)  # Wrong shape
        invalid_quat = np.random.randn(100, 3)  # Wrong shape
        
        self.assertFalse(self._validate_emg_data(invalid_emg))
        self.assertFalse(self._validate_quaternion_data(invalid_quat))
        
    def test_ui_file_operations(self):
        """Test UI file operations"""
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            temp_path = f.name
            
        try:
            # Test file saving
            self._save_test_data(temp_path)
            self.assertTrue(os.path.exists(temp_path))
            
            # Test file loading
            loaded_data = self._load_test_data(temp_path)
            self.assertIsNotNone(loaded_data)
            
            # Check data integrity
            for label in ['A', 'B', 'C']:
                self.assertIn(f'{label}_emg', loaded_data)
                self.assertIn(f'{label}_quaternion', loaded_data)
                
        finally:
            # Handle Windows file permission issues
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except PermissionError:
                time.sleep(0.1)
                try:
                    os.unlink(temp_path)
                except PermissionError:
                    pass
                
    # Helper methods for testing
    def _validate_emg_data(self, data):
        """Validate EMG data format"""
        if data is None:
            return False
        if not isinstance(data, np.ndarray):
            return False
        if len(data.shape) != 2 or data.shape[1] != 8:
            return False
        return True
        
    def _validate_quaternion_data(self, data):
        """Validate quaternion data format"""
        if data is None:
            return False
        if not isinstance(data, np.ndarray):
            return False
        if len(data.shape) != 2 or data.shape[1] != 4:
            return False
        return True
        
    def _save_test_data(self, path):
        """Save test data"""
        save_dict = {}
        for label in ['A', 'B', 'C']:
            save_dict[f'{label}_emg'] = self.emg_data[label]
            save_dict[f'{label}_quaternion'] = self.quaternion_data[label]
        np.savez(path, **save_dict)
        
    def _load_test_data(self, path):
        """Load test data"""
        return np.load(path, allow_pickle=True)


if __name__ == '__main__':
    unittest.main() 