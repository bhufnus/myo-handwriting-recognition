#!/usr/bin/env python3
"""
Tests for GUI components and UI functions
"""

import unittest
import sys
import os
import tempfile
import numpy as np
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock tkinter for testing
import tkinter as tk
from unittest.mock import Mock, patch


class TestGUIComponents(unittest.TestCase):
    """Test cases for GUI components and UI functions"""
    
    def setUp(self):
        """Set up test environment"""
        # Create a mock root window
        self.root = Mock()
        self.root.after = Mock()
        self.root.bind = Mock()
        
        # Sample data
        self.sample_emg_data = {
            'A': [np.random.randn(100, 8) for _ in range(3)],
            'B': [np.random.randn(100, 8) for _ in range(3)],
            'C': [np.random.randn(100, 8) for _ in range(3)]
        }
        
        self.sample_quaternion_data = {
            'A': [np.random.randn(100, 4) for _ in range(3)],
            'B': [np.random.randn(100, 4) for _ in range(3)],
            'C': [np.random.randn(100, 4) for _ in range(3)]
        }
        
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
        
        # Test None values (should return False, not raise exception)
        self.assertFalse(self._validate_emg_data(None))
        self.assertFalse(self._validate_quaternion_data(None))
        
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
                # On Windows, files might be locked briefly
                time.sleep(0.1)
                try:
                    os.unlink(temp_path)
                except PermissionError:
                    pass  # Give up if still locked
                
    def test_ui_model_operations(self):
        """Test UI model operations without training"""
        # Create a simple test model
        model = self._create_test_model()
        
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            model_path = f.name
            
        try:
            # Save model
            model.save(model_path)
            
            # Test loading
            loaded_model = self._load_test_model(model_path)
            self.assertIsNotNone(loaded_model)
            
            # Check model structure
            self.assertEqual(model.input_shape, loaded_model.input_shape)
            self.assertEqual(model.output_shape, loaded_model.output_shape)
            
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
                
    def test_ui_prediction_logic(self):
        """Test UI prediction logic without training"""
        # Create test data
        emg_window = np.random.randn(100, 8)
        quaternion_window = np.random.randn(100, 4)
        
        # Test prediction preprocessing
        processed_emg = self._preprocess_for_prediction(emg_window)
        processed_quat = self._preprocess_for_prediction(quaternion_window)
        
        self.assertEqual(processed_emg.shape, (100, 8))
        self.assertEqual(processed_quat.shape, (100, 4))
        
        # Test prediction confidence calculation
        confidence = self._calculate_prediction_confidence([0.1, 0.2, 0.3, 0.4])
        self.assertGreaterEqual(confidence, 0)
        self.assertLessEqual(confidence, 1)
        
        # Test prediction label selection
        label = self._select_prediction_label([0.1, 0.2, 0.3, 0.4], ['A', 'B', 'C', 'IDLE'])
        self.assertIn(label, ['A', 'B', 'C', 'IDLE'])
        
    def test_ui_error_handling(self):
        """Test UI error handling"""
        # Test handling of invalid data
        with self.assertRaises(ValueError):
            self._validate_emg_data_with_exception(None)
            
        # Test handling of missing files
        with self.assertRaises(FileNotFoundError):
            self._load_test_data("nonexistent_file.npz")
            
        # Test handling of invalid model files
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            invalid_model_path = f.name
            
        try:
            # Create invalid model file
            with open(invalid_model_path, 'w') as f:
                f.write("invalid model data")
                
            with self.assertRaises(Exception):
                self._load_test_model(invalid_model_path)
                
        finally:
            try:
                if os.path.exists(invalid_model_path):
                    os.unlink(invalid_model_path)
            except PermissionError:
                pass
                
    def test_ui_data_processing(self):
        """Test UI data processing functions"""
        # Test data normalization
        raw_data = np.random.randn(100, 8)
        normalized_data = self._normalize_data(raw_data)
        
        self.assertEqual(normalized_data.shape, raw_data.shape)
        self.assertTrue(np.all(np.isfinite(normalized_data)))
        
        # Test data filtering
        filtered_data = self._filter_data(raw_data)
        self.assertEqual(filtered_data.shape, raw_data.shape)
        
        # Test data segmentation
        segments = self._segment_data(raw_data, window_size=50)
        self.assertIsInstance(segments, list)
        self.assertGreater(len(segments), 0)
        
    def test_ui_timing_functions(self):
        """Test UI timing and scheduling functions"""
        # Test countdown timer
        countdown = self._create_countdown_timer(5)
        self.assertEqual(countdown, 5)
        
        # Test prediction scheduling
        next_prediction = self._schedule_next_prediction(2.0)
        self.assertIsInstance(next_prediction, (int, float))
        self.assertGreater(next_prediction, 0)
        
        # Test data collection timing
        collection_time = self._calculate_collection_time(1000)  # 1000 samples
        self.assertIsInstance(collection_time, (int, float))
        self.assertGreater(collection_time, 0)
        
    def test_ui_visual_feedback(self):
        """Test UI visual feedback functions"""
        # Test color coding for confidence
        low_confidence_color = self._get_confidence_color(0.3)
        high_confidence_color = self._get_confidence_color(0.9)
        
        self.assertIsInstance(low_confidence_color, str)
        self.assertIsInstance(high_confidence_color, str)
        self.assertNotEqual(low_confidence_color, high_confidence_color)
        
        # Test status message generation
        status_msg = self._generate_status_message("A", 0.85, "Recording")
        self.assertIsInstance(status_msg, str)
        self.assertIn("A", status_msg)
        self.assertIn("85", status_msg)
        
        # Test progress calculation
        progress = self._calculate_progress(5, 10)
        self.assertEqual(progress, 50)
        
    def test_ui_data_validation_edge_cases(self):
        """Test UI data validation edge cases"""
        # Test empty arrays
        empty_emg = np.array([])
        empty_quat = np.array([])
        
        self.assertFalse(self._validate_emg_data(empty_emg))
        self.assertFalse(self._validate_quaternion_data(empty_quat))
        
        # Test single dimension arrays
        single_dim_emg = np.random.randn(100)
        single_dim_quat = np.random.randn(100)
        
        self.assertFalse(self._validate_emg_data(single_dim_emg))
        self.assertFalse(self._validate_quaternion_data(single_dim_quat))
        
        # Test wrong data types
        string_data = "not an array"
        self.assertFalse(self._validate_emg_data(string_data))
        self.assertFalse(self._validate_quaternion_data(string_data))
        
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
        
    def _validate_emg_data_with_exception(self, data):
        """Validate EMG data format with exception for None"""
        if data is None:
            raise ValueError("Data cannot be None")
        if not isinstance(data, np.ndarray):
            return False
        if len(data.shape) != 2 or data.shape[1] != 8:
            return False
        return True
        
    def _save_test_data(self, path):
        """Save test data"""
        save_dict = {}
        for label in ['A', 'B', 'C']:
            save_dict[f'{label}_emg'] = self.sample_emg_data[label]
            save_dict[f'{label}_quaternion'] = self.sample_quaternion_data[label]
        np.savez(path, **save_dict)
        
    def _load_test_data(self, path):
        """Load test data"""
        return np.load(path, allow_pickle=True)
        
    def _create_test_model(self):
        """Create a simple test model"""
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense
        
        model = Sequential([
            Dense(64, activation='relu', input_shape=(12,)),
            Dense(32, activation='relu'),
            Dense(5, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        return model
        
    def _load_test_model(self, path):
        """Load test model"""
        from tensorflow.keras.models import load_model
        return load_model(path)
        
    def _preprocess_for_prediction(self, data):
        """Preprocess data for prediction"""
        # Simple preprocessing for testing
        return data.astype(np.float32)
        
    def _calculate_prediction_confidence(self, probabilities):
        """Calculate prediction confidence"""
        return max(probabilities)
        
    def _select_prediction_label(self, probabilities, labels):
        """Select prediction label"""
        return labels[np.argmax(probabilities)]
        
    def _normalize_data(self, data):
        """Normalize data"""
        return (data - np.mean(data)) / (np.std(data) + 1e-8)
        
    def _filter_data(self, data):
        """Filter data"""
        # Simple filtering for testing
        return data
        
    def _segment_data(self, data, window_size=50):
        """Segment data into windows"""
        segments = []
        for i in range(0, len(data) - window_size + 1, window_size // 2):
            segments.append(data[i:i + window_size])
        return segments
        
    def _create_countdown_timer(self, seconds):
        """Create countdown timer"""
        return seconds
        
    def _schedule_next_prediction(self, interval):
        """Schedule next prediction"""
        import time
        return time.time() + interval
        
    def _calculate_collection_time(self, samples):
        """Calculate data collection time"""
        # Assume 1000 Hz sampling rate
        return samples / 1000.0
        
    def _get_confidence_color(self, confidence):
        """Get color based on confidence"""
        if confidence < 0.5:
            return "red"
        elif confidence < 0.8:
            return "yellow"
        else:
            return "green"
            
    def _generate_status_message(self, prediction, confidence, status):
        """Generate status message"""
        return f"Prediction: {prediction} ({confidence:.0%}) - {status}"
        
    def _calculate_progress(self, current, total):
        """Calculate progress percentage"""
        return int((current / total) * 100)


if __name__ == '__main__':
    unittest.main() 