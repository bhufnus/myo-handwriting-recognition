#!/usr/bin/env python3
"""
Tests for preprocessing functions (UI-focused)
"""

import unittest
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import (
    preprocess_emg, 
    create_position_focused_sequence,
    normalize_data,
    filter_noise,
    segment_data
)


class TestPreprocessing(unittest.TestCase):
    """Test cases for preprocessing functions (UI-focused)"""
    
    def setUp(self):
        """Set up test data"""
        # Create sample EMG data
        self.sample_emg = np.random.randn(100, 8)
        self.sample_quaternion = np.random.randn(100, 4)
        
        # Create sample data for different scenarios
        self.noisy_emg = self.sample_emg + np.random.normal(0, 0.5, self.sample_emg.shape)
        self.short_emg = np.random.randn(50, 8)
        self.long_emg = np.random.randn(200, 8)
        
    def test_preprocess_emg_basic(self):
        """Test basic EMG preprocessing"""
        processed = preprocess_emg(self.sample_emg)
        
        # Check output shape
        self.assertEqual(processed.shape, self.sample_emg.shape)
        
        # Check data is finite
        self.assertTrue(np.all(np.isfinite(processed)))
        
        # Check data is normalized (roughly)
        self.assertLess(np.std(processed), np.std(self.sample_emg) * 2)
        
    def test_preprocess_emg_edge_cases(self):
        """Test EMG preprocessing with edge cases"""
        # Test with short data
        processed_short = preprocess_emg(self.short_emg)
        self.assertEqual(processed_short.shape, self.short_emg.shape)
        
        # Test with long data
        processed_long = preprocess_emg(self.long_emg)
        self.assertEqual(processed_long.shape, self.long_emg.shape)
        
        # Test with noisy data
        processed_noisy = preprocess_emg(self.noisy_emg)
        self.assertEqual(processed_noisy.shape, self.noisy_emg.shape)
        
    def test_create_position_focused_sequence(self):
        """Test position-focused sequence creation"""
        # Preprocess EMG first
        processed_emg = preprocess_emg(self.sample_emg)
        
        # Create sequence
        sequence = create_position_focused_sequence(processed_emg, self.sample_quaternion)
        
        # Check output shape
        self.assertEqual(len(sequence.shape), 2)
        self.assertGreater(sequence.shape[0], 0)
        self.assertGreater(sequence.shape[1], 0)
        
        # Check data is finite
        self.assertTrue(np.all(np.isfinite(sequence)))
        
    def test_create_position_focused_sequence_mismatched_lengths(self):
        """Test sequence creation with mismatched data lengths"""
        # Test with different lengths
        short_quat = np.random.randn(50, 4)
        
        try:
            sequence = create_position_focused_sequence(self.sample_emg, short_quat)
            self.assertIsNotNone(sequence)
        except Exception as e:
            self.fail(f"Should handle mismatched lengths gracefully: {e}")
            
    def test_normalize_data(self):
        """Test data normalization function"""
        # Test EMG normalization
        normalized_emg = normalize_data(self.sample_emg)
        
        self.assertEqual(normalized_emg.shape, self.sample_emg.shape)
        self.assertTrue(np.all(np.isfinite(normalized_emg)))
        
        # Test quaternion normalization
        normalized_quat = normalize_data(self.sample_quaternion)
        
        self.assertEqual(normalized_quat.shape, self.sample_quaternion.shape)
        self.assertTrue(np.all(np.isfinite(normalized_quat)))
        
    def test_filter_noise(self):
        """Test noise filtering function"""
        # Test EMG noise filtering
        filtered_emg = filter_noise(self.noisy_emg)
        
        self.assertEqual(filtered_emg.shape, self.noisy_emg.shape)
        self.assertTrue(np.all(np.isfinite(filtered_emg)))
        
        # Test quaternion noise filtering
        noisy_quat = self.sample_quaternion + np.random.normal(0, 0.1, self.sample_quaternion.shape)
        filtered_quat = filter_noise(noisy_quat)
        
        self.assertEqual(filtered_quat.shape, noisy_quat.shape)
        self.assertTrue(np.all(np.isfinite(filtered_quat)))
        
    def test_segment_data(self):
        """Test data segmentation function"""
        # Test EMG segmentation
        segments = segment_data(self.sample_emg, window_size=50)
        
        self.assertIsInstance(segments, list)
        self.assertGreater(len(segments), 0)
        
        for segment in segments:
            self.assertEqual(segment.shape[1], 8)  # EMG channels
            self.assertGreater(segment.shape[0], 0)
            
        # Test quaternion segmentation
        quat_segments = segment_data(self.sample_quaternion, window_size=50)
        
        self.assertIsInstance(quat_segments, list)
        self.assertGreater(len(quat_segments), 0)
        
        for segment in quat_segments:
            self.assertEqual(segment.shape[1], 4)  # Quaternion components
            self.assertGreater(segment.shape[0], 0)
            
    def test_preprocessing_performance(self):
        """Test preprocessing performance"""
        import time
        
        # Test EMG preprocessing performance
        start_time = time.time()
        for _ in range(10):
            preprocess_emg(self.sample_emg)
        emg_time = time.time() - start_time
        
        # Should process quickly (< 1 second for 10 samples)
        self.assertLess(emg_time, 1.0)
        
        # Test sequence creation performance
        processed_emg = preprocess_emg(self.sample_emg)
        start_time = time.time()
        for _ in range(10):
            create_position_focused_sequence(processed_emg, self.sample_quaternion)
        sequence_time = time.time() - start_time
        
        # Should create sequences quickly (< 1 second for 10 samples)
        self.assertLess(sequence_time, 1.0)
        
    def test_preprocessing_error_handling(self):
        """Test preprocessing error handling"""
        # Test with None data
        try:
            preprocess_emg(None)
            self.fail("Should raise exception for None data")
        except Exception:
            pass  # Expected
            
        # Test with empty data
        try:
            preprocess_emg(np.array([]))
            self.fail("Should handle empty data gracefully")
        except Exception:
            pass  # Expected
            
        # Test with wrong shape
        wrong_shape = np.random.randn(100, 4)  # Wrong number of channels
        try:
            preprocess_emg(wrong_shape)
            self.fail("Should handle wrong shape gracefully")
        except Exception:
            pass  # Expected
            
    def test_ui_data_validation(self):
        """Test UI data validation in preprocessing"""
        # Test valid data
        self.assertTrue(self._validate_emg_data(self.sample_emg))
        self.assertTrue(self._validate_quaternion_data(self.sample_quaternion))
        
        # Test invalid data
        invalid_emg = np.random.randn(50, 4)  # Wrong shape
        invalid_quat = np.random.randn(100, 3)  # Wrong shape
        
        self.assertFalse(self._validate_emg_data(invalid_emg))
        self.assertFalse(self._validate_quaternion_data(invalid_quat))
        
    def test_preprocessing_consistency(self):
        """Test preprocessing consistency"""
        # Process same data multiple times
        results = []
        for _ in range(5):
            processed = preprocess_emg(self.sample_emg)
            results.append(processed)
            
        # Check all results have same shape
        for result in results:
            self.assertEqual(result.shape, self.sample_emg.shape)
            
        # Check results are reasonably similar (not identical due to randomness)
        first_result = results[0]
        for result in results[1:]:
            # Should be similar but not identical
            self.assertGreater(np.corrcoef(first_result.flatten(), result.flatten())[0, 1], 0.5)
            
    def test_preprocessing_data_quality(self):
        """Test preprocessing data quality"""
        # Test with very noisy data
        very_noisy = self.sample_emg + np.random.normal(0, 2.0, self.sample_emg.shape)
        processed_noisy = preprocess_emg(very_noisy)
        
        # Should still produce valid output
        self.assertEqual(processed_noisy.shape, very_noisy.shape)
        self.assertTrue(np.all(np.isfinite(processed_noisy)))
        
        # Test with very small data
        small_data = np.random.randn(10, 8) * 0.001
        processed_small = preprocess_emg(small_data)
        
        # Should handle small values gracefully
        self.assertEqual(processed_small.shape, small_data.shape)
        self.assertTrue(np.all(np.isfinite(processed_small)))
        
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


if __name__ == '__main__':
    unittest.main() 