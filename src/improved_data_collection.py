# Improved Data Collection System
# src/improved_data_collection.py
import numpy as np
import time
import threading
from collections import deque
from scipy import signal
from scipy.stats import zscore
import warnings
warnings.filterwarnings('ignore')

class DataQualityChecker:
    """Checks data quality during collection to ensure good training data."""
    
    def __init__(self):
        self.min_emg_variance = 0.5  # Minimum EMG variance for non-idle
        self.max_emg_variance = 50.0  # Maximum EMG variance (avoid noise)
        self.min_quaternion_variance = 0.01  # Minimum quaternion movement
        self.max_quaternion_variance = 2.0  # Maximum quaternion movement
        self.min_duration = 0.8  # Minimum recording duration (seconds)
        self.max_duration = 3.0  # Maximum recording duration (seconds)
        
    def check_emg_quality(self, emg_data):
        """Check if EMG data has sufficient activity."""
        if len(emg_data) == 0:
            return False, "No EMG data"
            
        # Calculate variance for each channel
        variances = np.var(emg_data, axis=0)
        mean_variance = np.mean(variances)
        
        if mean_variance < self.min_emg_variance:
            return False, f"EMG too static (variance: {mean_variance:.2f})"
        elif mean_variance > self.max_emg_variance:
            return False, f"EMG too noisy (variance: {mean_variance:.2f})"
            
        return True, f"EMG OK (variance: {mean_variance:.2f})"
    
    def check_quaternion_quality(self, quaternion_data):
        """Check if quaternion data shows sufficient movement."""
        if len(quaternion_data) == 0:
            return False, "No quaternion data"
            
        # Calculate variance for each component
        variances = np.var(quaternion_data, axis=0)
        mean_variance = np.mean(variances)
        
        if mean_variance < self.min_quaternion_variance:
            return False, f"Quaternion too static (variance: {mean_variance:.4f})"
        elif mean_variance > self.max_quaternion_variance:
            return False, f"Quaternion too noisy (variance: {mean_variance:.4f})"
            
        return True, f"Quaternion OK (variance: {mean_variance:.4f})"
    
    def check_duration(self, duration_seconds):
        """Check if recording duration is appropriate."""
        if duration_seconds < self.min_duration:
            return False, f"Recording too short ({duration_seconds:.1f}s)"
        elif duration_seconds > self.max_duration:
            return False, f"Recording too long ({duration_seconds:.1f}s)"
        return True, f"Duration OK ({duration_seconds:.1f}s)"
    
    def check_overall_quality(self, emg_data, quaternion_data, duration_seconds):
        """Comprehensive quality check."""
        results = []
        
        # Check EMG
        emg_ok, emg_msg = self.check_emg_quality(emg_data)
        results.append(("EMG", emg_ok, emg_msg))
        
        # Check quaternion
        quat_ok, quat_msg = self.check_quaternion_quality(quaternion_data)
        results.append(("Quaternion", quat_ok, quat_msg))
        
        # Check duration
        dur_ok, dur_msg = self.check_duration(duration_seconds)
        results.append(("Duration", dur_ok, dur_msg))
        
        # Overall assessment
        all_ok = all(ok for _, ok, _ in results)
        return all_ok, results

class ImprovedDataProcessor:
    """Processes and standardizes data for better training."""
    
    def __init__(self, target_length=100, sample_rate=200):
        self.target_length = target_length
        self.sample_rate = sample_rate
        
    def standardize_data(self, emg_data, quaternion_data):
        """Standardize data to fixed length with proper preprocessing."""
        if len(emg_data) == 0 or len(quaternion_data) == 0:
            return None, None, "No data to process"
            
        # Get minimum length
        min_len = min(len(emg_data), len(quaternion_data))
        if min_len < 50:  # Too short
            return None, None, f"Data too short ({min_len} samples)"
            
        # Truncate to same length
        emg_data = np.array(emg_data[:min_len])
        quaternion_data = np.array(quaternion_data[:min_len])
        
        # Apply preprocessing
        emg_processed = self.preprocess_emg(emg_data)
        quaternion_processed = self.preprocess_quaternion(quaternion_data)
        
        # Resize to target length
        emg_standardized = self.resize_to_target(emg_processed)
        quaternion_standardized = self.resize_to_target(quaternion_processed)
        
        return emg_standardized, quaternion_standardized, "OK"
    
    def preprocess_emg(self, emg_data):
        """Enhanced EMG preprocessing."""
        # Remove DC offset
        emg_centered = emg_data - np.mean(emg_data, axis=0)
        
        # Apply bandpass filter (20-90 Hz)
        b, a = signal.butter(4, [20, 90], btype='band', fs=self.sample_rate)
        emg_filtered = signal.filtfilt(b, a, emg_centered, axis=0)
        
        # Normalize
        std_vals = np.std(emg_filtered, axis=0)
        std_vals = np.where(std_vals < 1e-10, 1.0, std_vals)  # Avoid division by zero
        emg_normalized = emg_filtered / std_vals
        
        return emg_normalized
    
    def preprocess_quaternion(self, quaternion_data):
        """Enhanced quaternion preprocessing."""
        # Normalize quaternions to unit length
        norms = np.linalg.norm(quaternion_data, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)  # Avoid division by zero
        quaternion_normalized = quaternion_data / norms
        
        # Remove DC offset
        quaternion_centered = quaternion_normalized - np.mean(quaternion_normalized, axis=0)
        
        return quaternion_centered
    
    def resize_to_target(self, data):
        """Resize data to target length using interpolation."""
        if len(data) == self.target_length:
            return data
            
        # Use linear interpolation to resize
        old_indices = np.linspace(0, len(data) - 1, len(data))
        new_indices = np.linspace(0, len(data) - 1, self.target_length)
        
        resized_data = np.zeros((self.target_length, data.shape[1]))
        for col in range(data.shape[1]):
            resized_data[:, col] = np.interp(new_indices, old_indices, data[:, col])
            
        return resized_data

class DataAugmenter:
    """Augments training data to improve model robustness."""
    
    def __init__(self):
        self.noise_level = 0.01
        self.time_shift_range = 0.1  # 10% time shift
        self.amplitude_scale_range = (0.9, 1.1)  # 10% amplitude variation
        
    def augment_sample(self, emg_data, quaternion_data, num_augmentations=3):
        """Create augmented versions of a sample."""
        augmented_samples = []
        
        for i in range(num_augmentations):
            # Original sample
            if i == 0:
                augmented_samples.append((emg_data, quaternion_data))
                continue
                
            # Add noise
            emg_noisy = emg_data + np.random.normal(0, self.noise_level, emg_data.shape)
            quat_noisy = quaternion_data + np.random.normal(0, self.noise_level * 0.1, quaternion_data.shape)
            
            # Time shift (circular shift)
            shift = int(np.random.uniform(-self.time_shift_range, self.time_shift_range) * len(emg_data))
            emg_shifted = np.roll(emg_noisy, shift, axis=0)
            quat_shifted = np.roll(quat_noisy, shift, axis=0)
            
            # Amplitude scaling
            scale_factor = np.random.uniform(*self.amplitude_scale_range)
            emg_scaled = emg_shifted * scale_factor
            quat_scaled = quat_shifted * scale_factor
            
            augmented_samples.append((emg_scaled, quat_scaled))
            
        return augmented_samples

class MovementGuide:
    """Provides guidance for consistent movement patterns."""
    
    def __init__(self):
        self.movement_patterns = {
            'A': {
                'description': 'Up-Down-Up pattern (like writing "A")',
                'steps': ['Start with arm down', 'Move up', 'Move down', 'Move up again'],
                'duration': '2 seconds',
                'key_points': ['Peak at 25%', 'Valley at 50%', 'Peak at 75%']
            },
            'B': {
                'description': 'Vertical line + two curves (like writing "B")',
                'steps': ['Start at top', 'Draw vertical line down', 'Curve right', 'Curve left'],
                'duration': '2 seconds',
                'key_points': ['Vertical line', 'Right curve', 'Left curve']
            },
            'C': {
                'description': 'Smooth curved movement (like writing "C")',
                'steps': ['Start at top-right', 'Draw smooth curve left', 'End at bottom'],
                'duration': '2 seconds',
                'key_points': ['Smooth curve', 'No sharp angles']
            },
            'IDLE': {
                'description': 'Keep arm completely still',
                'steps': ['Relax arm', 'Hold position', 'Don\'t move'],
                'duration': '2 seconds',
                'key_points': ['Minimal movement', 'Low EMG variance']
            },
            'NOISE': {
                'description': 'Random movements (for robustness)',
                'steps': ['Move arm randomly', 'Different directions', 'Varying speeds'],
                'duration': '2 seconds',
                'key_points': ['Random patterns', 'High variance']
            }
        }
    
    def get_guidance(self, label):
        """Get movement guidance for a specific label."""
        if label not in self.movement_patterns:
            return None
            
        pattern = self.movement_patterns[label]
        return {
            'label': label,
            'description': pattern['description'],
            'steps': pattern['steps'],
            'duration': pattern['duration'],
            'key_points': pattern['key_points']
        }
    
    def get_all_guidance(self):
        """Get guidance for all labels."""
        return {label: self.get_guidance(label) for label in self.movement_patterns.keys()}

class TrainingDataManager:
    """Manages training data collection and organization."""
    
    def __init__(self, labels=['A', 'B', 'C', 'IDLE', 'NOISE']):
        self.labels = labels
        self.quality_checker = DataQualityChecker()
        self.data_processor = ImprovedDataProcessor()
        self.data_augmenter = DataAugmenter()
        self.movement_guide = MovementGuide()
        
        # Data storage
        self.raw_data = {label: [] for label in labels}
        self.processed_data = {label: [] for label in labels}
        self.quality_scores = {label: [] for label in labels}
        self.collection_metadata = {label: [] for label in labels}
        
    def add_sample(self, label, emg_data, quaternion_data, duration_seconds, metadata=None):
        """Add a new sample with quality checking."""
        if label not in self.labels:
            print(f"Warning: Unknown label '{label}'")
            return False, "Unknown label"
            
        # Quality check
        quality_ok, quality_results = self.quality_checker.check_overall_quality(
            emg_data, quaternion_data, duration_seconds
        )
        
        # Process data
        emg_processed, quat_processed, process_msg = self.data_processor.standardize_data(
            emg_data, quaternion_data
        )
        
        if not quality_ok:
            return False, f"Quality check failed: {quality_results}"
            
        if emg_processed is None:
            return False, f"Processing failed: {process_msg}"
            
        # Store raw data
        self.raw_data[label].append({
            'emg': emg_data,
            'quaternion': quaternion_data,
            'duration': duration_seconds,
            'metadata': metadata or {}
        })
        
        # Store processed data
        self.processed_data[label].append({
            'emg': emg_processed,
            'quaternion': quat_processed,
            'quality_scores': quality_results,
            'metadata': metadata or {}
        })
        
        # Store quality scores
        self.quality_scores[label].append(quality_results)
        
        # Store collection metadata
        self.collection_metadata[label].append({
            'timestamp': time.time(),
            'duration': duration_seconds,
            'quality_ok': quality_ok,
            'process_msg': process_msg
        })
        
        return True, f"Sample added successfully. Quality: {quality_results}"
    
    def get_sample_count(self, label=None):
        """Get sample count for a label or all labels."""
        if label:
            return len(self.processed_data.get(label, []))
        return {label: len(self.processed_data.get(label, [])) for label in self.labels}
    
    def get_quality_summary(self, label=None):
        """Get quality summary for collected data."""
        if label:
            samples = self.quality_scores.get(label, [])
            if not samples:
                return f"No samples for {label}"
            
            quality_ok_count = sum(1 for sample in samples if all(ok for _, ok, _ in sample))
            total_count = len(samples)
            
            return f"{label}: {quality_ok_count}/{total_count} samples passed quality check"
        
        summaries = []
        for label in self.labels:
            summaries.append(self.get_quality_summary(label))
        return "\n".join(summaries)
    
    def prepare_training_data(self, augment=True):
        """Prepare data for training with optional augmentation."""
        all_samples = []
        all_labels = []
        
        for label in self.labels:
            samples = self.processed_data[label]
            
            for sample in samples:
                emg_data = sample['emg']
                quaternion_data = sample['quaternion']
                
                # Create base sample
                all_samples.append((emg_data, quaternion_data))
                all_labels.append(label)
                
                # Add augmented samples
                if augment:
                    augmented = self.data_augmenter.augment_sample(emg_data, quaternion_data)
                    for aug_emg, aug_quat in augmented[1:]:  # Skip first (original)
                        all_samples.append((aug_emg, aug_quat))
                        all_labels.append(label)
        
        return all_samples, all_labels
    
    def save_data(self, filename):
        """Save collected data to file."""
        import pickle
        
        save_data = {
            'raw_data': self.raw_data,
            'processed_data': self.processed_data,
            'quality_scores': self.quality_scores,
            'collection_metadata': self.collection_metadata,
            'labels': self.labels
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"Data saved to {filename}")
    
    def load_data(self, filename):
        """Load collected data from file."""
        import pickle
        
        with open(filename, 'rb') as f:
            save_data = pickle.load(f)
        
        self.raw_data = save_data['raw_data']
        self.processed_data = save_data['processed_data']
        self.quality_scores = save_data['quality_scores']
        self.collection_metadata = save_data['collection_metadata']
        self.labels = save_data['labels']
        
        print(f"Data loaded from {filename}") 