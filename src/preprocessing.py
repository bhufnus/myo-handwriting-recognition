# EMG and Quaternion preprocessing and feature extraction
# src/preprocessing.py
import numpy as np
from scipy.signal import butter, filtfilt

def butter_bandpass(lowcut, highcut, fs, order=5):
    """Design a Butterworth bandpass filter."""
    nyq = 0.5 * fs
    if lowcut <= 0 or highcut >= nyq:
        raise ValueError(f"Cutoff frequencies must be 0 < lowcut < {nyq}, highcut < {nyq}. Got lowcut={lowcut}, highcut={highcut}.")
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def preprocess_emg(emg_data, fs=200, lowcut=20, highcut=90):
    """Preprocess EMG data with filtering and normalization."""
    if len(emg_data) == 0:
        print("Warning: No EMG data to preprocess")
        return emg_data
    b, a = butter_bandpass(lowcut, highcut, fs)
    filtered = filtfilt(b, a, emg_data, axis=0)
    
    # Handle zero variance case to prevent NaN
    std_val = np.std(filtered)
    if std_val < 1e-10:  # Very small standard deviation
        # Return zeros for static data (idle state)
        return np.zeros_like(filtered)
    else:
        normalized = (filtered - np.mean(filtered)) / std_val
        return normalized

def extract_emg_features(emg_window):
    """Extract RMS and MAV features from EMG window."""
    if len(emg_window) == 0:
        return np.zeros(16)
    rms = np.sqrt(np.mean(emg_window**2, axis=0))
    mav = np.mean(np.abs(emg_window), axis=0)
    return np.concatenate([rms, mav])

def extract_quaternion_features(quaternion_window):
    """Extract features from quaternion window."""
    if len(quaternion_window) == 0:
        return np.zeros(16)
    quaternion_window = np.array(quaternion_window)
    mean = np.mean(quaternion_window, axis=0)
    std = np.std(quaternion_window, axis=0)
    min_ = np.min(quaternion_window, axis=0)
    max_ = np.max(quaternion_window, axis=0)
    return np.concatenate([mean, std, min_, max_])

def extract_quaternion_only_features(quaternion_window):
    """Extract enhanced features from quaternion window only (no EMG)."""
    if len(quaternion_window) == 0:
        return np.zeros(24)  # Increased feature count
    quaternion_window = np.array(quaternion_window)
    
    # Basic statistics
    mean = np.mean(quaternion_window, axis=0)
    std = np.std(quaternion_window, axis=0)
    min_ = np.min(quaternion_window, axis=0)
    max_ = np.max(quaternion_window, axis=0)
    
    # Additional features
    range_ = max_ - min_
    variance = np.var(quaternion_window, axis=0)
    
    # Rate of change (velocity-like features)
    if len(quaternion_window) > 1:
        diff = np.diff(quaternion_window, axis=0)
        diff_mean = np.mean(diff, axis=0)
        diff_std = np.std(diff, axis=0)
    else:
        diff_mean = np.zeros(4)
        diff_std = np.zeros(4)
    
    return np.concatenate([mean, std, min_, max_, range_, variance, diff_mean, diff_std])

def extract_all_features(emg_window, quaternion_window):
    """Extract and concatenate features from EMG and quaternion windows."""
    emg_feat = extract_emg_features(emg_window)
    quaternion_feat = extract_quaternion_features(quaternion_window)
    return np.concatenate([emg_feat, quaternion_feat])

def create_weighted_sequence(emg_data, quaternion_data, emg_weight=0.3, quaternion_weight=0.7):
    """
    Create a weighted combination of EMG and quaternion data for sequence input.
    
    Args:
        emg_data: Preprocessed EMG data (window_size, 8)
        quaternion_data: Raw quaternion data (window_size, 4)
        emg_weight: Weight for EMG data (0.0 to 1.0)
        quaternion_weight: Weight for quaternion data (0.0 to 1.0)
    
    Returns:
        Weighted sequence data (window_size, 12)
    """
    if len(emg_data) == 0 or len(quaternion_data) == 0:
        print("Warning: Empty EMG or quaternion data")
        return np.zeros((100, 12))  # Default size
    
    # Ensure both have the same length
    min_len = min(len(emg_data), len(quaternion_data))
    emg_data = emg_data[:min_len]
    quaternion_data = quaternion_data[:min_len]
    
    # Normalize weights to sum to 1
    total_weight = emg_weight + quaternion_weight
    emg_weight = emg_weight / total_weight
    quaternion_weight = quaternion_weight / total_weight
    
    # Apply weights to the data
    weighted_emg = emg_data * emg_weight
    weighted_quaternion = quaternion_data * quaternion_weight
    
    # Concatenate weighted data
    weighted_sequence = np.concatenate([weighted_emg, weighted_quaternion], axis=1)
    
    return weighted_sequence

def create_position_focused_sequence(emg_data, quaternion_data, position_emphasis=0.8):
    """
    Create a sequence with emphasis on position data (quaternion).
    
    Args:
        emg_data: Preprocessed EMG data (window_size, 8)
        quaternion_data: Raw quaternion data (window_size, 4)
        position_emphasis: How much to emphasize position (0.0 to 1.0)
                          - 0.0: Equal weighting
                          - 0.8: Heavy emphasis on position (recommended)
                          - 1.0: Position only
    
    Returns:
        Position-focused sequence data (window_size, 12)
    """
    emg_weight = 1.0 - position_emphasis
    quaternion_weight = position_emphasis
    
    return create_weighted_sequence(emg_data, quaternion_data, emg_weight, quaternion_weight)