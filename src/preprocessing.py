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
    normalized = (filtered - np.mean(filtered)) / np.std(filtered)
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

def extract_all_features(emg_window, quaternion_window):
    """Extract and concatenate features from EMG and quaternion windows."""
    emg_feat = extract_emg_features(emg_window)
    quaternion_feat = extract_quaternion_features(quaternion_window)
    return np.concatenate([emg_feat, quaternion_feat])