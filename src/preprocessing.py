# EMG preprocessing and feature extraction
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

def extract_features(emg_window):
    """Extract RMS and MAV features from EMG window."""
    if len(emg_window) == 0:
        return np.zeros(16)
    rms = np.sqrt(np.mean(emg_window**2, axis=0))
    mav = np.mean(np.abs(emg_window), axis=0)
    return np.concatenate([rms, mav])