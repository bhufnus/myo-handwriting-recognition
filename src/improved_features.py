#!/usr/bin/env python3
"""
Improved feature extraction for gesture recognition
Implements the 5 better approaches instead of relying solely on EMG variance
"""
import numpy as np
from scipy.fft import fft
from scipy.signal import correlate
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

def extract_temporal_features(emg_data, quaternion_data):
    """
    1. Temporal patterns - How signals change over time
    """
    features = []
    
    # EMG temporal features
    emg_diff = np.diff(emg_data, axis=0)
    emg_accel = np.diff(emg_diff, axis=0)  # Second derivative
    
    # Rate of change
    features.append(np.mean(np.abs(emg_diff)))  # Average change rate
    features.append(np.std(emg_diff))  # Change rate variability
    features.append(np.mean(np.abs(emg_accel)))  # Acceleration rate
    
    # Quaternion temporal features
    quat_diff = np.diff(quaternion_data, axis=0)
    quat_accel = np.diff(quat_diff, axis=0)
    
    features.append(np.mean(np.abs(quat_diff)))
    features.append(np.std(quat_diff))
    features.append(np.mean(np.abs(quat_accel)))
    
    return features

def extract_frequency_features(emg_data, quaternion_data, sample_rate=200):
    """
    2. Frequency domain analysis - Different gestures have different frequency signatures
    """
    features = []
    
    # EMG frequency analysis
    for channel in range(emg_data.shape[1]):
        emg_channel = emg_data[:, channel]
        emg_fft = np.abs(fft(emg_channel))
        
        # Frequency domain features
        dominant_freq = np.argmax(emg_fft[:len(emg_fft)//2])
        spectral_centroid = np.sum(np.arange(len(emg_fft)//2) * emg_fft[:len(emg_fft)//2]) / np.sum(emg_fft[:len(emg_fft)//2])
        spectral_bandwidth = np.sqrt(np.sum((np.arange(len(emg_fft)//2) - spectral_centroid)**2 * emg_fft[:len(emg_fft)//2]) / np.sum(emg_fft[:len(emg_fft)//2]))
        
        features.extend([dominant_freq, spectral_centroid, spectral_bandwidth])
    
    # Quaternion frequency analysis
    for channel in range(quaternion_data.shape[1]):
        quat_channel = quaternion_data[:, channel]
        quat_fft = np.abs(fft(quat_channel))
        
        dominant_freq = np.argmax(quat_fft[:len(quat_fft)//2])
        spectral_centroid = np.sum(np.arange(len(quat_fft)//2) * quat_fft[:len(quat_fft)//2]) / np.sum(quat_fft[:len(quat_fft)//2])
        
        features.extend([dominant_freq, spectral_centroid])
    
    return features

def extract_correlation_features(emg_data, quaternion_data):
    """
    3. Cross-correlation between channels - How EMG channels relate to each other
    """
    features = []
    
    # EMG channel correlations
    emg_corr_matrix = np.corrcoef(emg_data.T)
    # Get upper triangle (excluding diagonal)
    upper_triangle = emg_corr_matrix[np.triu_indices_from(emg_corr_matrix, k=1)]
    features.extend([np.mean(upper_triangle), np.std(upper_triangle), np.min(upper_triangle), np.max(upper_triangle)])
    
    # Quaternion channel correlations
    quat_corr_matrix = np.corrcoef(quaternion_data.T)
    upper_triangle = quat_corr_matrix[np.triu_indices_from(quat_corr_matrix, k=1)]
    features.extend([np.mean(upper_triangle), np.std(upper_triangle), np.min(upper_triangle), np.max(upper_triangle)])
    
    # Cross-correlation between EMG and quaternion
    for emg_ch in range(min(4, emg_data.shape[1])):  # Use first 4 EMG channels
        for quat_ch in range(quaternion_data.shape[1]):
            corr, _ = pearsonr(emg_data[:, emg_ch], quaternion_data[:, quat_ch])
            features.append(corr)
    
    return features

def extract_gesture_specific_features(emg_data, quaternion_data):
    """
    4. Gesture-specific movement patterns - A, B, C have different orientation changes
    """
    features = []
    
    # A gesture: typically involves up-down movement
    # Look for vertical orientation changes
    quat_vertical = quaternion_data[:, 1]  # Y component
    vertical_movement = np.std(quat_vertical)
    vertical_range = np.ptp(quat_vertical)
    vertical_zero_crossings = np.sum(np.diff(np.sign(quat_vertical)) != 0)
    
    # B gesture: typically involves horizontal movement
    # Look for horizontal orientation changes
    quat_horizontal = quaternion_data[:, 0]  # X component
    horizontal_movement = np.std(quat_horizontal)
    horizontal_range = np.ptp(quat_horizontal)
    horizontal_zero_crossings = np.sum(np.diff(np.sign(quat_horizontal)) != 0)
    
    # C gesture: typically involves circular movement
    # Look for rotational changes
    quat_rotation = quaternion_data[:, 3]  # W component
    rotational_movement = np.std(quat_rotation)
    rotational_range = np.ptp(quat_rotation)
    rotational_zero_crossings = np.sum(np.diff(np.sign(quat_rotation)) != 0)
    
    # Z-axis movement (depth)
    quat_depth = quaternion_data[:, 2]  # Z component
    depth_movement = np.std(quat_depth)
    depth_range = np.ptp(quat_depth)
    depth_zero_crossings = np.sum(np.diff(np.sign(quat_depth)) != 0)
    
    features.extend([
        vertical_movement, vertical_range, vertical_zero_crossings,
        horizontal_movement, horizontal_range, horizontal_zero_crossings,
        rotational_movement, rotational_range, rotational_zero_crossings,
        depth_movement, depth_range, depth_zero_crossings
    ])
    
    return features

def extract_idle_detection_features(emg_data, quaternion_data):
    """
    5. Multiple criteria for idle detection - Not just variance
    """
    features = []
    
    # 1. Autocorrelation (idle should be more random)
    emg_autocorr = correlate(emg_data[:, 0], emg_data[:, 0], mode='full')
    autocorr_peak = np.max(emg_autocorr[len(emg_autocorr)//2:])
    autocorr_ratio = autocorr_peak / (np.sum(emg_autocorr) + 1e-10)
    
    # 2. Trend analysis (idle should have no strong trend)
    emg_trend = np.polyfit(np.arange(len(emg_data)), emg_data[:, 0], 1)[0]
    quat_trend = np.polyfit(np.arange(len(quaternion_data)), quaternion_data[:, 0], 1)[0]
    
    # 3. Frequency ratio (idle should not be dominated by low frequencies)
    emg_fft = np.abs(fft(emg_data[:, 0]))
    low_freq_power = np.sum(emg_fft[:len(emg_fft)//4])
    high_freq_power = np.sum(emg_fft[len(emg_fft)//4:])
    freq_ratio = low_freq_power / (high_freq_power + 1e-10)
    
    # 4. Entropy (idle should have higher entropy/more randomness)
    emg_hist, _ = np.histogram(emg_data[:, 0], bins=20)
    emg_hist = emg_hist / np.sum(emg_hist)
    emg_entropy = -np.sum(emg_hist * np.log(emg_hist + 1e-10))
    
    # 5. Stationarity (idle should be more stationary)
    # Split data into segments and compare statistics
    segment_size = len(emg_data) // 4
    segment_stds = []
    for i in range(4):
        start_idx = i * segment_size
        end_idx = start_idx + segment_size
        segment_stds.append(np.std(emg_data[start_idx:end_idx, 0]))
    stationarity = np.std(segment_stds)  # Lower = more stationary
    
    features.extend([
        autocorr_peak, autocorr_ratio, emg_trend, quat_trend,
        freq_ratio, emg_entropy, stationarity
    ])
    
    return features

def extract_all_improved_features(emg_data, quaternion_data):
    """
    Extract all improved features using the 5 approaches
    """
    features = []
    
    # 1. Temporal patterns
    temporal_features = extract_temporal_features(emg_data, quaternion_data)
    features.extend(temporal_features)
    
    # 2. Frequency domain analysis
    freq_features = extract_frequency_features(emg_data, quaternion_data)
    features.extend(freq_features)
    
    # 3. Cross-correlation features
    corr_features = extract_correlation_features(emg_data, quaternion_data)
    features.extend(corr_features)
    
    # 4. Gesture-specific features
    gesture_features = extract_gesture_specific_features(emg_data, quaternion_data)
    features.extend(gesture_features)
    
    # 5. Idle detection features
    idle_features = extract_idle_detection_features(emg_data, quaternion_data)
    features.extend(idle_features)
    
    return np.array(features, dtype=np.float32)

def analyze_feature_importance(features, labels):
    """
    Analyze which features are most important for classification
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import mutual_info_classif
    
    # Train a random forest to get feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(features, labels)
    
    # Get feature importance
    importance = rf.feature_importances_
    
    # Get mutual information scores
    mi_scores = mutual_info_classif(features, labels, random_state=42)
    
    return importance, mi_scores

def print_feature_analysis(emg_data, quaternion_data, class_name):
    """
    Print detailed analysis of features for a sample
    """
    print(f"\n=== Feature Analysis for {class_name} ===")
    
    # Extract all features
    features = extract_all_improved_features(emg_data, quaternion_data)
    
    # Basic statistics
    print(f"Total features: {len(features)}")
    print(f"Feature range: {np.min(features):.4f} to {np.max(features):.4f}")
    print(f"Feature mean: {np.mean(features):.4f}")
    print(f"Feature std: {np.std(features):.4f}")
    
    # Check for NaN or Inf values
    nan_count = np.sum(np.isnan(features))
    inf_count = np.sum(np.isinf(features))
    print(f"NaN values: {nan_count}")
    print(f"Inf values: {inf_count}")
    
    if nan_count > 0 or inf_count > 0:
        print("⚠️ Warning: Features contain NaN or Inf values!")
    
    return features 