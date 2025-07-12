#!/usr/bin/env python3
"""
Improved approaches for gesture recognition that don't rely solely on EMG variance
"""
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def analyze_gesture_patterns(emg_data, quaternion_data):
    """
    Analyze gesture patterns using multiple features, not just variance
    """
    print("=== Improved Gesture Analysis ===")
    
    # 1. Temporal patterns (how signals change over time)
    emg_diff = np.diff(emg_data, axis=0)
    quat_diff = np.diff(quaternion_data, axis=0)
    
    print(f"Temporal Analysis:")
    print(f"  EMG change rate: {np.mean(np.abs(emg_diff)):.2f}")
    print(f"  Quaternion change rate: {np.mean(np.abs(quat_diff)):.4f}")
    
    # 2. Frequency domain analysis
    from scipy.fft import fft
    emg_fft = np.abs(fft(emg_data, axis=0))
    dominant_freq = np.argmax(np.mean(emg_fft, axis=1))
    print(f"  Dominant frequency: {dominant_freq} Hz")
    
    # 3. Cross-correlation between channels
    emg_corr = np.corrcoef(emg_data.T)
    print(f"  EMG channel correlation: {np.mean(emg_corr):.3f}")
    
    # 4. Gesture-specific features
    # Look for patterns that indicate intentional movement vs. noise
    
    return {
        'temporal_change': np.mean(np.abs(emg_diff)),
        'orientation_change': np.mean(np.abs(quat_diff)),
        'frequency_content': dominant_freq,
        'channel_correlation': np.mean(emg_corr)
    }

def better_idle_detection(emg_data, quaternion_data):
    """
    Better idle detection using multiple criteria
    """
    print("\n=== Better Idle Detection ===")
    
    # 1. Check for consistent patterns (idle should be more random)
    emg_autocorr = np.correlate(emg_data[:, 0], emg_data[:, 0], mode='full')
    autocorr_peak = np.max(emg_autocorr[len(emg_autocorr)//2:])
    
    # 2. Check for muscle fatigue patterns (gradual changes)
    emg_trend = np.polyfit(np.arange(len(emg_data)), emg_data[:, 0], 1)[0]
    
    # 3. Check for intentional movement patterns
    # Intentional movements often have specific frequency characteristics
    from scipy.fft import fft
    emg_fft = np.abs(fft(emg_data[:, 0]))
    low_freq_power = np.sum(emg_fft[:len(emg_fft)//4])
    high_freq_power = np.sum(emg_fft[len(emg_fft)//4:])
    freq_ratio = low_freq_power / (high_freq_power + 1e-10)
    
    print(f"Idle Detection Criteria:")
    print(f"  Autocorrelation peak: {autocorr_peak:.2f}")
    print(f"  EMG trend: {emg_trend:.4f}")
    print(f"  Frequency ratio (low/high): {freq_ratio:.2f}")
    
    # Idle indicators
    is_likely_idle = (
        autocorr_peak < 1000 and  # Low autocorrelation
        abs(emg_trend) < 0.1 and  # No strong trend
        freq_ratio < 2.0  # Not dominated by low frequencies
    )
    
    return is_likely_idle

def gesture_specific_features(emg_data, quaternion_data):
    """
    Extract features specific to different gesture types
    """
    print("\n=== Gesture-Specific Features ===")
    
    # A gesture: typically involves up-down movement
    # Look for vertical orientation changes
    quat_vertical = quaternion_data[:, 1]  # Y component
    vertical_movement = np.std(quat_vertical)
    
    # B gesture: typically involves horizontal movement
    # Look for horizontal orientation changes
    quat_horizontal = quaternion_data[:, 0]  # X component
    horizontal_movement = np.std(quat_horizontal)
    
    # C gesture: typically involves circular movement
    # Look for rotational changes
    quat_rotation = quaternion_data[:, 3]  # W component
    rotational_movement = np.std(quat_rotation)
    
    print(f"Movement Analysis:")
    print(f"  Vertical movement (A-like): {vertical_movement:.4f}")
    print(f"  Horizontal movement (B-like): {horizontal_movement:.4f}")
    print(f"  Rotational movement (C-like): {rotational_movement:.4f}")
    
    # Determine most likely gesture based on movement patterns
    movements = {
        'A': vertical_movement,
        'B': horizontal_movement, 
        'C': rotational_movement
    }
    
    most_likely = max(movements, key=movements.get)
    print(f"  Most likely gesture: {most_likely}")
    
    return movements

def main():
    print("=== Improved Gesture Recognition Approaches ===")
    print()
    print("The problem with current approach:")
    print("- Relying solely on EMG variance is unreliable")
    print("- EMG can be high during idle due to tension")
    print("- EMG can be low during smooth gestures")
    print()
    print("Better approaches:")
    print("1. Temporal patterns (how signals change over time)")
    print("2. Frequency domain analysis")
    print("3. Cross-correlation between channels")
    print("4. Gesture-specific movement patterns")
    print("5. Multiple criteria for idle detection")
    print()
    print("Key insights:")
    print("- Idle: Random, uncorrelated, no strong trends")
    print("- Gestures: Structured patterns, correlated changes")
    print("- Different gestures have different movement signatures")
    print()

if __name__ == "__main__":
    main() 