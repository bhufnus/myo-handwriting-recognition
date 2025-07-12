#!/usr/bin/env python3
"""
Human-like feature extraction for gesture recognition
Captures the patterns that humans use to recognize letters
"""
import numpy as np
from scipy.signal import find_peaks
from scipy.spatial.distance import euclidean
import warnings
warnings.filterwarnings('ignore')

def extract_temporal_patterns(positions, emg_data):
    """
    Extract temporal patterns like humans do
    """
    features = []
    
    # 1. Start, middle, end positions
    start_pos = positions[0]
    mid_pos = positions[len(positions) // 2]
    end_pos = positions[-1]
    
    features.extend([start_pos[0], start_pos[1], start_pos[2]])  # Start x,y,z
    features.extend([mid_pos[0], mid_pos[1], mid_pos[2]])       # Mid x,y,z
    features.extend([end_pos[0], end_pos[1], end_pos[2]])       # End x,y,z
    
    # 2. Trajectory length and duration
    total_length = 0
    for i in range(1, len(positions)):
        total_length += euclidean(positions[i], positions[i-1])
    
    features.append(total_length)  # Total trajectory length
    features.append(len(positions) / 200)  # Duration in seconds
    
    # 3. Average speed
    avg_speed = total_length / (len(positions) / 200)
    features.append(avg_speed)
    
    # 4. Start-end distance
    start_end_distance = euclidean(start_pos, end_pos)
    features.append(start_end_distance)
    
    return features

def extract_spatial_patterns(positions):
    """
    Extract spatial patterns like humans do
    """
    features = []
    
    # 1. Direction changes
    direction_changes = 0
    for i in range(1, len(positions) - 1):
        prev_vec = positions[i] - positions[i-1]
        next_vec = positions[i+1] - positions[i]
        
        # Check if direction changed significantly
        if np.dot(prev_vec, next_vec) < 0:
            direction_changes += 1
    
    features.append(direction_changes)
    
    # 2. Curvature analysis
    dx = np.gradient(positions[:, 0])
    dy = np.gradient(positions[:, 1])
    d2x = np.gradient(dx)
    d2y = np.gradient(dy)
    
    curvature = np.abs(dx * d2y - dy * d2x) / (dx**2 + dy**2)**1.5
    # Handle division by zero
    curvature = np.nan_to_num(curvature, nan=0.0, posinf=0.0, neginf=0.0)
    
    features.append(np.mean(curvature))  # Average curvature
    features.append(np.max(curvature))   # Max curvature
    features.append(np.std(curvature))   # Curvature variability
    
    # 3. Movement ranges
    x_range = np.ptp(positions[:, 0])
    y_range = np.ptp(positions[:, 1])
    z_range = np.ptp(positions[:, 2])
    
    features.extend([x_range, y_range, z_range])
    
    # 4. Movement symmetry
    # Compare first half vs second half
    mid_idx = len(positions) // 2
    first_half = positions[:mid_idx]
    second_half = positions[mid_idx:]
    
    if len(first_half) > 0 and len(second_half) > 0:
        first_centroid = np.mean(first_half, axis=0)
        second_centroid = np.mean(second_half, axis=0)
        symmetry_score = euclidean(first_centroid, second_centroid)
        features.append(symmetry_score)
    else:
        features.append(0.0)
    
    return features

def extract_letter_specific_patterns(positions, class_name):
    """
    Extract letter-specific patterns like humans do
    """
    features = []
    
    if class_name == 'A':
        # A pattern: up-down-up
        y_values = positions[:, 1]
        peaks, _ = find_peaks(y_values, height=np.mean(y_values))
        valleys, _ = find_peaks(-y_values, height=-np.mean(y_values))
        
        features.append(len(peaks))      # Number of vertical peaks
        features.append(len(valleys))    # Number of vertical valleys
        
        # Up-down-up pattern score
        if len(peaks) >= 2:
            pattern_score = 1.0
        else:
            pattern_score = 0.0
        features.append(pattern_score)
        
        # Vertical symmetry
        y_symmetry = np.std(y_values)
        features.append(y_symmetry)
        
    elif class_name == 'B':
        # B pattern: vertical line + curves
        x_values = positions[:, 0]
        y_values = positions[:, 1]
        
        # Horizontal direction changes
        x_changes = np.diff(np.sign(np.diff(x_values)))
        direction_changes = np.sum(np.abs(x_changes))
        features.append(direction_changes)
        
        # Vertical line detection (should have consistent x position)
        x_std = np.std(x_values)
        features.append(x_std)
        
        # Curve count (approximate)
        curve_score = len(find_peaks(y_values)) + len(find_peaks(-y_values))
        features.append(curve_score)
        
    elif class_name == 'C':
        # C pattern: curved movement
        # Calculate curvature
        dx = np.gradient(positions[:, 0])
        dy = np.gradient(positions[:, 1])
        d2x = np.gradient(dx)
        d2y = np.gradient(dy)
        
        curvature = np.abs(dx * d2y - dy * d2x) / (dx**2 + dy**2)**1.5
        curvature = np.nan_to_num(curvature, nan=0.0, posinf=0.0, neginf=0.0)
        
        features.append(np.mean(curvature))  # Average curvature
        
        # Curve completeness (start-end proximity)
        start_end_dist = euclidean(positions[0], positions[-1])
        features.append(start_end_dist)
        
        # Curve smoothness
        smoothness = np.std(curvature)
        features.append(smoothness)
    
    else:
        # Generic features for other classes
        features.extend([0.0, 0.0, 0.0, 0.0])
    
    return features

def extract_gesture_phases(positions, emg_data):
    """
    Extract features from different phases of the gesture
    """
    features = []
    
    # Split into phases
    n_points = len(positions)
    start_phase = positions[:n_points//5]      # First 20%
    middle_phase = positions[n_points//5:4*n_points//5]  # Middle 60%
    end_phase = positions[4*n_points//5:]      # Last 20%
    
    # Start phase features
    if len(start_phase) > 0:
        start_speed = np.mean(np.linalg.norm(np.diff(start_phase, axis=0), axis=1))
        start_emg = np.mean(emg_data[:n_points//5]) if len(emg_data) >= n_points//5 else 0
    else:
        start_speed = 0.0
        start_emg = 0.0
    
    # Middle phase features
    if len(middle_phase) > 0:
        middle_speed = np.mean(np.linalg.norm(np.diff(middle_phase, axis=0), axis=1))
        middle_emg = np.mean(emg_data[n_points//5:4*n_points//5]) if len(emg_data) >= 4*n_points//5 else 0
    else:
        middle_speed = 0.0
        middle_emg = 0.0
    
    # End phase features
    if len(end_phase) > 0:
        end_speed = np.mean(np.linalg.norm(np.diff(end_phase, axis=0), axis=1))
        end_emg = np.mean(emg_data[4*n_points//5:]) if len(emg_data) >= 4*n_points//5 else 0
    else:
        end_speed = 0.0
        end_emg = 0.0
    
    features.extend([start_speed, middle_speed, end_speed])
    features.extend([start_emg, middle_emg, end_emg])
    
    return features

def extract_all_human_like_features(emg_data, quaternion_data, class_name):
    """
    Extract all human-like features for gesture recognition
    """
    # Convert quaternions to positions (simplified)
    positions = []
    for i in range(len(quaternion_data)):
        x = quaternion_data[i, 0]
        y = quaternion_data[i, 1]
        z = quaternion_data[i, 2]
        positions.append([x, y, z])
    
    positions = np.array(positions)
    
    # Extract all feature types
    temporal_features = extract_temporal_patterns(positions, emg_data)
    spatial_features = extract_spatial_patterns(positions)
    letter_features = extract_letter_specific_patterns(positions, class_name)
    phase_features = extract_gesture_phases(positions, emg_data)
    
    # Combine all features
    all_features = []
    all_features.extend(temporal_features)
    all_features.extend(spatial_features)
    all_features.extend(letter_features)
    all_features.extend(phase_features)
    
    return np.array(all_features, dtype=np.float32)

def analyze_human_like_features(emg_data, quaternion_data, class_name):
    """
    Analyze and print human-like features for debugging
    """
    print(f"\n=== Human-Like Feature Analysis for {class_name} ===")
    
    features = extract_all_human_like_features(emg_data, quaternion_data, class_name)
    
    print(f"Total features: {len(features)}")
    print(f"Feature range: [{np.min(features):.4f}, {np.max(features):.4f}]")
    print(f"Feature mean: {np.mean(features):.4f}")
    print(f"Feature std: {np.std(features):.4f}")
    
    # Check for invalid values
    nan_count = np.sum(np.isnan(features))
    inf_count = np.sum(np.isinf(features))
    print(f"NaN values: {nan_count}")
    print(f"Inf values: {inf_count}")
    
    if nan_count > 0 or inf_count > 0:
        print("⚠️ Warning: Features contain NaN or Inf values!")
    
    return features

def main():
    print("=== Human-Like Feature Extraction ===")
    print()
    print("This module extracts features the way humans recognize gestures:")
    print("1. Temporal patterns (start, middle, end)")
    print("2. Spatial trajectories (movement paths)")
    print("3. Letter-specific patterns (A: up-down-up, B: line+curves, C: curve)")
    print("4. Gesture phases and timing")
    print()
    print("This is much better than just looking at EMG variance!")

if __name__ == "__main__":
    main() 