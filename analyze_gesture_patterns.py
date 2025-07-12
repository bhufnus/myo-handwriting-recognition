#!/usr/bin/env python3
"""
Analyze gesture patterns like a human would - looking at temporal and spatial trajectories
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def analyze_letter_trajectories(emg_data, quaternion_data, class_name):
    """
    Analyze letter trajectories like a human would
    """
    print(f"\n=== Analyzing {class_name} Trajectories ===")
    
    # Convert quaternions to 3D positions (simplified)
    # In reality, you'd use proper quaternion to rotation matrix conversion
    positions = []
    for i in range(len(quaternion_data)):
        # Extract x, y, z from quaternion (simplified)
        x = quaternion_data[i, 0]
        y = quaternion_data[i, 1] 
        z = quaternion_data[i, 2]
        positions.append([x, y, z])
    
    positions = np.array(positions)
    
    # Analyze temporal patterns
    print(f"Temporal Analysis:")
    print(f"  Total time points: {len(positions)}")
    print(f"  Time span: {len(positions) / 200:.2f} seconds")  # Assuming 200Hz
    
    # Find key points in the trajectory
    # 1. Start point
    start_pos = positions[0]
    print(f"  Start position: {start_pos}")
    
    # 2. End point  
    end_pos = positions[-1]
    print(f"  End position: {end_pos}")
    
    # 3. Midpoint
    mid_idx = len(positions) // 2
    mid_pos = positions[mid_idx]
    print(f"  Mid position: {mid_pos}")
    
    # 4. Direction changes (where trajectory changes direction)
    direction_changes = []
    for i in range(1, len(positions) - 1):
        prev_vec = positions[i] - positions[i-1]
        next_vec = positions[i+1] - positions[i]
        
        # Check if direction changed significantly
        if np.dot(prev_vec, next_vec) < 0:  # Negative dot product means direction change
            direction_changes.append(i)
    
    print(f"  Direction changes: {len(direction_changes)} at points {direction_changes}")
    
    # 5. Analyze letter-specific patterns
    if class_name == 'A':
        analyze_letter_a_pattern(positions, emg_data)
    elif class_name == 'B':
        analyze_letter_b_pattern(positions, emg_data)
    elif class_name == 'C':
        analyze_letter_c_pattern(positions, emg_data)
    
    return positions

def analyze_letter_a_pattern(positions, emg_data):
    """
    Analyze letter A pattern - should have up-down-up movement
    """
    print(f"  Letter A Analysis:")
    
    # A should have: up, down, up pattern
    # Look for vertical movement
    y_values = positions[:, 1]  # Y component
    
    # Find peaks (up movements)
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(y_values, height=np.mean(y_values))
    valleys, _ = find_peaks(-y_values, height=-np.mean(y_values))
    
    print(f"    Vertical peaks: {len(peaks)} at points {peaks}")
    print(f"    Vertical valleys: {len(valleys)} at points {valleys}")
    
    # Check for A pattern: should have 2 peaks (up-down-up)
    if len(peaks) >= 2:
        print(f"    ✅ A pattern detected: {len(peaks)} peaks")
    else:
        print(f"    ❌ Not A-like: only {len(peaks)} peaks")

def analyze_letter_b_pattern(positions, emg_data):
    """
    Analyze letter B pattern - should have vertical line + curves
    """
    print(f"  Letter B Analysis:")
    
    # B should have: vertical line + two curves
    # Look for horizontal movement
    x_values = positions[:, 0]  # X component
    
    # Find horizontal direction changes
    x_changes = np.diff(np.sign(np.diff(x_values)))
    direction_changes = np.where(x_changes != 0)[0]
    
    print(f"    Horizontal direction changes: {len(direction_changes)}")
    
    # B should have multiple horizontal direction changes
    if len(direction_changes) >= 3:
        print(f"    ✅ B pattern detected: {len(direction_changes)} direction changes")
    else:
        print(f"    ❌ Not B-like: only {len(direction_changes)} direction changes")

def analyze_letter_c_pattern(positions, emg_data):
    """
    Analyze letter C pattern - should have curved movement
    """
    print(f"  Letter C Analysis:")
    
    # C should have: curved movement
    # Calculate curvature
    dx = np.gradient(positions[:, 0])
    dy = np.gradient(positions[:, 1])
    d2x = np.gradient(dx)
    d2y = np.gradient(dy)
    
    curvature = np.abs(dx * d2y - dy * d2x) / (dx**2 + dy**2)**1.5
    avg_curvature = np.mean(curvature)
    
    print(f"    Average curvature: {avg_curvature:.4f}")
    
    # C should have high curvature
    if avg_curvature > 0.1:
        print(f"    ✅ C pattern detected: high curvature")
    else:
        print(f"    ❌ Not C-like: low curvature")

def visualize_gesture_patterns():
    """
    Visualize gesture patterns to show what the AI should learn
    """
    print("\n=== Visualizing Gesture Patterns ===")
    
    # Load data
    data_path = os.path.join("data", "fixed_data.npz")
    if not os.path.exists(data_path):
        print(f"❌ No fixed data found at {data_path}")
        return
    
    data = np.load(data_path, allow_pickle=True)
    
    # Count classes
    classes = []
    for key in data.keys():
        if key.endswith('_emg'):
            class_name = key.replace('_emg', '')
            classes.append(class_name)
    
    n_classes = len(classes)
    print(f"Found {n_classes} classes: {classes}")
    
    # Create appropriate subplot layout
    if n_classes <= 3:
        fig, axes = plt.subplots(1, n_classes, figsize=(5*n_classes, 5))
        if n_classes == 1:
            axes = [axes]
    elif n_classes <= 6:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.flatten()
    
    fig.suptitle('Gesture Pattern Analysis - What the AI Should Learn', fontsize=16)
    
    for i, key in enumerate(data.keys()):
        if key.endswith('_emg'):
            class_name = key.replace('_emg', '')
            emg_data = data[key]
            quaternion_key = f"{class_name}_quaternion"
            
            if quaternion_key in data and len(emg_data) > 0:
                quaternion_data = data[quaternion_key]
                
                # Use first sample for visualization
                emg_sample = emg_data[0]
                quat_sample = quaternion_data[0]
                
                # Convert to positions
                positions = []
                for j in range(len(quat_sample)):
                    x = quat_sample[j, 0]
                    y = quat_sample[j, 1]
                    z = quat_sample[j, 2]
                    positions.append([x, y, z])
                
                positions = np.array(positions)
                
                # Plot trajectory
                ax = axes[i]
                
                # 3D trajectory
                ax.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, label='Trajectory')
                ax.scatter(positions[0, 0], positions[0, 1], c='green', s=100, label='Start')
                ax.scatter(positions[-1, 0], positions[-1, 1], c='red', s=100, label='End')
                
                # Mark direction changes
                for j in range(1, len(positions) - 1):
                    prev_vec = positions[j] - positions[j-1]
                    next_vec = positions[j+1] - positions[j]
                    if np.dot(prev_vec, next_vec) < 0:
                        ax.scatter(positions[j, 0], positions[j, 1], c='orange', s=50, alpha=0.7)
                
                ax.set_title(f'Letter {class_name}')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.legend()
                ax.grid(True)
    
    # Hide unused subplots
    for i in range(n_classes, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('gesture_patterns.png', dpi=300, bbox_inches='tight')
    print("✅ Saved gesture pattern visualization to gesture_patterns.png")

def create_improved_features_for_ai():
    """
    Create features that capture the human-like pattern recognition
    """
    print("\n=== Creating AI Features for Pattern Recognition ===")
    
    # These are the features the AI should learn:
    features = {
        "temporal_features": [
            "Start position (x, y, z)",
            "End position (x, y, z)", 
            "Mid position (x, y, z)",
            "Total trajectory length",
            "Time duration",
            "Average speed"
        ],
        "spatial_features": [
            "Direction changes count",
            "Curvature at each point",
            "Vertical movement range",
            "Horizontal movement range",
            "Depth movement range"
        ],
        "letter_specific_features": {
            "A": [
                "Number of vertical peaks",
                "Up-down-up pattern score",
                "Vertical symmetry"
            ],
            "B": [
                "Vertical line detection",
                "Curve count",
                "Horizontal direction changes"
            ],
            "C": [
                "Average curvature",
                "Curve completeness",
                "Start-end proximity"
            ]
        },
        "gesture_phases": [
            "Start phase (first 20%)",
            "Middle phase (20-80%)", 
            "End phase (last 20%)"
        ]
    }
    
    print("Features the AI should learn:")
    for category, feature_list in features.items():
        print(f"\n{category.replace('_', ' ').title()}:")
        if isinstance(feature_list, dict):
            for letter, letter_features in feature_list.items():
                print(f"  {letter}:")
                for feature in letter_features:
                    print(f"    - {feature}")
        else:
            for feature in feature_list:
                print(f"  - {feature}")
    
    return features

def main():
    print("=== Human-Like Gesture Pattern Analysis ===")
    print()
    print("This script analyzes gestures the way humans do:")
    print("1. Looking at temporal patterns (start, middle, end)")
    print("2. Analyzing spatial trajectories (movement paths)")
    print("3. Identifying letter-specific patterns")
    print("4. Detecting direction changes and curvature")
    print()
    
    # Load and analyze data
    data_path = os.path.join("data", "fixed_data.npz")
    if os.path.exists(data_path):
        data = np.load(data_path, allow_pickle=True)
        
        # Analyze each class
        for key in data.keys():
            if key.endswith('_emg'):
                class_name = key.replace('_emg', '')
                emg_data = data[key]
                quaternion_key = f"{class_name}_quaternion"
                
                if quaternion_key in data and len(emg_data) > 0:
                    quaternion_data = data[quaternion_key]
                    
                    # Analyze first sample
                    emg_sample = emg_data[0]
                    quat_sample = quaternion_data[0]
                    
                    analyze_letter_trajectories(emg_sample, quat_sample, class_name)
    
    # Create visualization
    visualize_gesture_patterns()
    
    # Define AI features
    create_improved_features_for_ai()
    
    print("\n=== Key Insights ===")
    print("The AI should learn to recognize:")
    print("✅ Temporal patterns (start → middle → end)")
    print("✅ Spatial trajectories (movement paths)")
    print("✅ Direction changes and curvature")
    print("✅ Letter-specific patterns (A: up-down-up, B: line+curves, C: curve)")
    print("✅ Gesture phases and timing")
    print()
    print("This is much better than just looking at EMG variance!")

if __name__ == "__main__":
    main() 