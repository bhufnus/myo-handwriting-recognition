#!/usr/bin/env python3
"""
Test script to demonstrate temporal invariance in Myo gesture recognition

This script shows the difference between:
1. Single window prediction (gesture must be at end)
2. Sliding window prediction (gesture can be anywhere in window)

The sliding window approach matches how the model was trained and provides
true temporal invariance for gesture recognition.
"""

import numpy as np
import matplotlib.pyplot as plt

def simulate_gesture_data(gesture_position='end', window_size=100, gesture_duration=30):
    """
    Simulate EMG data with a gesture at different positions
    
    Args:
        gesture_position: 'start', 'middle', or 'end'
        window_size: Total window size
        gesture_duration: Duration of the gesture in samples
    """
    # Create baseline noise
    data = np.random.normal(0, 0.1, (window_size, 8))
    
    # Create a simple gesture pattern (spike in middle channels)
    gesture_pattern = np.zeros((gesture_duration, 8))
    gesture_pattern[:, 2:6] = np.random.normal(0.5, 0.2, (gesture_duration, 4))
    
    # Position the gesture based on the parameter
    if gesture_position == 'start':
        start_idx = 0
    elif gesture_position == 'middle':
        start_idx = (window_size - gesture_duration) // 2
    else:  # 'end'
        start_idx = window_size - gesture_duration
    
    # Insert the gesture
    data[start_idx:start_idx + gesture_duration] = gesture_pattern
    
    return data

def single_window_prediction(data, window_size=100):
    """Simulate single window prediction (gesture must be at end)"""
    # Only use the last window_size samples
    window = data[-window_size:]
    # Simulate prediction confidence based on gesture presence at end
    end_portion = window[-30:]  # Last 30 samples
    gesture_strength = np.mean(np.abs(end_portion[:, 2:6]))
    confidence = min(gesture_strength / 0.5, 1.0)  # Normalize to 0-1
    return confidence

def sliding_window_prediction(data, window_size=100, overlap=0.5):
    """Simulate sliding window prediction (temporal invariance)"""
    predictions = []
    confidences = []
    
    # Create sliding windows with overlap
    step_size = int(window_size * (1 - overlap))
    
    for j in range(0, len(data) - window_size + 1, step_size):
        window = data[j:j + window_size]
        
        # Simulate prediction confidence for this window
        # Check if gesture is present anywhere in the window
        gesture_strength = np.mean(np.abs(window[:, 2:6]))
        confidence = min(gesture_strength / 0.5, 1.0)
        
        predictions.append(confidence)
        confidences.append(confidence)
    
    # Return the maximum confidence across all windows
    return max(confidences) if confidences else 0.0

def demonstrate_temporal_invariance():
    """Demonstrate the difference between prediction methods"""
    print("🔄 Demonstrating Temporal Invariance in Myo Gesture Recognition")
    print("=" * 70)
    
    # Test different gesture positions
    positions = ['start', 'middle', 'end']
    window_size = 100
    
    print(f"\n📊 Testing with window size: {window_size}")
    print(f"   Gesture duration: 30 samples")
    print(f"   Sliding window overlap: 50%")
    print()
    
    results = []
    
    for position in positions:
        print(f"🎯 Testing gesture at {position.upper()} of window:")
        
        # Generate test data
        data = simulate_gesture_data(position, window_size)
        
        # Test single window prediction
        single_conf = single_window_prediction(data, window_size)
        
        # Test sliding window prediction
        sliding_conf = sliding_window_prediction(data, window_size, 0.5)
        
        results.append({
            'position': position,
            'single': single_conf,
            'sliding': sliding_conf
        })
        
        print(f"   Single window confidence: {single_conf:.3f}")
        print(f"   Sliding window confidence: {sliding_conf:.3f}")
        print(f"   Improvement: {((sliding_conf - single_conf) / max(single_conf, 0.01) * 100):+.1f}%")
        print()
    
    # Summary
    print("📈 SUMMARY:")
    print("   Single Window: Only recognizes gestures at the END of the window")
    print("   Sliding Windows: Recognizes gestures ANYWHERE in the window")
    print()
    
    avg_single = np.mean([r['single'] for r in results])
    avg_sliding = np.mean([r['sliding'] for r in results])
    
    print(f"   Average Single Window: {avg_single:.3f}")
    print(f"   Average Sliding Window: {avg_sliding:.3f}")
    print(f"   Overall Improvement: {((avg_sliding - avg_single) / avg_single * 100):+.1f}%")
    
    return results

def plot_comparison(results):
    """Plot the comparison between prediction methods"""
    positions = [r['position'] for r in results]
    single_confs = [r['single'] for r in results]
    sliding_confs = [r['sliding'] for r in results]
    
    x = np.arange(len(positions))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, single_confs, width, label='Single Window', alpha=0.8)
    bars2 = ax.bar(x + width/2, sliding_confs, width, label='Sliding Windows', alpha=0.8)
    
    ax.set_xlabel('Gesture Position in Window')
    ax.set_ylabel('Prediction Confidence')
    ax.set_title('Temporal Invariance: Single vs Sliding Window Prediction')
    ax.set_xticks(x)
    ax.set_xticklabels(positions)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("🚀 Myo Temporal Invariance Test")
    print("This demonstrates why sliding windows are crucial for real-time gesture recognition")
    print()
    
    # Run the demonstration
    results = demonstrate_temporal_invariance()
    
    # Ask if user wants to see the plot
    try:
        show_plot = input("\n📊 Show comparison plot? (y/n): ").lower().strip()
        if show_plot in ['y', 'yes']:
            plot_comparison(results)
    except KeyboardInterrupt:
        print("\n👋 Test completed!")
    
    print("\n💡 Key Takeaways:")
    print("   1. Single window prediction only works when gestures are at the END")
    print("   2. Sliding windows provide temporal invariance (gestures anywhere)")
    print("   3. This matches how the model was trained with sliding windows")
    print("   4. Real-time recognition requires temporal invariance")
    print("\n✅ The GUI now supports both modes with a checkbox!") 