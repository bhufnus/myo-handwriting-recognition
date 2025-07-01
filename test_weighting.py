#!/usr/bin/env python3
"""
Test script to demonstrate EMG vs Quaternion weighting
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import create_position_focused_sequence, create_weighted_sequence

def test_weighting():
    """Test different weighting approaches"""
    
    # Create sample data
    window_size = 100
    emg_data = np.random.randn(window_size, 8) * 10  # EMG typically has higher variance
    quaternion_data = np.random.randn(window_size, 4) * 0.1  # Quaternions are typically smaller values
    
    print("🔬 EMG vs Quaternion Weighting Test")
    print("=" * 50)
    
    # Test 1: Equal weighting (original approach)
    print("\n1️⃣ Equal Weighting (Original):")
    equal_weighted = np.concatenate([emg_data, quaternion_data], axis=1)
    print(f"   EMG range: {np.min(equal_weighted[:, :8]):.3f} to {np.max(equal_weighted[:, :8]):.3f}")
    print(f"   Quaternion range: {np.min(equal_weighted[:, 8:]):.3f} to {np.max(equal_weighted[:, 8:]):.3f}")
    print(f"   EMG std: {np.std(equal_weighted[:, :8]):.3f}")
    print(f"   Quaternion std: {np.std(equal_weighted[:, 8:]):.3f}")
    
    # Test 2: Position-focused weighting (80% position, 20% EMG)
    print("\n2️⃣ Position-Focused Weighting (80% position, 20% EMG):")
    position_focused = create_position_focused_sequence(emg_data, quaternion_data, position_emphasis=0.8)
    print(f"   EMG range: {np.min(position_focused[:, :8]):.3f} to {np.max(position_focused[:, :8]):.3f}")
    print(f"   Quaternion range: {np.min(position_focused[:, 8:]):.3f} to {np.max(position_focused[:, 8:]):.3f}")
    print(f"   EMG std: {np.std(position_focused[:, :8]):.3f}")
    print(f"   Quaternion std: {np.std(position_focused[:, 8:]):.3f}")
    
    # Test 3: Custom weighting (30% EMG, 70% position)
    print("\n3️⃣ Custom Weighting (30% EMG, 70% position):")
    custom_weighted = create_weighted_sequence(emg_data, quaternion_data, emg_weight=0.3, quaternion_weight=0.7)
    print(f"   EMG range: {np.min(custom_weighted[:, :8]):.3f} to {np.max(custom_weighted[:, :8]):.3f}")
    print(f"   Quaternion range: {np.min(custom_weighted[:, 8:]):.3f} to {np.max(custom_weighted[:, 8:]):.3f}")
    print(f"   EMG std: {np.std(custom_weighted[:, :8]):.3f}")
    print(f"   Quaternion std: {np.std(custom_weighted[:, 8:]):.3f}")
    
    # Test 4: Position-only (100% position)
    print("\n4️⃣ Position-Only (100% position):")
    position_only = create_position_focused_sequence(emg_data, quaternion_data, position_emphasis=1.0)
    print(f"   EMG range: {np.min(position_only[:, :8]):.3f} to {np.max(position_only[:, :8]):.3f}")
    print(f"   Quaternion range: {np.min(position_only[:, 8:]):.3f} to {np.max(position_only[:, 8:]):.3f}")
    print(f"   EMG std: {np.std(position_only[:, :8]):.3f}")
    print(f"   Quaternion std: {np.std(position_only[:, 8:]):.3f}")
    
    print("\n📊 Summary:")
    print("   • Equal weighting: EMG dominates due to higher variance")
    print("   • Position-focused: Balances the influence, giving more weight to position")
    print("   • Position-only: Completely ignores EMG data")
    print("\n💡 Recommendation: Use position emphasis of 0.8 for handwriting recognition")
    print("   This gives 80% weight to position data and 20% to EMG data")

if __name__ == "__main__":
    test_weighting() 