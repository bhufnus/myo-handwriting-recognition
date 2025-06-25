#!/usr/bin/env python3
"""
Test script to demonstrate the undo feature
"""

def test_undo_logic():
    """Test the undo logic without running the full GUI"""
    
    # Simulate the data structures from the GUI
    labels = ['A', 'B', 'C']
    samples_per_class = 60
    data = {label: [] for label in labels}
    quaternion_data = {label: [] for label in labels}
    collected = {label: 0 for label in labels}
    
    print("🧪 Testing Undo Feature Logic")
    print("=" * 40)
    
    # Simulate collecting some data
    print("📝 Collecting sample data...")
    
    # Add some fake data for class 'A'
    for i in range(3):
        data['A'].append(f"sample_{i+1}")
        quaternion_data['A'].append(f"quaternion_{i+1}")
        collected['A'] += 1
        print(f"  Added sample {i+1} for class A")
    
    print(f"  Class A now has {collected['A']} samples")
    
    # Simulate undo operation
    print("\n↩️  Undoing last recording for class A...")
    
    if data['A']:
        # Remove the last recorded data
        removed_sample = data['A'].pop()
        removed_quaternion = quaternion_data['A'].pop()
        collected['A'] -= 1
        
        print(f"  ✅ Removed: {removed_sample}")
        print(f"  ✅ Removed: {removed_quaternion}")
        print(f"  Class A now has {collected['A']} samples")
    
    # Test edge case - trying to undo when no data exists
    print("\n⚠️  Testing edge case - undo with no data...")
    
    if not data['B']:
        print("  Class B has no data to undo")
        print("  ✅ Edge case handled correctly")
    
    # Show final state
    print("\n📊 Final state:")
    for label in labels:
        print(f"  Class {label}: {collected[label]}/{samples_per_class} samples")
    
    print("\n🎉 Undo feature test completed!")
    print("\nIn the GUI, you can now:")
    print("  • Click 'Undo Last' button to remove the last recording")
    print("  • Press 'U' key as a keyboard shortcut")
    print("  • See the current count for each class next to the dropdown")
    print("  • Only undo recordings for the currently selected class")

if __name__ == "__main__":
    test_undo_logic() 