#!/usr/bin/env python3
"""
Fix idle classification by retraining with realistic idle data
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.train_gui_simple import SimpleMyoGUI

def main():
    print("=== Fix Idle Classification Issue ===")
    print()
    print("PROBLEM IDENTIFIED:")
    print("Your model was trained with 'clean' idle data (low EMG variance)")
    print("But real idle data has higher EMG variance due to muscle tension/noise")
    print()
    print("CURRENT IDLE CHARACTERISTICS:")
    print("- EMG variance: 54-104 (too high for current model)")
    print("- Quaternion variance: ~0.248 (good - minimal movement)")
    print("- Model confusion: High EMG = gesture activity")
    print()
    print("SOLUTION:")
    print("Retrain with realistic idle data that includes natural EMG noise")
    print()
    print("INSTRUCTIONS:")
    print("1. Collect 200 samples for IDLE with natural muscle tension")
    print("2. Collect 100 samples each for A, B, C (write the letters)")
    print("3. Collect 100 samples for NOISE (random movements)")
    print("4. Train the model with the new data")
    print("5. Test predictions")
    print()
    print("IMPORTANT FOR IDLE COLLECTION:")
    print("- Keep your arm in a natural, relaxed position")
    print("- Don't try to suppress all muscle activity")
    print("- Allow natural breathing and slight muscle tension")
    print("- Target EMG variance: 50-150 (realistic range)")
    print("- Target quaternion variance: <0.5 (minimal movement)")
    print()
    
    # Create GUI with realistic sample counts
    app = SimpleMyoGUI(
        labels=['A', 'B', 'C', 'IDLE', 'NOISE'], 
        samples_per_class=200,  # More samples for better training
        duration_ms=2000
    )
    
    print("GUI launched! Follow these steps:")
    print("1. Switch to Training tab")
    print("2. Collect 200 samples for IDLE (natural relaxed position)")
    print("3. Collect 100 samples each for A, B, C (write the letters)")
    print("4. Collect 100 samples for NOISE (random movements)")
    print("5. Train the model")
    print("6. Test in Prediction tab")
    print()
    print("Expected improvement:")
    print("- IDLE should be recognized with >80% accuracy")
    print("- EMG variance 50-150 should be classified as IDLE")
    print("- Only actual gestures should trigger A/B/C predictions")
    print()
    
    app.mainloop()

if __name__ == "__main__":
    main() 