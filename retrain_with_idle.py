#!/usr/bin/env python3
"""
Retrain the model with proper idle state detection and more comprehensive training data
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.train_gui_simple import SimpleMyoGUI

def main():
    print("=== Myo Handwriting Recognition - Retraining with Idle Detection ===")
    print("This will help create a better model that can distinguish between:")
    print("- Actual handwriting gestures (A, B, C)")
    print("- Idle states (no movement)")
    print("- Noise states (random movements)")
    print()
    print("Instructions:")
    print("1. Collect 100 samples each for A, B, C")
    print("2. Collect 200 samples for IDLE (keep arm still)")
    print("3. Collect 200 samples for NOISE (random movements)")
    print("4. Train the improved model")
    print()
    
    # Create GUI with more comprehensive labels
    app = SimpleMyoGUI(
        labels=['A', 'B', 'C', 'IDLE', 'NOISE'], 
        samples_per_class=100,  # Reduced for faster collection
        duration_ms=2000
    )
    
    print("GUI launched! Follow these steps:")
    print("1. Switch to Training tab")
    print("2. Collect 100 samples each for A, B, C (write the letters)")
    print("3. Collect 200 samples for IDLE (keep arm completely still)")
    print("4. Collect 200 samples for NOISE (make random movements)")
    print("5. Train the model")
    print("6. Test in Prediction tab")
    
    app.mainloop()

if __name__ == "__main__":
    main() 