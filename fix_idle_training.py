#!/usr/bin/env python3
"""
Fix the IDLE training data issue by retraining with proper static IDLE data
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.train_gui_simple import SimpleMyoGUI

def main():
    print("=== Fix IDLE Training Data Issue ===")
    print()
    print("PROBLEM IDENTIFIED:")
    print("Your current model was trained with IDLE data that had high variance")
    print("(EMG range: -125 to 122), making it indistinguishable from actual gestures.")
    print()
    print("SOLUTION:")
    print("Retrain with proper static IDLE data (low variance)")
    print()
    print("INSTRUCTIONS:")
    print("1. Collect 100 samples for A, B, C (write the letters)")
    print("2. Collect 200 samples for IDLE (keep arm COMPLETELY still)")
    print("3. Collect 100 samples for NOISE (random movements)")
    print("4. Train the model")
    print("5. Test predictions")
    print()
    print("IMPORTANT FOR IDLE COLLECTION:")
    print("- Keep your arm COMPLETELY still")
    print("- Don't tense muscles")
    print("- Don't move fingers or wrist")
    print("- Just relax and hold position")
    print("- Target EMG std < 2.0 for IDLE")
    print()
    
    # Create GUI with reduced samples for faster collection
    app = SimpleMyoGUI(
        labels=['A', 'B', 'C', 'IDLE', 'NOISE'], 
        samples_per_class=100,  # Reduced for faster collection
        duration_ms=2000
    )
    
    print("GUI launched! Follow these steps:")
    print("1. Switch to Training tab")
    print("2. Collect 100 samples each for A, B, C (write the letters)")
    print("3. Collect 200 samples for IDLE (keep arm COMPLETELY still)")
    print("4. Collect 100 samples for NOISE (random movements)")
    print("5. Train the model")
    print("6. Test in Prediction tab")
    print()
    print("Expected improvement:")
    print("- IDLE should have EMG std < 2.0")
    print("- Gestures should have EMG std > 5.0")
    print("- Model should correctly distinguish between static and movement")
    
    app.mainloop()

if __name__ == "__main__":
    main() 