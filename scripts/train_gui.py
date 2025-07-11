import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.train_gui_simple import SimpleMyoGUI

if __name__ == "__main__":
    print("🚀 Starting Myo Handwriting Recognition GUI")
    print("Using EMG + Quaternion data only")
    print("=" * 50)
    
    try:
        print("🔍 DEBUG: Creating SimpleMyoGUI instance...")
        app = SimpleMyoGUI(labels=['A', 'B', 'C', 'IDLE', 'NOISE'], samples_per_class=300, duration_ms=2000)
        print("✅ GUI created successfully")
        print("🔍 DEBUG: Starting mainloop...")
        app.mainloop()
    except Exception as e:
        print(f"❌ Error starting GUI: {e}")
        import traceback
        traceback.print_exc() 