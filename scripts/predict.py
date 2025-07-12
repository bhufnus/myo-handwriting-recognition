# scripts/predict.py
import sys
import os
import traceback
# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train_gui_simple import SimpleMyoGUI
from src.model import load_trained_model

def main():
    try:
        print("Loading trained model...")
        model, le = load_trained_model()
        print("Launching GUI with prediction mode...")
        
        # Use the SimpleMyoGUI which has built-in prediction functionality
        app = SimpleMyoGUI(labels=['A', 'B', 'C'], samples_per_class=300, duration_ms=2000)
        
        # Set the loaded model and label encoder
        app.model = model
        app.le = le
        
        # Switch to prediction tab automatically
        app.notebook.select(1)  # Select the prediction tab (index 1)
        
        print("GUI launched successfully! You're now in the Prediction tab.")
        print("Click 'Load Model' and then 'Start Prediction' to begin.")
        app.mainloop()
        
    except Exception as e:
        tb = traceback.format_exc()
        print(f"❌ Error starting prediction GUI: {e}")
        print(f"Full traceback:\n{tb}")
        try:
            from tkinter import messagebox
            messagebox.showerror("Error", f"Failed to start prediction GUI:\n{e}\n\n{tb}")
        except Exception:
            pass
        sys.exit(1)

if __name__ == "__main__":
    main()