# scripts/predict.py
import sys
import os
# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.myo_interface import init_myo
from src.model import load_trained_model
from src.gui import App

def main():
    init_myo()
    model, le = load_trained_model()
    print("Launching GUI...")
    app = App(model, le)
    app.run()

if __name__ == "__main__":
    main()