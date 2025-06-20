# scripts/train.py
import sys
import os
# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.myo_interface import init_myo, collect_data
from src.model import train_model
import myo
import tkinter as tk

sdk_path = r"C:\Users\brian\__CODING__\MyoArmband\myo-handwriting-recognition\myo-sdk-win-0.9.0"
myo.init(sdk_path=sdk_path)

def main():
    init_myo()
    labels = ['A', 'B', 'C']
    all_features, all_labels = [], []
    from src.preprocessing import preprocess_emg, extract_all_features
    
    for label in labels:
        for attempt in range(1, 2):
            print(f"\n=== Collection {attempt}/3 for {label} ===")
            emg, accel, gyro, lbl = collect_data(label)
            if len(emg) > 0 and len(accel) > 0 and len(gyro) > 0:
                # Windowing: use the minimum length among the three
                min_len = min(len(emg), len(accel), len(gyro))
                window_size = 100
                for i in range(0, min_len - window_size, window_size):
                    emg_win = emg[i:i+window_size]
                    accel_win = accel[i:i+window_size]
                    gyro_win = gyro[i:i+window_size]
                    features = extract_all_features(emg_win, accel_win, gyro_win)
                    all_features.append(features)
                    all_labels.append(label)
            else:
                print(f"No data collected")
        print(f"=== Data collection complete for {label} ===\n")
    
    if all_features:
        X = np.array(all_features)
        model, le = train_model(X, all_labels, all_labels)
        print("Label mapping:", le.classes_)
    else:
        print("No data to train model. Exiting.")

def on_event(self, event):
    print(f"Event type: {type(event)}")

def on_imu_data(self, event):
    ...

def _run_hub(self):
    self.hub.run_forever(self)

if __name__ == "__main__":
    main()

class TrainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Myo Handwriting Recognition")

        self.connect_var = tk.BooleanVar(value=True)
        self.connect_btn = tk.Checkbutton(self.root, text="Connect", variable=self.connect_var, command=self.toggle_connect, onvalue=True, offvalue=False)
        self.connect_btn.pack(pady=5)
        self.toggle_connect()  # Ensure we start connected and UI is in sync

    def toggle_connect(self):
        # Implementation of toggle_connect method
        pass