# scripts/train.py
import sys
import os
# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.myo_interface import init_myo, collect_data
from src.model import train_model

def main():
    init_myo()
    labels = ['A', 'B', 'C']
    all_emg, all_labels = [], []
    
    for label in labels:
        for attempt in range(1, 4):
            print(f"\n=== Collection {attempt}/3 for {label} ===")
            emg, lbl = collect_data(label)
            if len(emg) > 0:
                from src.preprocessing import preprocess_emg, extract_features
                emg_proc = preprocess_emg(emg)
                features = [extract_features(emg_proc[i:i+100]) for i in range(0, len(emg_proc)-100, 100)]
                all_emg.extend(features)
                all_labels.extend(lbl[:len(features)])
            else:
                print(f"No data collected")
        print(f"=== Data collection complete for {label} ===\n")
    
    if all_emg:
        X = np.array(all_emg)
        model, le = train_model(X, all_labels, all_labels)
    else:
        print("No data to train model. Exiting.")

if __name__ == "__main__":
    main()