# Tkinter GUI for live prediction
# src/gui.py
import tkinter as tk
import tkinter.ttk as ttk
import time
import myo
from .preprocessing import preprocess_emg, extract_features
import numpy as np
import threading

class App(tk.Tk, myo.DeviceListener):
    def __init__(self, model, le):
        tk.Tk.__init__(self)
        myo.DeviceListener.__init__(self)
        self.title("Handwriting Recognition")
        self.label = tk.Label(self, text="Waiting for gesture...", font=("Helvetica", 16))
        self.label.pack(pady=10)
        self.progress = ttk.Progressbar(self, orient="horizontal", length=300, mode="determinate", maximum=2000)
        self.progress.pack(pady=10)
        self.last_label = tk.Label(self, text="Last gesture: None", font=("Helvetica", 14))
        self.last_label.pack(pady=5)
        self.model = model
        self.le = le
        self.emg_buffer = []
        self.hub = myo.Hub()
        self.last_pred = None
        self.last_print_time = 0
        self.collecting = True
        self.start_time = time.time()
        print("GUI initialized")
        try:
            print("Starting real-time prediction...")
            self.hub_thread = threading.Thread(target=self._run_hub, daemon=True)
            self.hub_thread.start()
            self.update_progress()
        except Exception as e:
            print(f"Error starting Myo hub: {e}")

    def _run_hub(self):
        try:
            self.hub.run(self.on_event, 1000000)
        except Exception as e:
            print(f"Error in Myo hub thread: {e}")

    def on_connected(self, event):
        print(f"Myo connected: {event.device_name}")
        event.device.stream_emg(True)

    def on_disconnected(self, event):
        print("Myo disconnected!")

    def on_emg(self, event):
        if len(event.emg) == 8 and self.collecting:
            self.emg_buffer.append(event.emg)

    def update_progress(self):
        elapsed = (time.time() - self.start_time) * 1000  # ms
        if elapsed < 2000:
            self.progress['value'] = elapsed
            self.label.config(text="Waiting for gesture...")
            self.after(20, self.update_progress)
        else:
            self.progress['value'] = 2000
            self.collecting = False
            self.make_prediction()
            # Reset for next interval
            self.start_time = time.time()
            self.collecting = True
            self.progress['value'] = 0
            self.after(20, self.update_progress)

    def make_prediction(self):
        if len(self.emg_buffer) >= 50:  # Require some data
            try:
                emg_proc = preprocess_emg(np.array(self.emg_buffer))
                features = extract_features(emg_proc)
                pred = self.model.predict(features.reshape(1, -1, 1), verbose=0)
                text = self.le.inverse_transform([np.argmax(pred)])[0]
                self.label.config(text=f"Predicted Text: {text}")
                self.last_label.config(text=f"Last gesture: {text}")
                print(f"Predicted: {text}")
                self.last_pred = text
                self.last_print_time = time.time()
            except Exception as e:
                self.label.config(text="Prediction error!")
                print(f"Prediction error: {e}")
        else:
            self.label.config(text="Not enough data for prediction.")
        self.emg_buffer = []

    def run(self):
        try:
            self.mainloop()
        except Exception as e:
            print(f"GUI mainloop error: {e}")