# Tkinter GUI for live prediction
# src/gui.py
import tkinter as tk
import time
from .preprocessing import preprocess_emg, extract_features

class App(tk.Tk, myo.DeviceListener):
    def __init__(self, model, le):
        tk.Tk.__init__(self)
        myo.DeviceListener.__init__(self)
        self.title("Handwriting Recognition")
        self.label = tk.Label(self, text="Predicted Text: ", font=("Helvetica", 16))
        self.label.pack()
        self.model = model
        self.le = le
        self.emg_buffer = []
        self.hub = myo.Hub()
        self.last_pred = None
        self.last_print_time = 0
        print("GUI initialized")
        try:
            print("Starting real-time prediction...")
            self.hub.run(self.on_event, 1000000)
        except Exception as e:
            print(f"Error starting Myo hub: {e}")

    def on_connected(self, event):
        print(f"Myo connected: {event.device_name}")
        event.device.stream_emg(True)

    def on_disconnected(self, event):
        print("Myo disconnected!")

    def on_emg(self, event):
        if len(event.emg) == 8:
            self.emg_buffer.append(event.emg)
            if len(self.emg_buffer) >= 100:
                try:
                    emg_proc = preprocess_emg(np.array(self.emg_buffer))
                    features = extract_features(emg_proc)
                    pred = self.model.predict(features.reshape(1, -1, 1), verbose=0)
                    text = self.le.inverse_transform([np.argmax(pred)])[0]
                    self.label.config(text=f"Predicted Text: {text}")
                    current_time = time.time()
                    if text != self.last_pred or current_time - self.last_print_time >= 10:
                        print(f"Predicted: {text}")
                        self.last_pred = text
                        self.last_print_time = current_time
                    self.emg_buffer = []
                except Exception as e:
                    print(f"Prediction error: {e}")
                    self.emg_buffer = []

    def run(self):
        try:
            self.mainloop()
        except Exception as e:
            print(f"GUI mainloop error: {e}")