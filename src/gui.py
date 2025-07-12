# Tkinter GUI for live prediction
# src/gui.py
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import myo
import threading
import time
import os
import pickle
import json
from .preprocessing import preprocess_emg, extract_emg_features, extract_all_features
from .model import train_model
import traceback

# Use the same SDK path as the working test_myo_imu.py
sdk_path = r"C:\Users\brian\__CODING__\MyoArmband\myo-handwriting-recognition\myo-sdk-win-0.9.0"
myo.init(sdk_path=sdk_path)

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
        # Data buffers - only EMG and quaternions
        self.emg_buffer = []
        self.quaternion_buffer = []  # Raw quaternions from orientation
        self.last_emg_time = 0
        self.last_quaternion_time = 0
        self.hub = myo.Hub()
        self.last_pred = None
        self.last_print_time = 0
        self.collecting = True
        self.start_time = time.time()
        self.last_imu_time = 0  # For interval check
        print("GUI initialized")
        try:
            print("Starting real-time prediction...")
            self.hub_thread = threading.Thread(target=self._run_hub, daemon=True)
            self.hub_thread.start()
            self.update_progress()
        except Exception as e:
            tb = traceback.format_exc()
            print(f"Error starting Myo hub: {e}")
            print(tb)
            try:
                from tkinter import messagebox
                messagebox.showerror("Error", f"Could not start Myo hub:\n{e}\n\n{tb}")
            except Exception:
                pass
            self.destroy()

    def _run_hub(self):
        self.hub.run_forever(self)

    def on_connected(self, event):
        event.device.stream_emg(True)
        # self.status.set(f"Myo connected: {event.device_name}")  # Removed: not used in App

    def on_disconnected(self, event):
        print("Myo disconnected!")

    def on_emg(self, event):
        if len(event.emg) == 8 and self.collecting:
            self.emg_buffer.append(event.emg)

    def on_orientation(self, event):
        now = time.time()
        if now - self.last_quaternion_time < 0.05:
            return
        self.last_quaternion_time = now
        print("on_orientation called")
        if self.collecting:
            # Extract quaternion data (x, y, z, w)
            quaternion = [event.orientation.x, event.orientation.y, event.orientation.z, event.orientation.w]
            self.quaternion_buffer.append(quaternion)
            # self.log(f"on_orientation: quaternion={quaternion}")  # Removed: not used in App

    def on_imu(self, event):
        # Keep this as a fallback, but quaternions should come through on_orientation
        print("on_imu called - but quaternions should use on_orientation callback")

    def on_event(self, event):
        print(f"Event type: {type(event)}")

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
        min_len = min(len(self.emg_buffer), len(self.quaternion_buffer))
        window_size = 100
        if min_len >= window_size:
            try:
                emg_win = np.array(self.emg_buffer[:window_size])
                quaternion_win = np.array(self.quaternion_buffer[:window_size])
                
                # Debug: print shapes
                print(f"EMG shape: {emg_win.shape}, Quaternion shape: {quaternion_win.shape}")
                
                emg_proc = preprocess_emg(emg_win)
                X_win = np.concatenate([emg_proc, quaternion_win], axis=1)  # shape (window_size, 12)
                
                # Debug: print processed shape
                print(f"Processed input shape: {X_win.shape}")
                print(f"Model expects input shape: {self.model.input_shape}")
                
                # Add batch dimension
                X_win_batch = X_win[np.newaxis, ...]  # shape (1, window_size, 12)
                print(f"Batch input shape: {X_win_batch.shape}")
                
                pred = self.model.predict(X_win_batch, verbose=0)
                text = self.le.inverse_transform([np.argmax(pred)])[0]
                confidence = np.max(pred)
                
                self.label.config(text=f"Predicted Text: {text}")
                self.last_label.config(text=f"Last gesture: {text}")
                print(f"Predicted: {text} (confidence: {confidence:.3f})")
                self.last_pred = text
                self.last_print_time = time.time()
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                error_msg = f"Prediction error: {e}\n{tb}"
                print(error_msg)
                self.label.config(text="Prediction error!")
                try:
                    from tkinter import messagebox
                    messagebox.showerror("Prediction Error", error_msg)
                except Exception:
                    pass
        else:
            self.label.config(text="Not enough data for prediction.")
        self.emg_buffer = []
        self.quaternion_buffer = []

    def run(self):
        try:
            self.mainloop()
        except Exception as e:
            print(f"GUI mainloop error: {e}")

class TrainApp(tk.Tk, myo.DeviceListener):
    def __init__(self, labels=['A', 'B', 'C'], samples_per_class=3, duration_ms=2000):
        tk.Tk.__init__(self)
        myo.DeviceListener.__init__(self)
        self.title("Myo Training Data Collector")
        self.geometry("950x600")
        self.configure(bg="#e6e6e6")  # Light gray background
        self.labels = labels
        self.samples_per_class = samples_per_class
        self.duration_ms = duration_ms
        self.current_label = tk.StringVar(value=labels[0])
        self.status = tk.StringVar(value="Idle")
        self.countdown = tk.StringVar(value="")
        self.progress = tk.IntVar(value=0)
        # Initialize data storage
        self.data = {label: [] for label in labels}
        self.quaternion_data = {label: [] for label in labels}  # Only EMG and quaternion data
        self.collected = {label: 0 for label in labels}
        self.emg_buffer = []
        self.quaternion_buffer = []
        self.accel_buffer = []
        self.gyro_buffer = []
        self.last_imu_time = 0  # For interval check
        self._build_ui()
        self.hub_thread = threading.Thread(target=self._run_hub, daemon=True)
        self.hub_thread.start()
        self.toggle_connect()  # Ensure we start connected and UI is in sync

    def _build_ui(self):
        # Main horizontal frame
        main_frame = tk.Frame(self, bg="#e6e6e6")
        main_frame.pack(fill='both', expand=True)
        # Left: controls
        left_frame = tk.Frame(main_frame, bg="#e6e6e6")
        left_frame.pack(side='left', fill='y', padx=10, pady=10)
        # Right: plot
        self.plot_frame = tk.Frame(main_frame, bg="#e6e6e6")
        self.plot_frame.pack(side='right', fill='both', expand=True, padx=10, pady=10)
        # Controls in left_frame
        tk.Label(left_frame, text="Select Label:", bg="#e6e6e6").pack(pady=5)
        label_menu = tk.OptionMenu(left_frame, self.current_label, *self.labels)
        label_menu.config(bg="#e6e6e6")
        label_menu.pack(pady=5)
        # LED indicator
        self.led_canvas = tk.Canvas(left_frame, width=20, height=20, highlightthickness=0, bg="#e6e6e6")
        self.led = self.led_canvas.create_oval(2, 2, 18, 18, fill="gray")
        self.led_canvas.pack(pady=5)
        # Connect toggle
        self.connect_var = tk.BooleanVar(value=True)
        self.connect_btn = tk.Checkbutton(left_frame, text="Connect", variable=self.connect_var, command=self.toggle_connect, onvalue=True, offvalue=False, bg="#e6e6e6")
        self.connect_btn.pack(pady=5)
        tk.Label(left_frame, textvariable=self.status, font=("Helvetica", 14), bg="#e6e6e6").pack(pady=5)
        tk.Label(left_frame, textvariable=self.countdown, font=("Helvetica", 18), bg="#e6e6e6").pack(pady=5)
        self.progressbar = ttk.Progressbar(left_frame, orient="horizontal", length=300, mode="determinate", maximum=self.samples_per_class)
        self.progressbar.pack(pady=10)
        tk.Button(left_frame, text="Start Recording", command=self.start_collection, bg="#cccccc").pack(pady=10)
        tk.Button(left_frame, text="Save Data", command=self.save_data, bg="#cccccc").pack(pady=5)
        self.sample_labels = {}
        for label in self.labels:
            l = tk.Label(left_frame, text=f"{label}: 0/{self.samples_per_class}", bg="#e6e6e6")
            l.pack()
            self.sample_labels[label] = l
        # Log/console area
        self.log_widget = scrolledtext.ScrolledText(left_frame, height=8, width=45, state='disabled', font=("Consolas", 9), bg="#d9d9d9")
        self.log_widget.pack(pady=5, fill='x')
        # Matplotlib live plot in right frame (hidden by default)
        self.fig, self.axs = plt.subplots(2, 1, figsize=(5, 4), sharex=True)
        self.fig.tight_layout(pad=2.0)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(pady=5, fill='both', expand=True)
        self._init_plot_lines()
        self.plot_frame.pack_forget()  # Hide by default

    def _init_plot_lines(self):
        # EMG: 8 channels
        self.emg_lines = [self.axs[0].plot([], [], label=f'EMG {i+1}')[0] for i in range(8)]
        self.axs[0].set_ylabel('EMG')
        self.axs[0].legend(loc='upper right', fontsize=6)
        # Quaternion: 4 components (x, y, z, w)
        self.quaternion_lines = [self.axs[1].plot([], [], label=comp)[0] for comp in ['X', 'Y', 'Z', 'W']]
        self.axs[1].set_ylabel('Quaternion')
        self.axs[1].legend(loc='upper right', fontsize=6)
        self.axs[1].set_xlabel('Samples')
        self.fig.tight_layout(pad=2.0)

    def _update_plot(self):
        window = 200
        # EMG
        emg_arr = np.array(self.emg_buffer[-window:]) if len(self.emg_buffer) else np.zeros((window, 8))
        for i, line in enumerate(self.emg_lines):
            line.set_data(np.arange(len(emg_arr)), emg_arr[:, i] if emg_arr.shape[0] else np.zeros(window))
        self.axs[0].set_xlim(0, window)
        if emg_arr.size:
            self.axs[0].set_ylim(np.min(emg_arr)-5, np.max(emg_arr)+5)
        else:
            self.axs[0].set_ylim(-1, 1)
        # Quaternion (x, y, z, w)
        if len(self.quaternion_buffer) >= window:
            quaternion_arr = np.array(self.quaternion_buffer[-window:])
        elif len(self.quaternion_buffer) > 0:
            quaternion_arr = np.pad(np.array(self.quaternion_buffer), ((window-len(self.quaternion_buffer),0),(0,0)), 'constant')
        else:
            quaternion_arr = np.zeros((window, 4))
        for i, line in enumerate(self.quaternion_lines):
            line.set_data(np.arange(len(quaternion_arr)), quaternion_arr[:, i] if quaternion_arr.shape[0] else np.zeros(window))
        self.axs[1].set_xlim(0, window)
        if quaternion_arr.size:
            self.axs[1].set_ylim(np.min(quaternion_arr)-1, np.max(quaternion_arr)+1)
        else:
            self.axs[1].set_ylim(-1, 1)
        self.canvas.draw()

    def _set_led(self, color):
        self.led_canvas.itemconfig(self.led, fill=color)

    def _run_hub(self):
        self.hub.run_forever(self)

    def toggle_connect(self):
        if self.connect_var.get():
            self.status.set("Connecting to Myo...")
            self._set_led("gray")
            if not hasattr(self, 'hub_thread') or not self.hub_thread.is_alive():
                self.hub_thread = threading.Thread(target=self._run_hub, daemon=True)
                self.hub_thread.start()
        else:
            self.status.set("Disconnected (manual)")
            self._set_led("gray")
            try:
                # The hub will be cleaned up when the thread ends
                if hasattr(self, 'hub_thread') and self.hub_thread.is_alive():
                    # We can't directly stop the hub from here, but the thread will end when the app closes
                    pass
            except Exception as e:
                print(f"Error stopping hub: {e}")
            self.collecting = False

    def on_connected(self, event):
        event.device.stream_emg(True)
        self.status.set(f"Myo connected: {event.device_name}")
        self._set_led("blue")

    def on_disconnected(self, event):
        self.status.set("Myo disconnected!")
        self._set_led("gray")
        self.connect_var.set(False)

    def on_emg(self, event):
        if self.collecting:
            self.emg_buffer.append(event.emg)

    def on_orientation(self, event):
        now = time.time()
        if now - self.last_imu_time < 0.05:
            return
        self.last_imu_time = now
        print("on_orientation called")
        if self.collecting:
            # Extract orientation data (quaternions)
            orientation = [event.orientation.x, event.orientation.y, event.orientation.z, event.orientation.w]
            self.quaternion_buffer.append(orientation) # Store raw quaternion in quaternion_buffer
            # self.log(f"on_orientation: orientation={orientation}")  # Removed: not used in App

    def on_imu(self, event):
        # Keep this as a fallback, but orientation should come through on_orientation
        print("on_imu called - but orientation should use on_orientation callback")

    def on_imu_data(self, event):
        now = time.time()
        if now - self.last_imu_time < 0.05:
            return
        self.last_imu_time = now
        print("on_imu_data called")
        if self.collecting:
            accel = [event.accelerometer.x, event.accelerometer.y, event.accelerometer.z]
            self.accel_buffer.append(accel)
            # self.log(f"on_imu_data: accel={accel}")  # Removed: accel_data not used

    def start_collection(self):
        label = self.current_label.get()
        if self.collected[label] >= self.samples_per_class:
            self.status.set(f"Already have {self.samples_per_class} samples for {label}")
            return
        self.status.set(f"Get ready to write {label}!")
        self.countdown.set("")
        self.plot_frame.pack(side='right', fill='both', expand=True, padx=10, pady=10)  # Show plot
        self.plotting = True
        self._plot_update_loop()
        self.after(500, lambda: self._countdown(label, 3))

    def _plot_update_loop(self):
        if getattr(self, 'plotting', False):
            self._update_plot()
            self.after(100, self._plot_update_loop)

    def _countdown(self, label, n):
        if n > 0:
            self.countdown.set(f"Starting in {n}...")
            self.after(500, lambda: self._countdown(label, n-1))
        else:
            self.countdown.set("Collecting...")
            self.status.set(f"Collecting {label} sample {self.collected[label]+1}/{self.samples_per_class}")
            self.collecting = True
            self.emg_buffer = []
            self.accel_buffer = []
            self.quaternion_buffer = [] # Reset quaternion buffer
            self.after(self.duration_ms, lambda: self._finish_collection(label))

    def _finish_collection(self, label):
        self.collecting = False
        self.plotting = False
        if len(self.emg_buffer) > 0:
            self.data[label].append(np.array(self.emg_buffer))
            # self.accel_data[label].append(np.array(self.accel_buffer))  # Removed: accel_data not used
            self.quaternion_data[label].append(np.array(self.quaternion_buffer)) # Store quaternion data
            self.collected[label] += 1
            self.status.set(f"Sample {self.collected[label]}/{self.samples_per_class} for {label} collected!")
            # Log summary of accel data
            # self.log(f"Accel data for {label} sample {self.collected[label]}: length={len(self.accel_buffer)}")
            # if len(self.accel_buffer) > 0:
            #     self.log(f"First 5 accel samples: {self.accel_buffer[:5]}")
            # Keep the last sample on the plot
            self._show_last_sample_on_plot(label)
        else:
            self.status.set("No data collected. Try again.")
            self.plot_frame.pack_forget()  # Hide plot if nothing was collected
        self.countdown.set("")
        self.progressbar['value'] = self.collected[label]
        self.sample_labels[label].config(text=f"{label}: {self.collected[label]}/{self.samples_per_class}")
        # self.plot_frame.pack_forget()  # Do not hide plot after collection

    def _show_last_sample_on_plot(self, label):
        # Show the last collected sample for this label
        if self.data[label]:
            emg_arr = self.data[label][-1]
            # accel_arr = self.accel_data[label][-1] if self.accel_data[label] else np.zeros((len(emg_arr), 3))  # Removed
            quaternion_arr = self.quaternion_data[label][-1] if self.quaternion_data[label] else np.zeros((len(emg_arr), 4))
            window = min(200, len(emg_arr))
            # EMG
            for i, line in enumerate(self.emg_lines):
                line.set_data(np.arange(window), emg_arr[-window:, i] if emg_arr.shape[0] else np.zeros(window))
            self.axs[0].set_xlim(0, window)
            if emg_arr.size:
                self.axs[0].set_ylim(np.min(emg_arr)-5, np.max(emg_arr)+5)
            else:
                self.axs[0].set_ylim(-1, 1)
            # Quaternion (x, y, z, w)
            for i, line in enumerate(self.quaternion_lines):
                line.set_data(np.arange(window), quaternion_arr[-window:, i] if quaternion_arr.shape[0] else np.zeros(window))
            self.axs[1].set_ylim(np.min(quaternion_arr)-1, np.max(quaternion_arr)+1)
            self.canvas.draw()

    def log(self, msg):
        print(msg, flush=True)
        self.log_widget.config(state='normal')
        self.log_widget.insert('end', msg + '\n')
        self.log_widget.see('end')
        self.log_widget.config(state='disabled')

    def save_data(self):
        # Save all collected EMG and quaternion data for all labels to a single .npz file
        save_dir = os.path.join(os.path.expanduser('~'), 'MyoData')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'myo_training_data.npz')
        save_dict = {}
        for label in self.labels:
            emg_arrs = self.data[label]
            # accel_arrs = self.accel_data[label]  # Removed
            quaternion_arrs = self.quaternion_data[label] # Load quaternion data
            if emg_arrs:
                save_dict[f'{label}_emg'] = np.array(emg_arrs, dtype=object)
            # if accel_arrs:
            #     save_dict[f'{label}_accel'] = np.array(accel_arrs, dtype=object)
            if quaternion_arrs:
                save_dict[f'{label}_quaternion'] = np.array(quaternion_arrs, dtype=object) # Save quaternion data
        np.savez(save_path, **save_dict)
        self.log(f"Saved all data to {save_path}")
        # Start training in a background thread
        threading.Thread(target=self._train_model_from_saved_data, args=(save_path,), daemon=True).start()

    def _train_model_from_saved_data(self, save_path):
        try:
            self.log("Starting model training...")
            data = np.load(save_path, allow_pickle=True)
            all_windows, all_labels = [], []
            for label in self.labels:
                emg_arrs = data.get(f'{label}_emg', None)
                quaternion_arrs = data.get(f'{label}_quaternion', None)
                if emg_arrs is not None and quaternion_arrs is not None:
                    for emg, quaternion in zip(emg_arrs, quaternion_arrs):
                        min_len = min(len(emg), len(quaternion))
                        window_size = 100
                        for i in range(0, min_len - window_size + 1, window_size):
                            emg_win = emg[i:i+window_size]
                            quaternion_win = quaternion[i:i+window_size]
                            emg_proc = preprocess_emg(emg_win)
                            X_win = np.concatenate([emg_proc, quaternion_win], axis=1)  # (window_size, 12)
                            all_windows.append(X_win)
                            all_labels.append(label)
            if all_windows:
                X = np.array(all_windows)  # (num_samples, window_size, 12)
                model, le = train_model(X, all_labels, all_labels)
                # Save the model and label encoder for prediction
                save_dir = os.path.join(os.path.expanduser('~'), 'MyoData')
                os.makedirs(save_dir, exist_ok=True)
                model_path = os.path.join(save_dir, 'trained_model.h5')
                le_path = os.path.join(save_dir, 'label_encoder.pkl')
                model.save(model_path)
                import pickle
                with open(le_path, 'wb') as f:
                    pickle.dump(le, f)
                self.log("Model training complete! Model and label encoder saved.")
            else:
                self.log("No data to train model. Training skipped.")
        except Exception as e:
            self.log(f"Training error: {e}")

    def run(self):
        self.mainloop()

class UnifiedApp(tk.Tk, myo.DeviceListener):
    def __init__(self, labels=['A', 'B', 'C'], samples_per_class=3, duration_ms=2000):
        tk.Tk.__init__(self)
        myo.DeviceListener.__init__(self)
        self.title("Myo Unified App")
        self.geometry("950x600")
        self.configure(bg="#e6e6e6")
        self.labels = labels
        self.samples_per_class = samples_per_class
        self.duration_ms = duration_ms
        self.mode = tk.StringVar(value="Training")
        # Initialize all variables needed for training UI
        self.current_label = tk.StringVar(value=self.labels[0])
        self.status = tk.StringVar(value="Idle")
        self.countdown = tk.StringVar(value="")
        self.progress = tk.IntVar(value=0)
        # Initialize data storage
        self.data = {label: [] for label in labels}
        self.quaternion_data = {label: [] for label in labels}  # Only EMG and quaternion data
        self.collected = {label: 0 for label in labels}
        self.emg_buffer = []
        self.quaternion_buffer = []
        self.accel_buffer = []
        self.gyro_buffer = []
        self.last_imu_time = 0
        self.plotting = False
        self.collecting = False
        self.predicting = False
        self._build_ui()
        # Don't create hub here - let toggle_connect handle it
        # self.hub = myo.Hub()
        # self.hub_thread = threading.Thread(target=self._run_hub, daemon=True)
        # self.hub_thread.start()

    def _build_ui(self):
        # Mode switch
        mode_frame = tk.Frame(self, bg="#e6e6e6")
        mode_frame.pack(fill='x', pady=5)
        tk.Label(mode_frame, text="Mode:", bg="#e6e6e6").pack(side='left', padx=5)
        tk.Radiobutton(mode_frame, text="Training", variable=self.mode, value="Training", command=self._switch_mode, bg="#e6e6e6").pack(side='left')
        tk.Radiobutton(mode_frame, text="Prediction", variable=self.mode, value="Prediction", command=self._switch_mode, bg="#e6e6e6").pack(side='left')
        # Container for mode frames
        self.mode_container = tk.Frame(self, bg="#e6e6e6")
        self.mode_container.pack(fill='both', expand=True)
        # Training frame
        self.training_frame = tk.Frame(self.mode_container, bg="#e6e6e6")
        self.prediction_frame = tk.Frame(self.mode_container, bg="#e6e6e6")
        self._build_training_ui()
        self._build_prediction_ui()
        self._switch_mode()

    def _switch_mode(self):
        self.training_frame.pack_forget()
        self.prediction_frame.pack_forget()
        if self.mode.get() == "Training":
            self.training_frame.pack(fill='both', expand=True)
        else:
            self.prediction_frame.pack(fill='both', expand=True)

    def _build_training_ui(self):
        # Main horizontal frame (inside training_frame)
        main_frame = tk.Frame(self.training_frame, bg="#e6e6e6")
        main_frame.pack(fill='both', expand=True)
        # Left: controls
        left_frame = tk.Frame(main_frame, bg="#e6e6e6")
        left_frame.pack(side='left', fill='y', padx=10, pady=10)
        # Right: plot
        self.plot_frame = tk.Frame(main_frame, bg="#e6e6e6")
        self.plot_frame.pack(side='right', fill='both', expand=True, padx=10, pady=10)
        # Controls in left_frame
        tk.Label(left_frame, text="Select Label:", bg="#e6e6e6").pack(pady=5)
        label_menu = tk.OptionMenu(left_frame, self.current_label, *self.labels)
        label_menu.config(bg="#e6e6e6")
        label_menu.pack(pady=5)
        # LED indicator
        self.led_canvas = tk.Canvas(left_frame, width=20, height=20, highlightthickness=0, bg="#e6e6e6")
        self.led = self.led_canvas.create_oval(2, 2, 18, 18, fill="gray")
        self.led_canvas.pack(pady=5)
        # Connect toggle
        self.connect_var = tk.BooleanVar(value=True)
        self.connect_btn = tk.Checkbutton(left_frame, text="Connect", variable=self.connect_var, command=self.toggle_connect if hasattr(self, 'toggle_connect') else lambda: None, onvalue=True, offvalue=False, bg="#e6e6e6")
        self.connect_btn.pack(pady=5)
        tk.Label(left_frame, textvariable=self.status, font=("Helvetica", 14), bg="#e6e6e6").pack(pady=5)
        tk.Label(left_frame, textvariable=self.countdown, font=("Helvetica", 18), bg="#e6e6e6").pack(pady=5)
        self.progressbar = ttk.Progressbar(left_frame, orient="horizontal", length=300, mode="determinate", maximum=self.samples_per_class)
        self.progressbar.pack(pady=10)
        tk.Button(left_frame, text="Start Recording", command=self.start_collection, bg="#cccccc").pack(pady=10)
        tk.Button(left_frame, text="Save Data", command=self.save_data, bg="#cccccc").pack(pady=5)
        self.sample_labels = {}
        for label in self.labels:
            l = tk.Label(left_frame, text=f"{label}: 0/{self.samples_per_class}", bg="#e6e6e6")
            l.pack()
            self.sample_labels[label] = l
        # Log/console area
        self.log_widget = scrolledtext.ScrolledText(left_frame, height=8, width=45, state='disabled', font=("Consolas", 9), bg="#d9d9d9")
        self.log_widget.pack(pady=5, fill='x')
        # Matplotlib live plot in right frame (hidden by default)
        self.fig, self.axs = plt.subplots(2, 1, figsize=(5, 4), sharex=True)
        self.fig.tight_layout(pad=2.0)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(pady=5, fill='both', expand=True)
        self._init_plot_lines()
        self.plot_frame.pack_forget()  # Hide by default

    def _init_plot_lines(self):
        # EMG: 8 channels
        self.emg_lines = [self.axs[0].plot([], [], label=f'EMG {i+1}')[0] for i in range(8)]
        self.axs[0].set_ylabel('EMG')
        self.axs[0].legend(loc='upper right', fontsize=6)
        # Quaternion: 4 components (x, y, z, w)
        self.quaternion_lines = [self.axs[1].plot([], [], label=comp)[0] for comp in ['X', 'Y', 'Z', 'W']]
        self.axs[1].set_ylabel('Quaternion')
        self.axs[1].legend(loc='upper right', fontsize=6)
        self.axs[1].set_xlabel('Samples')
        self.fig.tight_layout(pad=2.0)

    def _update_plot(self):
        window = 200
        # EMG
        emg_arr = np.array(self.emg_buffer[-window:]) if len(self.emg_buffer) else np.zeros((window, 8))
        for i, line in enumerate(self.emg_lines):
            line.set_data(np.arange(len(emg_arr)), emg_arr[:, i] if emg_arr.shape[0] else np.zeros(window))
        self.axs[0].set_xlim(0, window)
        if emg_arr.size:
            self.axs[0].set_ylim(np.min(emg_arr)-5, np.max(emg_arr)+5)
        else:
            self.axs[0].set_ylim(-1, 1)
        # Quaternion (x, y, z, w)
        if len(self.quaternion_buffer) >= window:
            quaternion_arr = np.array(self.quaternion_buffer[-window:])
        elif len(self.quaternion_buffer) > 0:
            quaternion_arr = np.pad(np.array(self.quaternion_buffer), ((window-len(self.quaternion_buffer),0),(0,0)), 'constant')
        else:
            quaternion_arr = np.zeros((window, 4))
        for i, line in enumerate(self.quaternion_lines):
            line.set_data(np.arange(len(quaternion_arr)), quaternion_arr[:, i] if quaternion_arr.shape[0] else np.zeros(window))
        self.axs[1].set_xlim(0, window)
        if quaternion_arr.size:
            self.axs[1].set_ylim(np.min(quaternion_arr)-1, np.max(quaternion_arr)+1)
        else:
            self.axs[1].set_ylim(-1, 1)
        self.canvas.draw()

    def _set_led(self, color):
        self.led_canvas.itemconfig(self.led, fill=color)

    def _run_hub(self):
        # Create hub inside the thread like the working test_myo_imu.py
        hub = myo.Hub()
        hub.run_forever(self)

    def toggle_connect(self):
        if self.connect_var.get():
            self.status.set("Connecting to Myo...")
            self._set_led("gray")
            if not hasattr(self, 'hub_thread') or not self.hub_thread.is_alive():
                self.hub_thread = threading.Thread(target=self._run_hub, daemon=True)
                self.hub_thread.start()
        else:
            self.status.set("Disconnected (manual)")
            self._set_led("gray")
            try:
                # The hub will be cleaned up when the thread ends
                if hasattr(self, 'hub_thread') and self.hub_thread.is_alive():
                    # We can't directly stop the hub from here, but the thread will end when the app closes
                    pass
            except Exception as e:
                print(f"Error stopping hub: {e}")
            self.collecting = False

    def on_connected(self, event):
        event.device.stream_emg(True)
        self.status.set(f"Myo connected: {event.device_name}")
        self._set_led("blue")

    def on_disconnected(self, event):
        self.status.set("Myo disconnected!")
        self._set_led("gray")
        self.connect_var.set(False)

    def on_emg(self, event):
        if self.collecting:
            self.emg_buffer.append(event.emg)

    def on_imu(self, event):
        now = time.time()
        if now - self.last_imu_time < 0.05:
            return
        self.last_imu_time = now
        print("on_imu called")
        if self.collecting:
            # Use orientation data (quaternions) instead of raw IMU
            orientation = [event.orientation.x, event.orientation.y, event.orientation.z, event.orientation.w]
            self.quaternion_buffer.append(orientation) # Store raw quaternion in quaternion_buffer
            # self.log(f"on_imu: orientation={orientation}")  # Removed: not used in App

    def on_imu_data(self, event):
        now = time.time()
        if now - self.last_imu_time < 0.05:
            return
        self.last_imu_time = now
        print("on_imu_data called")
        if self.collecting:
            accel = [event.accelerometer.x, event.accelerometer.y, event.accelerometer.z]
            self.accel_buffer.append(accel)
            # self.log(f"on_imu_data: accel={accel}")  # Removed: accel_data not used

    def start_collection(self):
        label = self.current_label.get()
        if self.collected[label] >= self.samples_per_class:
            self.status.set(f"Already have {self.samples_per_class} samples for {label}")
            return
        self.status.set(f"Get ready to write {label}!")
        self.countdown.set("")
        self.plot_frame.pack(side='right', fill='both', expand=True, padx=10, pady=10)  # Show plot
        self.plotting = True
        self._plot_update_loop()
        self.after(500, lambda: self._countdown(label, 3))

    def _plot_update_loop(self):
        if getattr(self, 'plotting', False):
            self._update_plot()
            self.after(100, self._plot_update_loop)

    def _countdown(self, label, n):
        if n > 0:
            self.countdown.set(f"Starting in {n}...")
            self.after(500, lambda: self._countdown(label, n-1))
        else:
            self.countdown.set("Collecting...")
            self.status.set(f"Collecting {label} sample {self.collected[label]+1}/{self.samples_per_class}")
            self.collecting = True
            self.emg_buffer = []
            self.accel_buffer = []
            self.quaternion_buffer = [] # Reset quaternion buffer
            self.after(self.duration_ms, lambda: self._finish_collection(label))

    def _finish_collection(self, label):
        self.collecting = False
        self.plotting = False
        if len(self.emg_buffer) > 0:
            self.data[label].append(np.array(self.emg_buffer))
            # self.accel_data[label].append(np.array(self.accel_buffer))  # Removed: accel_data not used
            self.quaternion_data[label].append(np.array(self.quaternion_buffer)) # Store quaternion data
            self.collected[label] += 1
            self.status.set(f"Sample {self.collected[label]}/{self.samples_per_class} for {label} collected!")
            # Log summary of accel data
            # self.log(f"Accel data for {label} sample {self.collected[label]}: length={len(self.accel_buffer)}")
            # if len(self.accel_buffer) > 0:
            #     self.log(f"First 5 accel samples: {self.accel_buffer[:5]}")
            # Keep the last sample on the plot
            self._show_last_sample_on_plot(label)
        else:
            self.status.set("No data collected. Try again.")
            self.plot_frame.pack_forget()  # Hide plot if nothing was collected
        self.countdown.set("")
        self.progressbar['value'] = self.collected[label]
        self.sample_labels[label].config(text=f"{label}: {self.collected[label]}/{self.samples_per_class}")
        # self.plot_frame.pack_forget()  # Do not hide plot after collection

    def _show_last_sample_on_plot(self, label):
        # Show the last collected sample for this label
        if self.data[label]:
            emg_arr = self.data[label][-1]
            # accel_arr = self.accel_data[label][-1] if self.accel_data[label] else np.zeros((len(emg_arr), 3))  # Removed
            quaternion_arr = self.quaternion_data[label][-1] if self.quaternion_data[label] else np.zeros((len(emg_arr), 4))
            window = min(200, len(emg_arr))
            # EMG
            for i, line in enumerate(self.emg_lines):
                line.set_data(np.arange(window), emg_arr[-window:, i] if emg_arr.shape[0] else np.zeros(window))
            self.axs[0].set_xlim(0, window)
            if emg_arr.size:
                self.axs[0].set_ylim(np.min(emg_arr)-5, np.max(emg_arr)+5)
            else:
                self.axs[0].set_ylim(-1, 1)
            # Quaternion (x, y, z, w)
            for i, line in enumerate(self.quaternion_lines):
                line.set_data(np.arange(window), quaternion_arr[-window:, i] if quaternion_arr.shape[0] else np.zeros(window))
            self.axs[1].set_ylim(np.min(quaternion_arr)-1, np.max(quaternion_arr)+1)
            self.canvas.draw()

    def log(self, msg):
        print(msg, flush=True)
        self.log_widget.config(state='normal')
        self.log_widget.insert('end', msg + '\n')
        self.log_widget.see('end')
        self.log_widget.config(state='disabled')

    def save_data(self):
        # Save all collected EMG and quaternion data for all labels to a single .npz file
        save_dir = os.path.join(os.path.expanduser('~'), 'MyoData')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'myo_training_data.npz')
        save_dict = {}
        for label in self.labels:
            emg_arrs = self.data[label]
            # accel_arrs = self.accel_data[label]  # Removed
            quaternion_arrs = self.quaternion_data[label] # Load quaternion data
            if emg_arrs:
                save_dict[f'{label}_emg'] = np.array(emg_arrs, dtype=object)
            # if accel_arrs:
            #     save_dict[f'{label}_accel'] = np.array(accel_arrs, dtype=object)
            if quaternion_arrs:
                save_dict[f'{label}_quaternion'] = np.array(quaternion_arrs, dtype=object) # Save quaternion data
        np.savez(save_path, **save_dict)
        self.log(f"Saved all data to {save_path}")
        # Start training in a background thread
        threading.Thread(target=self._train_model_from_saved_data, args=(save_path,), daemon=True).start()

    def _train_model_from_saved_data(self, save_path):
        try:
            self.log("Starting model training...")
            data = np.load(save_path, allow_pickle=True)
            all_windows, all_labels = [], []
            for label in self.labels:
                emg_arrs = data.get(f'{label}_emg', None)
                quaternion_arrs = data.get(f'{label}_quaternion', None)
                if emg_arrs is not None and quaternion_arrs is not None:
                    for emg, quaternion in zip(emg_arrs, quaternion_arrs):
                        min_len = min(len(emg), len(quaternion))
                        window_size = 100
                        for i in range(0, min_len - window_size + 1, window_size):
                            emg_win = emg[i:i+window_size]
                            quaternion_win = quaternion[i:i+window_size]
                            emg_proc = preprocess_emg(emg_win)
                            X_win = np.concatenate([emg_proc, quaternion_win], axis=1)  # (window_size, 12)
                            all_windows.append(X_win)
                            all_labels.append(label)
            if all_windows:
                X = np.array(all_windows)  # (num_samples, window_size, 12)
                model, le = train_model(X, all_labels, all_labels)
                # Save the model and label encoder for prediction
                save_dir = os.path.join(os.path.expanduser('~'), 'MyoData')
                os.makedirs(save_dir, exist_ok=True)
                model_path = os.path.join(save_dir, 'trained_model.h5')
                le_path = os.path.join(save_dir, 'label_encoder.pkl')
                model.save(model_path)
                import pickle
                with open(le_path, 'wb') as f:
                    pickle.dump(le, f)
                self.log("Model training complete! Model and label encoder saved.")
            else:
                self.log("No data to train model. Training skipped.")
        except Exception as e:
            self.log(f"Training error: {e}")

    def run(self):
        self.mainloop()

    def _build_prediction_ui(self):
        # Main frame for prediction UI
        main_frame = tk.Frame(self.prediction_frame, bg="#e6e6e6")
        main_frame.pack(fill='both', expand=True)
        
        # Left: controls and status
        left_frame = tk.Frame(main_frame, bg="#e6e6e6")
        left_frame.pack(side='left', fill='y', padx=10, pady=10)
        
        # Right: plot
        self.pred_plot_frame = tk.Frame(main_frame, bg="#e6e6e6")
        self.pred_plot_frame.pack(side='right', fill='both', expand=True, padx=10, pady=10)
        
        # Prediction controls
        tk.Label(left_frame, text="Prediction Mode", font=("Helvetica", 16), bg="#e6e6e6").pack(pady=10)
        
        # Model status
        self.model_status = tk.StringVar(value="No model loaded")
        tk.Label(left_frame, textvariable=self.model_status, font=("Helvetica", 12), bg="#e6e6e6").pack(pady=5)
        
        # Load model button
        tk.Button(left_frame, text="Load Model", command=self._load_model, bg="#cccccc").pack(pady=5)
        
        # Prediction display
        self.pred_label = tk.Label(left_frame, text="Predicted: ...", font=("Helvetica", 18), bg="#e6e6e6")
        self.pred_label.pack(pady=10)
        
        # Confidence display
        self.confidence_label = tk.Label(left_frame, text="Confidence: ...", font=("Helvetica", 12), bg="#e6e6e6")
        self.confidence_label.pack(pady=5)
        
        # Start/Stop prediction button
        self.pred_btn = tk.Button(left_frame, text="Start Predicting", command=self._toggle_predict, bg="#cccccc")
        self.pred_btn.pack(pady=10)
        
        # Prediction log
        self.pred_log_widget = scrolledtext.ScrolledText(left_frame, height=8, width=45, state='disabled', font=("Consolas", 9), bg="#d9d9d9")
        self.pred_log_widget.pack(pady=5, fill='x')
        
        # Prediction plot
        self.pred_fig, self.pred_axs = plt.subplots(2, 1, figsize=(5, 4), sharex=True)
        self.pred_fig.tight_layout(pad=2.0)
        self.pred_canvas = FigureCanvasTkAgg(self.pred_fig, master=self.pred_plot_frame)
        self.pred_canvas.get_tk_widget().pack(pady=5, fill='both', expand=True)
        self._init_prediction_plot_lines()
        
        # Initialize prediction variables
        self.model = None
        self.le = None
        self.prediction_buffer = []
        self.prediction_window_size = 100
        self.prediction_interval = 1000  # ms

    def _init_prediction_plot_lines(self):
        # EMG: 8 channels
        self.pred_emg_lines = [self.pred_axs[0].plot([], [], label=f'EMG {i+1}')[0] for i in range(8)]
        self.pred_axs[0].set_ylabel('EMG')
        self.pred_axs[0].legend(loc='upper right', fontsize=6)
        # Orientation: 3 Euler angles (Roll, Pitch, Yaw)
        self.pred_accel_lines = [self.pred_axs[1].plot([], [], label=angle)[0] for angle in ['Roll', 'Pitch', 'Yaw']]
        self.pred_axs[1].set_ylabel('Orientation (degrees)')
        self.pred_axs[1].legend(loc='upper right', fontsize=6)
        self.pred_axs[1].set_xlabel('Samples')
        self.pred_fig.tight_layout(pad=2.0)

    def _load_model(self):
        try:
            # Try to load the trained model
            model_path = os.path.join(os.path.expanduser('~'), 'MyoData', 'trained_model.h5')
            le_path = os.path.join(os.path.expanduser('~'), 'MyoData', 'label_encoder.pkl')
            
            if os.path.exists(model_path) and os.path.exists(le_path):
                from tensorflow import keras
                import pickle
                
                self.model = keras.models.load_model(model_path)
                with open(le_path, 'rb') as f:
                    self.le = pickle.load(f)
                
                self.model_status.set("Model loaded successfully")
                self.pred_log("Model loaded successfully")
            else:
                self.model_status.set("No trained model found")
                self.pred_log("No trained model found. Please train a model first.")
        except Exception as e:
            self.model_status.set("Error loading model")
            self.pred_log(f"Error loading model: {e}")

    def _toggle_predict(self):
        if not self.model:
            self.pred_log("Please load a model first")
            return
            
        self.predicting = not self.predicting
        if self.predicting:
            self.pred_btn.config(text="Stop Predicting")
            self.pred_log("Starting prediction...")
            self.prediction_buffer = []
            self._start_prediction_loop()
        else:
            self.pred_btn.config(text="Start Predicting")
            self.pred_log("Stopped prediction")

    def _start_prediction_loop(self):
        if self.predicting:
            self._make_prediction()
            self._update_prediction_plot()
            self.after(self.prediction_interval, self._start_prediction_loop)

    def _make_prediction(self):
        if len(self.emg_buffer) >= self.prediction_window_size:
            try:
                # Get the latest window of data
                emg_win = np.array(self.emg_buffer[-self.prediction_window_size:])
                orientation_win = np.array(self.accel_buffer[-self.prediction_window_size:]) if len(self.accel_buffer) >= self.prediction_window_size else np.zeros((self.prediction_window_size, 3))
                quaternion_win = np.array(self.gyro_buffer[-self.prediction_window_size:]) if len(self.gyro_buffer) >= self.prediction_window_size else np.zeros((self.prediction_window_size, 4))
                
                # Preprocess and extract features
                emg_proc = preprocess_emg(emg_win)
                features = np.concatenate([emg_proc, orientation_win, quaternion_win], axis=1)  # shape (window_size, 12)
                
                # Make prediction
                pred = self.model.predict(features[np.newaxis, ...], verbose=0)
                predicted_class = np.argmax(pred)
                confidence = np.max(pred)
                predicted_label = self.le.inverse_transform([predicted_class])[0]
                
                # Update UI
                self.pred_label.config(text=f"Predicted: {predicted_label}")
                self.confidence_label.config(text=f"Confidence: {confidence:.2f}")
                
                if confidence > 0.5:  # Only log high confidence predictions
                    self.pred_log(f"Predicted: {predicted_label} (confidence: {confidence:.2f})")
                    
            except Exception as e:
                self.pred_log(f"Prediction error: {e}")

    def _update_prediction_plot(self):
        window = 200
        # EMG
        emg_arr = np.array(self.emg_buffer[-window:]) if len(self.emg_buffer) else np.zeros((window, 8))
        for i, line in enumerate(self.pred_emg_lines):
            line.set_data(np.arange(len(emg_arr)), emg_arr[:, i] if emg_arr.shape[0] else np.zeros(window))
        self.pred_axs[0].set_xlim(0, window)
        if emg_arr.size:
            self.pred_axs[0].set_ylim(np.min(emg_arr)-5, np.max(emg_arr)+5)
        else:
            self.pred_axs[0].set_ylim(-1, 1)
        
        # Orientation (Euler angles)
        if len(self.accel_buffer) >= window:
            orientation_arr = np.array(self.accel_buffer[-window:])
        elif len(self.accel_buffer) > 0:
            orientation_arr = np.pad(np.array(self.accel_buffer), ((window-len(self.accel_buffer),0),(0,0)), 'constant')
        else:
            orientation_arr = np.zeros((window, 3))
        for i, line in enumerate(self.pred_accel_lines):
            line.set_data(np.arange(len(orientation_arr)), orientation_arr[:, i] if orientation_arr.shape[0] else np.zeros(window))
        self.pred_axs[1].set_xlim(0, window)
        if orientation_arr.size:
            self.pred_axs[1].set_ylim(np.min(orientation_arr)-10, np.max(orientation_arr)+10)
        else:
            self.pred_axs[1].set_ylim(-180, 180)
        self.pred_canvas.draw()

    def pred_log(self, msg):
        print(f"[Prediction] {msg}", flush=True)
        self.pred_log_widget.config(state='normal')
        self.pred_log_widget.insert('end', msg + '\n')
        self.pred_log_widget.see('end')
        self.pred_log_widget.config(state='disabled')

if __name__ == "__main__":
    pass
    # app = TrainApp(labels=['A', 'B', 'C'], samples_per_class=10, duration_ms=2000)
    # app.run()