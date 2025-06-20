# Tkinter GUI for live prediction
# src/gui.py
import tkinter as tk
import tkinter.ttk as ttk
import time
import myo
from .preprocessing import preprocess_emg, extract_features, extract_all_features
import numpy as np
import threading
from .utils import get_sdk_path
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import tkinter.scrolledtext as scrolledtext
import os
from src.model import train_model

myo.init(sdk_path=get_sdk_path())

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
        self.accel_buffer = []
        self.gyro_buffer = []
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
            print(f"Error starting Myo hub: {e}")

    def _run_hub(self):
        self.hub.run_forever(self)

    def on_connected(self, event):
        event.device.stream_emg(True)
        self.status.set(f"Myo connected: {event.device_name}")

    def on_disconnected(self, event):
        print("Myo disconnected!")

    def on_emg(self, event):
        if len(event.emg) == 8 and self.collecting:
            self.emg_buffer.append(event.emg)

    def on_imu(self, event):
        now = time.time()
        if now - self.last_imu_time < 0.05:
            return
        self.last_imu_time = now
        print("on_imu called")
        if self.collecting:
            accel = [event.accelerometer.x, event.accelerometer.y, event.accelerometer.z]
            self.accel_buffer.append(accel)
            self.log(f"on_imu: accel={accel}")

    def on_imu_data(self, event):
        now = time.time()
        if now - self.last_imu_time < 0.05:
            return
        self.last_imu_time = now
        print("on_imu_data called")
        if self.collecting:
            accel = [event.accelerometer.x, event.accelerometer.y, event.accelerometer.z]
            self.accel_buffer.append(accel)
            self.log(f"on_imu_data: accel={accel}")

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
        min_len = min(len(self.emg_buffer), len(self.accel_buffer), len(self.gyro_buffer))
        window_size = 100
        if min_len >= window_size:
            try:
                emg_win = np.array(self.emg_buffer[:window_size])
                accel_win = np.array(self.accel_buffer[:window_size])
                gyro_win = np.array(self.gyro_buffer[:window_size])
                emg_proc = preprocess_emg(emg_win)
                features = extract_all_features(emg_proc, accel_win, gyro_win)
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
        self.accel_buffer = []
        self.gyro_buffer = []

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
        self.collected = {label: 0 for label in labels}
        self.data = {label: [] for label in labels}
        self.accel_data = {label: [] for label in labels}
        self.gyro_data = {label: [] for label in labels}
        self.hub = myo.Hub()
        self.collecting = False
        self.emg_buffer = []
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
        # Accel: 3 axes
        self.accel_lines = [self.axs[1].plot([], [], label=axis)[0] for axis in ['X', 'Y', 'Z']]
        self.axs[1].set_ylabel('Accel')
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
        # Accel
        if len(self.accel_buffer) >= window:
            accel_arr = np.array(self.accel_buffer[-window:])
        elif len(self.accel_buffer) > 0:
            accel_arr = np.pad(np.array(self.accel_buffer), ((window-len(self.accel_buffer),0),(0,0)), 'constant')
        else:
            accel_arr = np.zeros((window, 3))
        for i, line in enumerate(self.accel_lines):
            line.set_data(np.arange(len(accel_arr)), accel_arr[:, i] if accel_arr.shape[0] else np.zeros(window))
        self.axs[1].set_xlim(0, window)
        if accel_arr.size:
            self.axs[1].set_ylim(np.min(accel_arr)-1, np.max(accel_arr)+1)
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
                self.hub = myo.Hub()
                self.hub_thread = threading.Thread(target=self._run_hub, daemon=True)
                self.hub_thread.start()
        else:
            self.status.set("Disconnected (manual)")
            self._set_led("gray")
            try:
                if hasattr(self, 'hub') and self.hub:
                    self.hub.stop()
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
            accel = [event.accelerometer.x, event.accelerometer.y, event.accelerometer.z]
            self.accel_buffer.append(accel)
            self.log(f"on_imu: accel={accel}")

    def on_imu_data(self, event):
        now = time.time()
        if now - self.last_imu_time < 0.05:
            return
        self.last_imu_time = now
        print("on_imu_data called")
        if self.collecting:
            accel = [event.accelerometer.x, event.accelerometer.y, event.accelerometer.z]
            self.accel_buffer.append(accel)
            self.log(f"on_imu_data: accel={accel}")

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
            self.gyro_buffer = []
            self.after(self.duration_ms, lambda: self._finish_collection(label))

    def _finish_collection(self, label):
        self.collecting = False
        self.plotting = False
        if len(self.emg_buffer) > 0:
            self.data[label].append(np.array(self.emg_buffer))
            self.accel_data[label].append(np.array(self.accel_buffer))
            self.gyro_data[label].append(np.array(self.gyro_buffer))
            self.collected[label] += 1
            self.status.set(f"Sample {self.collected[label]}/{self.samples_per_class} for {label} collected!")
            # Log summary of accel data
            self.log(f"Accel data for {label} sample {self.collected[label]}: length={len(self.accel_buffer)}")
            if len(self.accel_buffer) > 0:
                self.log(f"First 5 accel samples: {self.accel_buffer[:5]}")
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
            accel_arr = self.accel_data[label][-1] if self.accel_data[label] else np.zeros((len(emg_arr), 3))
            window = min(200, len(emg_arr))
            # EMG
            for i, line in enumerate(self.emg_lines):
                line.set_data(np.arange(window), emg_arr[-window:, i] if emg_arr.shape[0] else np.zeros(window))
            self.axs[0].set_xlim(0, window)
            if emg_arr.size:
                self.axs[0].set_ylim(np.min(emg_arr)-5, np.max(emg_arr)+5)
            else:
                self.axs[0].set_ylim(-1, 1)
            # Accel
            for i, line in enumerate(self.accel_lines):
                line.set_data(np.arange(window), accel_arr[-window:, i] if accel_arr.shape[0] else np.zeros(window))
            self.axs[1].set_xlim(0, window)
            if accel_arr.size:
                self.axs[1].set_ylim(np.min(accel_arr)-1, np.max(accel_arr)+1)
            else:
                self.axs[1].set_ylim(-1, 1)
            self.canvas.draw()

    def log(self, msg):
        print(msg, flush=True)
        self.log_widget.config(state='normal')
        self.log_widget.insert('end', msg + '\n')
        self.log_widget.see('end')
        self.log_widget.config(state='disabled')

    def save_data(self):
        # Save all collected EMG and accel data for all labels to a single .npz file
        save_dir = os.path.join(os.path.expanduser('~'), 'MyoData')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'myo_training_data.npz')
        save_dict = {}
        for label in self.labels:
            emg_arrs = self.data[label]
            accel_arrs = self.accel_data[label]
            if emg_arrs:
                save_dict[f'{label}_emg'] = np.array(emg_arrs, dtype=object)
            if accel_arrs:
                save_dict[f'{label}_accel'] = np.array(accel_arrs, dtype=object)
        np.savez(save_path, **save_dict)
        self.log(f"Saved all data to {save_path}")
        # Start training in a background thread
        threading.Thread(target=self._train_model_from_saved_data, args=(save_path,), daemon=True).start()

    def _train_model_from_saved_data(self, save_path):
        try:
            self.log("Starting model training...")
            data = np.load(save_path, allow_pickle=True)
            all_features, all_labels = [], []
            for label in self.labels:
                emg_arrs = data.get(f'{label}_emg', None)
                accel_arrs = data.get(f'{label}_accel', None)
                if emg_arrs is not None and accel_arrs is not None:
                    for emg, accel in zip(emg_arrs, accel_arrs):
                        min_len = min(len(emg), len(accel))
                        window_size = 100
                        for i in range(0, min_len - window_size, window_size):
                            emg_win = emg[i:i+window_size]
                            accel_win = accel[i:i+window_size]
                            emg_proc = preprocess_emg(emg_win)
                            features = extract_all_features(emg_proc, accel_win, np.zeros_like(accel_win))
                            all_features.append(features)
                            all_labels.append(label)
            if all_features:
                X = np.array(all_features)
                model, le = train_model(X, all_labels, all_labels)
                self.log("Model training complete! Model saved.")
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
        self.collected = {label: 0 for label in labels}
        self.data = {label: [] for label in labels}
        self.accel_data = {label: [] for label in labels}
        self.gyro_data = {label: [] for label in labels}
        self.emg_buffer = []
        self.accel_buffer = []
        self.gyro_buffer = []
        self.last_imu_time = 0
        self.plotting = False
        self.collecting = False
        self.predicting = False
        self._build_ui()
        self.hub = myo.Hub()
        self.hub_thread = threading.Thread(target=self._run_hub, daemon=True)
        self.hub_thread.start()
        # self.toggle_connect()  # Start connected (removed, not defined in UnifiedApp)

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
        # Accel: 3 axes
        self.accel_lines = [self.axs[1].plot([], [], label=axis)[0] for axis in ['X', 'Y', 'Z']]
        self.axs[1].set_ylabel('Accel')
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
        # Accel
        if len(self.accel_buffer) >= window:
            accel_arr = np.array(self.accel_buffer[-window:])
        elif len(self.accel_buffer) > 0:
            accel_arr = np.pad(np.array(self.accel_buffer), ((window-len(self.accel_buffer),0),(0,0)), 'constant')
        else:
            accel_arr = np.zeros((window, 3))
        for i, line in enumerate(self.accel_lines):
            line.set_data(np.arange(len(accel_arr)), accel_arr[:, i] if accel_arr.shape[0] else np.zeros(window))
        self.axs[1].set_xlim(0, window)
        if accel_arr.size:
            self.axs[1].set_ylim(np.min(accel_arr)-1, np.max(accel_arr)+1)
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
                self.hub = myo.Hub()
                self.hub_thread = threading.Thread(target=self._run_hub, daemon=True)
                self.hub_thread.start()
        else:
            self.status.set("Disconnected (manual)")
            self._set_led("gray")
            try:
                if hasattr(self, 'hub') and self.hub:
                    self.hub.stop()
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
            accel = [event.accelerometer.x, event.accelerometer.y, event.accelerometer.z]
            self.accel_buffer.append(accel)
            self.log(f"on_imu: accel={accel}")

    def on_imu_data(self, event):
        now = time.time()
        if now - self.last_imu_time < 0.05:
            return
        self.last_imu_time = now
        print("on_imu_data called")
        if self.collecting:
            accel = [event.accelerometer.x, event.accelerometer.y, event.accelerometer.z]
            self.accel_buffer.append(accel)
            self.log(f"on_imu_data: accel={accel}")

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
            self.gyro_buffer = []
            self.after(self.duration_ms, lambda: self._finish_collection(label))

    def _finish_collection(self, label):
        self.collecting = False
        self.plotting = False
        if len(self.emg_buffer) > 0:
            self.data[label].append(np.array(self.emg_buffer))
            self.accel_data[label].append(np.array(self.accel_buffer))
            self.gyro_data[label].append(np.array(self.gyro_buffer))
            self.collected[label] += 1
            self.status.set(f"Sample {self.collected[label]}/{self.samples_per_class} for {label} collected!")
            # Log summary of accel data
            self.log(f"Accel data for {label} sample {self.collected[label]}: length={len(self.accel_buffer)}")
            if len(self.accel_buffer) > 0:
                self.log(f"First 5 accel samples: {self.accel_buffer[:5]}")
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
            accel_arr = self.accel_data[label][-1] if self.accel_data[label] else np.zeros((len(emg_arr), 3))
            window = min(200, len(emg_arr))
            # EMG
            for i, line in enumerate(self.emg_lines):
                line.set_data(np.arange(window), emg_arr[-window:, i] if emg_arr.shape[0] else np.zeros(window))
            self.axs[0].set_xlim(0, window)
            if emg_arr.size:
                self.axs[0].set_ylim(np.min(emg_arr)-5, np.max(emg_arr)+5)
            else:
                self.axs[0].set_ylim(-1, 1)
            # Accel
            for i, line in enumerate(self.accel_lines):
                line.set_data(np.arange(window), accel_arr[-window:, i] if accel_arr.shape[0] else np.zeros(window))
            self.axs[1].set_xlim(0, window)
            if accel_arr.size:
                self.axs[1].set_ylim(np.min(accel_arr)-1, np.max(accel_arr)+1)
            else:
                self.axs[1].set_ylim(-1, 1)
            self.canvas.draw()

    def log(self, msg):
        print(msg, flush=True)
        self.log_widget.config(state='normal')
        self.log_widget.insert('end', msg + '\n')
        self.log_widget.see('end')
        self.log_widget.config(state='disabled')

    def save_data(self):
        # Save all collected EMG and accel data for all labels to a single .npz file
        save_dir = os.path.join(os.path.expanduser('~'), 'MyoData')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'myo_training_data.npz')
        save_dict = {}
        for label in self.labels:
            emg_arrs = self.data[label]
            accel_arrs = self.accel_data[label]
            if emg_arrs:
                save_dict[f'{label}_emg'] = np.array(emg_arrs, dtype=object)
            if accel_arrs:
                save_dict[f'{label}_accel'] = np.array(accel_arrs, dtype=object)
        np.savez(save_path, **save_dict)
        self.log(f"Saved all data to {save_path}")
        # Start training in a background thread
        threading.Thread(target=self._train_model_from_saved_data, args=(save_path,), daemon=True).start()

    def _train_model_from_saved_data(self, save_path):
        try:
            self.log("Starting model training...")
            data = np.load(save_path, allow_pickle=True)
            all_features, all_labels = [], []
            for label in self.labels:
                emg_arrs = data.get(f'{label}_emg', None)
                accel_arrs = data.get(f'{label}_accel', None)
                if emg_arrs is not None and accel_arrs is not None:
                    for emg, accel in zip(emg_arrs, accel_arrs):
                        min_len = min(len(emg), len(accel))
                        window_size = 100
                        for i in range(0, min_len - window_size, window_size):
                            emg_win = emg[i:i+window_size]
                            accel_win = accel[i:i+window_size]
                            emg_proc = preprocess_emg(emg_win)
                            features = extract_all_features(emg_proc, accel_win, np.zeros_like(accel_win))
                            all_features.append(features)
                            all_labels.append(label)
            if all_features:
                X = np.array(all_features)
                model, le = train_model(X, all_labels, all_labels)
                self.log("Model training complete! Model saved.")
            else:
                self.log("No data to train model. Training skipped.")
        except Exception as e:
            self.log(f"Training error: {e}")

    def run(self):
        self.mainloop()

    def _build_prediction_ui(self):
        tk.Label(self.prediction_frame, text="Prediction Mode", font=("Helvetica", 16), bg="#e6e6e6").pack(pady=20)
        self.pred_label = tk.Label(self.prediction_frame, text="Predicted: ...", font=("Helvetica", 18), bg="#e6e6e6")
        self.pred_label.pack(pady=10)
        self.pred_btn = tk.Button(self.prediction_frame, text="Start Predicting", command=self._toggle_predict, bg="#cccccc")
        self.pred_btn.pack(pady=10)

    def _toggle_predict(self):
        self.predicting = not self.predicting
        if self.predicting:
            self.pred_btn.config(text="Stop Predicting")
            # Start prediction loop (implement as needed)
        else:
            self.pred_btn.config(text="Start Predicting")
            # Stop prediction loop (implement as needed)

if __name__ == "__main__":
    pass
    # app = TrainApp(labels=['A', 'B', 'C'], samples_per_class=10, duration_ms=2000)
    # app.run()