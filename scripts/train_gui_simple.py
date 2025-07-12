#!/usr/bin/env python3
"""
Simplified Myo Handwriting Recognition GUI
Uses only EMG and quaternion data for training and prediction
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import myo
import threading
import time
import pickle
import json
import winsound
from src.preprocessing import preprocess_emg, create_position_focused_sequence
from src.model import train_model
from src.utils import get_sdk_path
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Try to import sounddevice for square wave beeps
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    print("⚠️ sounddevice not available, using winsound for beeps")

# Global audio stream for beeps
audio_stream = None

def play_sine_wave_beep(frequency, duration_ms, volume=0.3):
    """Play a bell-like sine wave beep with proper volume control."""
    print(f"🔊 DEBUG: play_sine_wave_beep called - SOUNDDEVICE_AVAILABLE={SOUNDDEVICE_AVAILABLE}")
    if SOUNDDEVICE_AVAILABLE:
        try:
            sample_rate = 48000
            duration_sec = duration_ms / 1000.0
            t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), False)
            wave = np.sin(2 * np.pi * frequency * t)
            # Envelope: 10ms attack, 70% sustain, 20% release
            total_samples = len(wave)
            attack_samples = int(sample_rate * 0.01)  # 10ms
            sustain_samples = int(total_samples * 0.7)
            release_samples = total_samples - attack_samples - sustain_samples
            envelope = np.ones(total_samples)
            # Attack
            if attack_samples > 0:
                envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
            # Sustain (already 1)
            # Release
            if release_samples > 0:
                envelope[-release_samples:] = np.linspace(1, 0, release_samples)
            # Apply envelope and use the passed volume parameter
            audio_data = (wave * envelope * volume).astype(np.float32)
            print(f"🔊 DEBUG: About to play sounddevice audio - shape={audio_data.shape}, max={np.max(audio_data):.3f}")
            sd.play(audio_data, samplerate=sample_rate)
            sd.wait()
            print("✅ DEBUG: sounddevice play completed")
        except Exception as e:
            print(f"❌ DEBUG: Error playing bell beep: {e}")
    else:
        try:
            print(f"🔊 DEBUG: Using winsound.Beep({int(frequency)}, {int(duration_ms)})")
            winsound.Beep(int(frequency), int(duration_ms))
            print("✅ DEBUG: winsound.Beep completed")
        except Exception as e:
            print(f"❌ DEBUG: Error with winsound.Beep: {e}")

# Initialize Myo SDK
sdk_path = get_sdk_path()
myo.init(sdk_path=sdk_path)

class SimpleMyoGUI(tk.Tk, myo.DeviceListener):
    def __init__(self, labels=['A', 'B', 'C'], samples_per_class=300, duration_ms=2000):
        tk.Tk.__init__(self)
        myo.DeviceListener.__init__(self)
        
        self.title("Myo Handwriting Recognition - EMG + Quaternion")
        self.geometry("1200x800")
        
        # Add simple border around window
        self.configure(relief='raised', bd=3)
        
        # Parameters
        self.labels = labels
        self.samples_per_class = samples_per_class
        self.duration_ms = duration_ms
        
        # Data storage
        self.data = {label: [] for label in labels}
        self.quaternion_data = {label: [] for label in labels}
        self.collected = {label: 0 for label in labels}
        
        # Variation tracking
        self.variation_types = ['Normal', 'Fast', 'Slow', 'High', 'Low', 'Relaxed', 'Focused']
        self.current_variation = 0
        self.variation_notes = {label: [] for label in labels}
        
        # Buffers
        self.emg_buffer = []  # For live display
        self.quaternion_buffer = []  # For live display
        self.collection_emg_buffer = []  # For collection
        self.collection_quaternion_buffer = []  # For collection
        self.collecting = False
        self.last_emg_time = 0
        self.last_quaternion_time = 0
        
        # Myo connection
        self.hub = myo.Hub()
        self.connected = False
        
        # Prediction variables
        self.predicting = False
        self.model = None
        self.le = None
        self.pred_emg_buffer = []
        self.pred_quaternion_buffer = []
        self.prediction_window_size = 400  # 2 seconds at 200Hz
        self.last_prediction_time = 0
        
        # Prediction configuration
        self.use_sliding_windows = True  # Enable temporal invariance
        self.sliding_window_size = 100  # Match training window size
        self.sliding_window_overlap = 0.5  # 50% overlap (matching training)
        
        # UI setup
        self._setup_ui()
        
        # Start Myo connection
        self._start_myo()
        
        # Start plot update timer
        self._schedule_plot_update()
        
        # Auto-load default data file if it exists
        self._auto_load_default_data()
        
        # Initialize class count display
        self._update_class_count(None)
        
        # Bind keyboard shortcuts
        self.bind('<Key>', self._on_key_press)
        
    def _setup_ui(self):
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Training tab
        self.train_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.train_frame, text="Training")
        
        # Prediction tab
        self.pred_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.pred_frame, text="Prediction")
        
        # Settings tab
        self.settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.settings_frame, text="Settings")
        self._setup_settings_ui()
        
        # Data Visualizer tab
        self.visualizer_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.visualizer_frame, text="Data Visualizer")
        self._setup_visualizer_ui()
        
        # Setup training UI
        self._setup_training_ui()
        
        # Setup prediction UI
        self._setup_prediction_ui()
        
    def _setup_settings_ui(self):
        # Create notebook for settings sections
        settings_notebook = ttk.Notebook(self.settings_frame)
        settings_notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # General tab (placeholder)
        general_frame = ttk.Frame(settings_notebook)
        settings_notebook.add(general_frame, text="General")
        ttk.Label(general_frame, text="General settings will go here.").pack(pady=20)
        
        # Audio tab
        audio_frame = ttk.Frame(settings_notebook)
        settings_notebook.add(audio_frame, text="Audio")
        ttk.Label(audio_frame, text="Volume:").pack(anchor='w', padx=10, pady=(20, 5))
        self.volume_var = tk.DoubleVar(value=0.3)
        def on_volume_change(event=None):
            self.volume = self.volume_var.get()
        self.volume_slider = ttk.Scale(audio_frame, from_=0, to=1, orient='horizontal', variable=self.volume_var, command=on_volume_change, length=200)
        self.volume_slider.pack(anchor='w', padx=10)
        self.volume = self.volume_var.get()
        
    def _setup_training_ui(self):
        # Main vertical frame for training tab
        main_frame = ttk.Frame(self.train_frame)
        main_frame.pack(fill='both', expand=True)
        
        # Top: controls
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill='x', padx=10, pady=5)
        
        # Bottom: plot
        self.plot_frame = ttk.Frame(main_frame)
        self.plot_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Control frame
        control_frame = ttk.Frame(top_frame)
        control_frame.pack(fill='x', padx=5, pady=5)
        
        # Model selection dropdown
        ttk.Label(control_frame, text="Model:").pack(side='left', padx=(0, 5))
        self.dl_model_var = tk.StringVar(value='LSTM')
        dl_model_options = ['LSTM', 'GRU', 'SimpleRNN', 'Conv1D', 'Dense']
        dl_model_dropdown = ttk.Combobox(control_frame, textvariable=self.dl_model_var, values=dl_model_options, state='readonly', width=10)
        dl_model_dropdown.pack(side='left', padx=(0, 10))
        
        # Status
        self.status_var = tk.StringVar(value="Status: Disconnected")
        status_label = ttk.Label(control_frame, textvariable=self.status_var, font=('Arial', 12), width=25)
        status_label.pack(side='left', padx=5)
        
        # Collection controls frame
        collection_frame = ttk.Frame(top_frame)
        collection_frame.pack(fill='x', padx=5, pady=5)
        
        # Class selection dropdown
        ttk.Label(collection_frame, text="Class:").pack(side='left', padx=(0, 5))
        self.class_var = tk.StringVar(value=self.labels[0])
        self.class_dropdown = ttk.Combobox(collection_frame, textvariable=self.class_var, 
                                         values=self.labels, state='readonly', width=10)
        self.class_dropdown.pack(side='left', padx=(0, 10))
        
        # Variation selection dropdown
        ttk.Label(collection_frame, text="Variation:").pack(side='left', padx=(0, 5))
        self.variation_var = tk.StringVar(value=self.variation_types[0])
        self.variation_dropdown = ttk.Combobox(collection_frame, textvariable=self.variation_var,
                                             values=self.variation_types, state='readonly', width=10)
        self.variation_dropdown.pack(side='left', padx=(0, 10))
        
        # Class count label
        self.class_count_var = tk.StringVar(value="(0/300)")
        self.class_count_label = ttk.Label(collection_frame, textvariable=self.class_count_var, 
                                         font=('Arial', 10, 'bold'))
        self.class_count_label.pack(side='left', padx=(0, 10))
        
        # Update class count when class changes
        self.class_dropdown.bind('<<ComboboxSelected>>', self._update_class_count)
        
        # Record button
        self.record_btn = ttk.Button(collection_frame, text="Record", 
                                   command=self._start_recording)
        self.record_btn.pack(side='left', padx=5)
        
        # Undo last recording button
        self.undo_btn = ttk.Button(collection_frame, text="Undo Last", 
                                  command=self._undo_last_recording)
        self.undo_btn.pack(side='left', padx=5)
        
        # Recording indicator and countdown
        self.recording_frame = ttk.Frame(collection_frame)
        self.recording_frame.pack(side='right', padx=20)
        
        self.recording_indicator = tk.Label(self.recording_frame, text="⏹", font=('Arial', 16), fg='red')
        self.recording_indicator.pack(side='left', padx=5)
        
        self.countdown_var = tk.StringVar(value="")
        self.countdown_label = tk.Label(self.recording_frame, textvariable=self.countdown_var, font=('Arial', 14, 'bold'))
        self.countdown_label.pack(side='left', padx=5)
        
        # Action buttons frame
        action_frame = ttk.Frame(top_frame)
        action_frame.pack(fill='x', padx=5, pady=5)
        
        # Save and Train buttons
        save_btn = ttk.Button(action_frame, text="Save Data", command=self.save_data)
        save_btn.pack(side='left', padx=5)
        
        load_btn = ttk.Button(action_frame, text="Load Data", command=self.load_data)
        load_btn.pack(side='left', padx=5)
        
        train_btn = ttk.Button(action_frame, text="Train Model", command=self.train_model)
        train_btn.pack(side='left', padx=5)
        
        stats_btn = ttk.Button(action_frame, text="Show Variations", command=self.show_variation_stats)
        stats_btn.pack(side='left', padx=5)
        
        # Progress
        self.progress_var = tk.StringVar(value="Progress: 0/0")
        progress_label = ttk.Label(action_frame, textvariable=self.progress_var, font=('Arial', 10))
        progress_label.pack(side='right', padx=5)
        
        # Log frame
        log_frame = ttk.Frame(top_frame)
        log_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=8)
        self.log_text.pack(fill='both', expand=True)
        
        # Reset Data button in bottom right corner
        reset_frame = ttk.Frame(top_frame)
        reset_frame.pack(fill='x', padx=5, pady=5)
        
        reset_btn = ttk.Button(reset_frame, text="Reset Data", command=self.reset_data, 
                             style='Danger.TButton')
        reset_btn.pack(side='right', padx=5)
        
        # Matplotlib live plot in right frame
        self.fig, self.axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        self.fig.tight_layout(pad=2.0)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        self._init_plot_lines()
        
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
        
    def _setup_prediction_ui(self):
        # Control frame for prediction
        pred_control_frame = ttk.Frame(self.pred_frame)
        pred_control_frame.pack(fill='x', padx=10, pady=5)
        
        # Model toggle
        self.use_feature_classifier = tk.BooleanVar(value=False)
        toggle_btn = ttk.Checkbutton(pred_control_frame, text="Use Feature Model", variable=self.use_feature_classifier, onvalue=True, offvalue=False)
        toggle_btn.pack(side='left', padx=5)
        
        # Load model button
        load_btn = ttk.Button(pred_control_frame, text="Load Model", command=self.load_model)
        load_btn.pack(side='left', padx=5)
        
        # Clear model button
        clear_model_btn = ttk.Button(pred_control_frame, text="Clear Model", command=self.clear_model)
        clear_model_btn.pack(side='left', padx=5)
        
        # Start/Stop prediction button
        self.pred_btn = ttk.Button(pred_control_frame, text="Start Prediction", command=self.toggle_prediction)
        self.pred_btn.pack(side='left', padx=5)
        
        # Sliding window checkbox
        self.sliding_window_var = tk.BooleanVar(value=self.use_sliding_windows)
        sliding_window_cb = ttk.Checkbutton(pred_control_frame, text="Sliding Windows", 
                                          variable=self.sliding_window_var, 
                                          command=self._toggle_sliding_windows)
        sliding_window_cb.pack(side='left', padx=10)
        
        # Prediction result
        self.pred_result_var = tk.StringVar(value="Prediction: None")
        pred_result_label = ttk.Label(pred_control_frame, textvariable=self.pred_result_var, font=('Arial', 14, 'bold'))
        pred_result_label.pack(side='left', padx=20)
        
        # Confidence
        self.confidence_var = tk.StringVar(value="Confidence: 0%")
        confidence_label = ttk.Label(pred_control_frame, textvariable=self.confidence_var, font=('Arial', 12))
        confidence_label.pack(side='left', padx=10)
        
        # Prediction indicator
        self.pred_indicator = tk.Label(pred_control_frame, text="⏹", font=('Arial', 16), fg='gray')
        self.pred_indicator.pack(side='left', padx=10)
        
        # Prediction log
        pred_log_frame = ttk.Frame(self.pred_frame)
        pred_log_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.pred_log_text = scrolledtext.ScrolledText(pred_log_frame, height=6)
        self.pred_log_text.pack(fill='both', expand=True)
        
    def _setup_visualizer_ui(self):
        # Main frame with notebook for different views
        vis_notebook = ttk.Notebook(self.visualizer_frame)
        vis_notebook.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Data Visualization tab
        data_vis_frame = ttk.Frame(vis_notebook)
        vis_notebook.add(data_vis_frame, text="Data Visualization")
        
        # Controls for data visualization
        control_frame = ttk.Frame(data_vis_frame)
        control_frame.pack(fill='x', padx=10, pady=5)
        ttk.Label(control_frame, text="Class:").pack(side='left')
        self.vis_class_var = tk.StringVar(value=self.labels[0])
        class_dropdown = ttk.Combobox(control_frame, textvariable=self.vis_class_var, values=list(dict.fromkeys(self.labels+['IDLE','NOISE'])), state='readonly', width=10)
        class_dropdown.pack(side='left', padx=5)
        ttk.Label(control_frame, text="Sample index:").pack(side='left', padx=(10,0))
        self.vis_index_var = tk.IntVar(value=0)
        index_spin = ttk.Spinbox(control_frame, from_=0, to=299, textvariable=self.vis_index_var, width=5)
        index_spin.pack(side='left', padx=5)
        plot_btn = ttk.Button(control_frame, text="Plot Sample", command=self._plot_visualizer_sample)
        plot_btn.pack(side='left', padx=10)
        
        # Figure for data visualization
        self.vis_fig, (self.vis_ax_emg, self.vis_ax_quat) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
        self.vis_fig.tight_layout(pad=2.0)
        self.vis_canvas = FigureCanvasTkAgg(self.vis_fig, master=data_vis_frame)
        self.vis_canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=5)
        
        # Message label for data visualization
        self.vis_msg_var = tk.StringVar(value="")
        self.vis_msg_label = ttk.Label(data_vis_frame, textvariable=self.vis_msg_var, foreground='red')
        self.vis_msg_label.pack(pady=5)
        
        # Feature Classifier tab
        feature_frame = ttk.Frame(vis_notebook)
        vis_notebook.add(feature_frame, text="Feature Classifier")
        
        # Setup feature classifier UI
        self._setup_feature_classifier_ui_internal(feature_frame)
        
    def _setup_feature_classifier_ui_internal(self, parent_frame):
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.svm import SVC
            from sklearn.linear_model import LogisticRegression
            from sklearn.neural_network import MLPClassifier
            from sklearn.metrics import accuracy_score, confusion_matrix
            from sklearn.model_selection import train_test_split
        except ImportError:
            msg = ttk.Label(parent_frame, text="scikit-learn is not installed. Please install it to use this feature.", foreground='red')
            msg.pack(pady=20)
            return
            
        # Controls
        control_frame = ttk.Frame(parent_frame)
        control_frame.pack(fill='x', padx=10, pady=5)
        
        # Classifier selection
        ttk.Label(control_frame, text="Classifier:").pack(side='left')
        self.fc_classifier_var = tk.StringVar(value="RandomForest")
        classifier_options = ["RandomForest", "kNN", "SVM", "LogisticRegression", "MLP"]
        classifier_dropdown = ttk.Combobox(control_frame, textvariable=self.fc_classifier_var, values=classifier_options, state='readonly', width=18)
        classifier_dropdown.pack(side='left', padx=5)
        train_btn = ttk.Button(control_frame, text="Train", command=self._train_feature_classifier)
        train_btn.pack(side='left', padx=5)
        
        ttk.Label(control_frame, text="Class:").pack(side='left', padx=(20,0))
        self.fc_class_var = tk.StringVar(value=self.labels[0])
        class_dropdown = ttk.Combobox(control_frame, textvariable=self.fc_class_var, values=list(dict.fromkeys(self.labels+['IDLE','NOISE'])), state='readonly', width=10)
        class_dropdown.pack(side='left', padx=5)
        ttk.Label(control_frame, text="Sample index:").pack(side='left', padx=(10,0))
        self.fc_index_var = tk.IntVar(value=0)
        index_spin = ttk.Spinbox(control_frame, from_=0, to=299, textvariable=self.fc_index_var, width=5)
        index_spin.pack(side='left', padx=5)
        predict_btn = ttk.Button(control_frame, text="Predict Sample", command=self._predict_feature_sample)
        predict_btn.pack(side='left', padx=10)
        self.fc_pred_var = tk.StringVar(value="Prediction: N/A")
        pred_label = ttk.Label(control_frame, textvariable=self.fc_pred_var, font=('Arial', 12, 'bold'))
        pred_label.pack(side='left', padx=10)
        
        # Feature importance/confusion matrix plot
        self.fc_fig, (self.fc_ax_imp, self.fc_ax_cm) = plt.subplots(1, 2, figsize=(14, 3))
        self.fc_canvas = FigureCanvasTkAgg(self.fc_fig, master=parent_frame)
        self.fc_canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=5)
        self.fc_msg_var = tk.StringVar(value="")
        self.fc_msg_label = ttk.Label(parent_frame, textvariable=self.fc_msg_var, foreground='red')
        self.fc_msg_label.pack(pady=5)
        self.feature_classifier = None
        self.feature_names = None
        self.fc_class_list = list(dict.fromkeys(self.labels + ['IDLE', 'NOISE']))
        

        
    def _start_myo(self):
        self.hub_thread = threading.Thread(target=self._run_hub, daemon=True)
        self.hub_thread.start()
        
    def _run_hub(self):
        self.hub.run_forever(self)
        
    def on_connected(self, event):
        self.connected = True
        self.status_var.set(f"Status: Connected to {event.device_name}")
        event.device.stream_emg(True)
        self.log(f"Connected to Myo: {event.device_name}")
        
    def on_disconnected(self, event):
        self.connected = False
        self.status_var.set("Status: Disconnected")
        self.log("Disconnected from Myo")
        
    def on_emg(self, event):
        now = time.time()
        if now - self.last_emg_time < 0.01:  # 100Hz max
            return
        self.last_emg_time = now
        
        # Always store EMG for live display
        self.emg_buffer.append(event.emg)
        
        # Keep only last 200 samples for live display
        if len(self.emg_buffer) > 200:
            self.emg_buffer = self.emg_buffer[-200:]
        
        # Store for collection if collecting
        if self.collecting:
            self.collection_emg_buffer.append(event.emg)
            
        # Store for prediction if predicting
        if self.predicting:
            self.pred_emg_buffer.append(event.emg)
            # Keep only last 2 seconds of data (400 samples at 200Hz)
            if len(self.pred_emg_buffer) > 400:
                self.pred_emg_buffer = self.pred_emg_buffer[-400:]
            
            # Start prediction cycle every 2 seconds
            if not hasattr(self, 'last_prediction_time'):
                self.last_prediction_time = 0
            
            current_time = time.time()
            if current_time - self.last_prediction_time >= 2.0:  # Every 2 seconds
                self.last_prediction_time = current_time
                self.after(10, self.make_prediction)  # Small delay to avoid blocking
            
    def on_orientation(self, event):
        # Remove throttling for quaternions - let them come at their natural rate
        # now = time.time()
        # if now - self.last_quaternion_time < 0.01:  # 100Hz max (was 0.05 = 20Hz)
        #     return
        # self.last_quaternion_time = now
        
        # Always store quaternion for live display
        quaternion = [event.orientation.x, event.orientation.y, event.orientation.z, event.orientation.w]
        self.quaternion_buffer.append(quaternion)
        
        # Debug: log quaternion rate occasionally
        if len(self.quaternion_buffer) % 50 == 0:  # Every 50th sample
            self.log(f"Quaternion samples: {len(self.quaternion_buffer)}")
        
        # Keep only last 200 samples for live display
        if len(self.quaternion_buffer) > 200:
            self.quaternion_buffer = self.quaternion_buffer[-200:]
        
        # Store for collection if collecting
        if self.collecting:
            self.collection_quaternion_buffer.append(quaternion)
            
        # Store for prediction if predicting
        if self.predicting:
            self.pred_quaternion_buffer.append(quaternion)
            # Keep only last window_size samples for prediction
            if len(self.pred_quaternion_buffer) > self.prediction_window_size:
                self.pred_quaternion_buffer = self.pred_quaternion_buffer[-self.prediction_window_size:]
            
    def _start_recording(self):
        """Start recording data for the selected class"""
        print("🔍 DEBUG: _start_recording called")
        if not self.connected:
            print("❌ DEBUG: Not connected to Myo!")
            messagebox.showerror("Error", "Not connected to Myo!")
            return
            
        if self.collected[self.class_var.get()] >= self.samples_per_class:
            print("❌ DEBUG: Already collected max samples")
            messagebox.showinfo("Info", f"Already collected {self.samples_per_class} samples for {self.class_var.get()}")
            return
            
        print("✅ DEBUG: Starting recording process...")
        self.collecting = False  # Don't start collecting yet
        self.collection_emg_buffer = []
        self.collection_quaternion_buffer = []
        
        # Show preparation indicator
        self.recording_indicator.config(text="⏸", fg='orange')
        self.status_var.set(f"Get ready for {self.class_var.get()}...")
        self.log(f"Preparing to collect {self.class_var.get()} - recording starts in 1 second")
        
        # Wait 1 second then start recording
        print("🔍 DEBUG: Scheduling _begin_recording in 1 second...")
        self.after(1000, lambda: self._begin_recording(self.class_var.get()))
        
    def _begin_recording(self, label):
        """Actually start recording data"""
        print(f"🔍 DEBUG: _begin_recording called for label: {label}")
        self.collecting = True
        
        # Show recording indicator
        self.recording_indicator.config(text="⏺", fg='red')
        
        # Change plot background to indicate recording
        self.axs[0].set_facecolor('#fff0f0')  # Light red background
        self.axs[1].set_facecolor('#fff0f0')
        self.canvas.draw_idle()  # Use draw_idle instead of draw()
        
        # Play start recording beep in background
        print("🔍 DEBUG: About to start beep thread...")
        threading.Thread(target=self._play_start_beep, daemon=True).start()
        
        self.status_var.set(f"Recording {label}... Move your arm!")
        self.log(f"Recording started for {label}")
        
        # Start recording countdown
        print("🔍 DEBUG: Starting countdown...")
        self._countdown(self.duration_ms // 1000, label)
        
    def _play_start_beep(self):
        """Play start click in background thread"""
        print("🔊 DEBUG: Attempting to play start beep...")
        try:
            self.play_beep(1200, 100)  # Higher frequency, shorter duration for click
            print("✅ DEBUG: Start beep call completed")
        except Exception as e:
            print(f"❌ DEBUG: Start beep error: {e}")
            pass
        
    def _countdown(self, seconds, label):
        """Countdown timer for data collection"""
        if seconds > 0 and self.collecting:
            self.countdown_var.set(f"{seconds}s")
            self.after(1000, lambda: self._countdown(seconds - 1, label))
        elif self.collecting:
            self._finish_collection(label)
            
    def _finish_collection(self, label):
        if not self.collecting:
            return
            
        self.collecting = False
        
        # Hide recording indicator and countdown immediately
        self.recording_indicator.config(text="⏹", fg='gray')
        self.countdown_var.set("")
        
        # Restore plot background immediately
        self.axs[0].set_facecolor('#ffffff')  # White background
        self.axs[1].set_facecolor('#ffffff')
        self.canvas.draw_idle()  # Use draw_idle instead of draw()
        
        # Play stop recording beep in background
        threading.Thread(target=self._play_stop_beep, daemon=True).start()
        
        # Store data with length standardization
        if len(self.collection_emg_buffer) > 0:
            # Standardize length to 100 samples (0.5 seconds at 200Hz)
            target_length = 100
            
            # Get the minimum length to ensure both have same length
            min_len = min(len(self.collection_emg_buffer), len(self.collection_quaternion_buffer))
            
            # Truncate both to the same length, then to target length
            emg_data = np.array(self.collection_emg_buffer[:min_len])
            quaternion_data = np.array(self.collection_quaternion_buffer[:min_len])
            
            # If we have more than target_length, take the middle portion
            if min_len > target_length:
                start_idx = (min_len - target_length) // 2
                emg_data = emg_data[start_idx:start_idx + target_length]
                quaternion_data = quaternion_data[start_idx:start_idx + target_length]
            elif min_len < target_length:
                # Pad with zeros if too short (shouldn't happen with 2-second recording)
                emg_pad = np.zeros((target_length - min_len, 8))
                quaternion_pad = np.zeros((target_length - min_len, 4))
                emg_data = np.vstack([emg_data, emg_pad])
                quaternion_data = np.vstack([quaternion_data, quaternion_pad])
            
            # Store the standardized data
            self.data[label].append(emg_data)
            self.quaternion_data[label].append(quaternion_data)
            self.collected[label] += 1
            
            self.log(f"Collected sample {self.collected[label]}/{self.samples_per_class} for {label}")
            self.log(f"  Original EMG: {len(self.collection_emg_buffer)}, Quaternion: {len(self.collection_quaternion_buffer)}")
            self.log(f"  Standardized to: {emg_data.shape[0]} timesteps")
            self.log(f"  Variation: {self.variation_var.get()}")
            
            # Store variation note
            self.variation_notes[label].append(self.variation_var.get())
            
            # Check if we've reached the limit for this class
            if self.collected[label] >= self.samples_per_class:
                self.log(f"✅ Reached limit of {self.samples_per_class} samples for {label}")
                # Disable recording for this class
                if self.class_var.get() == label:
                    self.record_btn.config(state='disabled')
                    self.record_btn.config(text="Limit Reached")
            
            # Update progress
            total_collected = sum(self.collected.values())
            total_needed = len(self.labels) * self.samples_per_class
            self.progress_var.set(f"Progress: {total_collected}/{total_needed}")
            
            # Update class count
            self._update_class_count(None)
            
            # Update plot
            self._update_plot(label)
            
        self.status_var.set("Status: Ready")
        
    def _play_stop_beep(self):
        """Play stop clicks in background thread"""
        print("🔊 DEBUG: Attempting to play stop beeps...")
        try:
            self.play_beep(800, 80)  # First click
            time.sleep(0.02)
            self.play_beep(600, 80)  # Second click (lower pitch)
            print("✅ DEBUG: Stop beeps call completed")
        except Exception as e:
            print(f"❌ DEBUG: Stop beeps error: {e}")
            pass
        
    def _update_live_plot(self):
        """Update the plot with live EMG data - optimized for performance"""
        if len(self.emg_buffer) == 0:
            return
            
        try:
            emg_arr = np.array(self.emg_buffer)
            window = len(emg_arr)
            
            # Update EMG plot more efficiently
            for i, line in enumerate(self.emg_lines):
                line.set_data(np.arange(window), emg_arr[:, i])
            self.axs[0].set_xlim(0, window)
            if emg_arr.size:
                self.axs[0].set_ylim(np.min(emg_arr)-5, np.max(emg_arr)+5)
                
            # Update quaternion plot if we have data
            if len(self.quaternion_buffer) > 0:
                quaternion_arr = np.array(self.quaternion_buffer)
                quat_window = min(window, len(quaternion_arr))
                for i, line in enumerate(self.quaternion_lines):
                    line.set_data(np.arange(quat_window), quaternion_arr[-quat_window:, i])
                self.axs[1].set_xlim(0, quat_window)
                if quaternion_arr.size:
                    self.axs[1].set_ylim(np.min(quaternion_arr)-0.1, np.max(quaternion_arr)+0.1)
                    
            # Use blitting for faster updates
            self.canvas.draw_idle()
            
        except Exception as e:
            # Don't let plot errors crash the GUI
            pass
        
    def _update_plot(self, label):
        if not self.data[label]:
            return
            
        # Get latest data
        emg_arr = self.data[label][-1]
        quaternion_arr = self.quaternion_data[label][-1]
        
        window = min(200, len(emg_arr))
        
        # Update EMG plot
        for i, line in enumerate(self.emg_lines):
            line.set_data(np.arange(window), emg_arr[-window:, i])
        self.axs[0].set_xlim(0, window)
        if emg_arr.size:
            self.axs[0].set_ylim(np.min(emg_arr)-5, np.max(emg_arr)+5)
            
        # Update quaternion plot
        if len(quaternion_arr) >= window:
            for i, line in enumerate(self.quaternion_lines):
                line.set_data(np.arange(window), quaternion_arr[-window:, i])
            self.axs[1].set_xlim(0, window)
            if quaternion_arr.size:
                self.axs[1].set_ylim(np.min(quaternion_arr)-0.1, np.max(quaternion_arr)+0.1)
                
        self.canvas.draw()
        
    def train_model(self):
        """Train the model with collected data"""
        if not any(self.data.values()):
            messagebox.showerror("Error", "No data to train with!")
            return
            
        try:
            self.log("🚀 Starting model training...")
            
            # Check if we have enough data
            total_samples = sum(len(self.data[label]) for label in self.labels)
            if total_samples < 3:
                messagebox.showerror("Error", f"Not enough data! Need at least 3 samples, got {total_samples}")
                return
            
            # Prepare training data for sequence model
            all_windows = []
            all_labels = []
            window_size = 100  # Match the model's expected window size
            
            self.log("📊 Preparing sequence training data...")
            
            for label in self.labels:
                if not self.data[label]:
                    self.log(f"   Skipping {label} - no data")
                    continue
                    
                emg_arrs = self.data[label]
                quaternion_arrs = self.quaternion_data[label]
                
                # Ensure all arrays are float32
                emg_arrs = [np.array(x, dtype=np.float32) for x in emg_arrs]
                quaternion_arrs = [np.array(x, dtype=np.float32) for x in quaternion_arrs]
                
                self.log(f"   Processing {label}: {len(emg_arrs)} samples")
                
                for i, (emg_arr, quaternion_arr) in enumerate(zip(emg_arrs, quaternion_arrs)):
                    # Data is now standardized to 100 timesteps during collection
                    if len(emg_arr) >= window_size and len(quaternion_arr) >= window_size:
                        # Create sliding windows for sequence model
                        for j in range(0, len(emg_arr) - window_size + 1, window_size // 2):
                            emg_win = emg_arr[j:j+window_size]
                            quaternion_win = quaternion_arr[j:j+window_size]
                            
                            # Preprocess EMG
                            emg_proc = preprocess_emg(emg_win)
                            
                            # Create position-focused sequence input for LSTM model
                            # Shape: (window_size, 12) - 8 EMG channels + 4 quaternion components
                            # Using position emphasis of 0.8 (80% focus on position, 20% on EMG)
                            X_win = create_position_focused_sequence(emg_proc, quaternion_win, position_emphasis=0.8)
                            all_windows.append(X_win)
                            all_labels.append(label)
            
            if not all_windows:
                messagebox.showerror("Error", "No valid training sequences could be extracted!")
                return
                
            # Convert to numpy arrays for sequence model (force float32)
            X = np.array(all_windows, dtype=np.float32)  # Shape: (num_samples, window_size, 12)
            y = np.array(all_labels)
            
            # Debug: print dtype and shape
            self.log(f"DEBUG: X dtype: {X.dtype}, shape: {X.shape}")
            self.log(f"DEBUG: y dtype: {y.dtype}, shape: {y.shape}")
            
            # Label encoding debug
            self.log(f"DEBUG: Unique labels: {np.unique(y)}")
            
            # Split into train/val (stratified)
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y)
            self.log(f"DEBUG: Train shape: {X_train.shape}, Val shape: {X_val.shape}")
            self.log(f"DEBUG: Train label distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
            self.log(f"DEBUG: Val label distribution: {dict(zip(*np.unique(y_val, return_counts=True)))}")
            
            # Train model
            self.log("🧠 Training LSTM sequence model...")
            model_type = self.dl_model_var.get() if hasattr(self, 'dl_model_var') else 'LSTM'
            self.log(f"Training model type: {model_type}")
            self.model, self.le = train_model(X_train, y_train, self.labels)
            self.log("✅ Model training completed!")
            
            # Save model automatically with timestamp
            import time
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            model_filename = f"myo_model_{timestamp}.h5"
            le_filename = f"myo_model_{timestamp}_labels.pkl"
            self.model.save(model_filename)
            import pickle
            with open(le_filename, 'wb') as f:
                pickle.dump(self.le, f)
            self.log(f"💾 Model saved automatically to: {model_filename}")
            self.log(f"💾 Label encoder saved to: {le_filename}")
            
            # --- DIAGNOSTICS ---
            self.log("\n🔬 DIAGNOSTICS:")
            from sklearn.metrics import confusion_matrix, classification_report
            # Predict on train and val
            y_train_pred = np.argmax(self.model.predict(X_train, verbose=0), axis=1)
            y_val_pred = np.argmax(self.model.predict(X_val, verbose=0), axis=1)
            # Decode labels if using LabelEncoder
            if self.le is not None:
                y_train_true = self.le.transform(y_train)
                y_val_true = self.le.transform(y_val)
            else:
                y_train_true = y_train
                y_val_true = y_val
            # Confusion matrix
            cm = confusion_matrix(y_val_true, y_val_pred)
            self.log(f"Confusion Matrix (Validation):\n{cm}")
            # Per-class accuracy
            class_report = classification_report(y_val_true, y_val_pred, target_names=list(self.le.classes_) if self.le is not None else None)
            self.log(f"Classification Report (Validation):\n{class_report}")
            # Print for train set too
            cm_train = confusion_matrix(y_train_true, y_train_pred)
            self.log(f"Confusion Matrix (Train):\n{cm_train}")
            class_report_train = classification_report(y_train_true, y_train_pred, target_names=list(self.le.classes_) if self.le is not None else None)
            self.log(f"Classification Report (Train):\n{class_report_train}")
            
            messagebox.showinfo("Success", f"Model trained and saved!\n\nModel: {model_filename}\nLabels: {le_filename}")
            
        except Exception as e:
            self.log(f"❌ Training error: {e}")
            import traceback
            self.log(f"   Traceback: {traceback.format_exc()}")
            messagebox.showerror("Error", f"Training failed:\n{e}")
            
    def load_model(self):
        """Load a trained model"""
        try:
            model_filename = filedialog.askopenfilename(
                filetypes=[("H5 files", "*.h5"), ("All files", "*.*")]
            )
            
            if model_filename:
                # Load model
                self.model = tf.keras.models.load_model(model_filename)
                
                # Check model input shape
                expected_shape = self.model.input_shape
                self.pred_log(f"Model expects input shape: {expected_shape}")
                
                # Load label encoder
                le_filename = model_filename.replace('.h5', '_labels.pkl')
                with open(le_filename, 'rb') as f:
                    self.le = pickle.load(f)
                    
                self.pred_log(f"Model loaded from {model_filename}")
                self.pred_log(f"Labels: {list(self.le.classes_)}")
                
        except Exception as e:
            self.pred_log(f"Load error: {e}")
            messagebox.showerror("Error", f"Failed to load model: {e}")
            
    def toggle_prediction(self):
        """Toggle prediction mode on/off"""
        if not hasattr(self, 'model') or self.model is None:
            messagebox.showerror("Error", "No model loaded!")
            return
            
        if not self.predicting:
            self.predicting = True
            self.pred_btn.config(text="Stop Prediction")
            self.prediction_window_size = 100  # Match training window size
            self.pred_emg_buffer = []
            self.pred_quaternion_buffer = []
            self.last_prediction_time = 0
            
            # Log prediction mode
            mode = "Sliding Windows" if self.use_sliding_windows else "Single Window"
            self.pred_log(f"Prediction started - Mode: {mode}")
            if self.use_sliding_windows:
                self.pred_log(f"  Temporal invariance: Gestures can be recognized at any time in the window")
                self.pred_log(f"  Window size: {self.sliding_window_size}, Overlap: {self.sliding_window_overlap*100}%")
            else:
                self.pred_log(f"  Single window: Gestures must be at the end of the window")
        else:
            self.predicting = False
            self.pred_btn.config(text="Start Prediction")
            self.pred_log("Prediction stopped")
            
    def make_prediction(self):
        # Use prediction buffers instead of display buffers
        min_len = min(len(self.pred_emg_buffer), len(self.pred_quaternion_buffer))
        window_size = 100
        
        # Visual feedback: show prediction is starting
        self.pred_result_var.set("Analyzing gesture...")
        self.pred_indicator.config(text="⏺", fg='red')  # Show recording indicator
        
        if min_len >= window_size:
            try:
                # Implement sliding window prediction
                if self.use_sliding_windows:
                    # Use the most recent window_size samples (sliding window)
                    emg_win = np.array(self.pred_emg_buffer[-window_size:])
                    quaternion_win = np.array(self.pred_quaternion_buffer[-window_size:])
                else:
                    # Use the first window_size samples (fixed window)
                    emg_win = np.array(self.pred_emg_buffer[:window_size])
                    quaternion_win = np.array(self.pred_quaternion_buffer[:window_size])
                
                # Check for movement/activity in the data (temporarily disabled for debugging)
                emg_variance = np.var(emg_win)
                quaternion_variance = np.var(quaternion_win)
                
                # Debug: log variance values and data characteristics
                self.pred_log(f"EMG variance: {emg_variance:.2f}, Quaternion variance: {quaternion_variance:.4f}")
                self.pred_log(f"EMG range: [{np.min(emg_win):.2f}, {np.max(emg_win):.2f}]")
                self.pred_log(f"Quaternion range: [{np.min(quaternion_win):.4f}, {np.max(quaternion_win):.4f}]")
                
                # If data is too static, consider it idle (temporarily disabled)
                # if emg_variance < 100 and quaternion_variance < 0.01:  # Thresholds for movement detection
                #     self.pred_result_var.set("Idle (no movement detected)")
                #     self.pred_indicator.config(text="⏹", fg='gray')
                #     return
                if hasattr(self, 'use_feature_classifier') and self.use_feature_classifier.get():
                    if hasattr(self, 'feature_classifier') and self.feature_classifier is not None:
                        # Use feature-based classifier
                        features = self.extract_features(emg_win, quaternion_win).reshape(1, -1)
                        pred = self.feature_classifier.predict(features)[0]
                        self.pred_result_var.set(f"Feature Model Prediction: {pred}")
                        self.last_pred = pred
                        self.last_print_time = time.time()
                    else:
                        self.pred_result_var.set("No feature model trained. Train one in the Feature Classifier tab.")
                        self.last_pred = None
                        self.last_print_time = time.time()
                else:
                    if not hasattr(self, 'model') or self.model is None:
                        self.pred_result_var.set("No deep model loaded. Load a model or use the feature model.")
                        self.last_pred = None
                        self.last_print_time = time.time()
                        return
                    # Use deep model
                    emg_proc = preprocess_emg(emg_win)
                    X_win = np.concatenate([emg_proc, quaternion_win], axis=1)  # shape (window_size, 12)
                    pred = self.model.predict(X_win[np.newaxis, ...], verbose=0)
                    text = self.le.inverse_transform([np.argmax(pred)])[0]
                    confidence = np.max(pred)
                    
                    # Debug: show all predictions regardless of confidence
                    current_time = time.strftime("%H:%M:%S")
                    self.pred_log(f"[{current_time}] Raw prediction: {text} (confidence: {confidence:.3f})")
                    
                    # Debug: show all class probabilities
                    self.pred_log(f"All probabilities: A={prediction[0][0]:.3f}, B={prediction[0][1]:.3f}, C={prediction[0][2]:.3f}, IDLE={prediction[0][3]:.3f}, NOISE={prediction[0][4]:.3f}")
                    
                    # Debug: show input data characteristics
                    self.pred_log(f"Input EMG mean: {np.mean(emg_win):.2f}, std: {np.std(emg_win):.2f}")
                    self.pred_log(f"Input quaternion mean: {np.mean(quaternion_win):.4f}, std: {np.std(quaternion_win):.4f}")
                    
                    # Show all predictions but highlight high confidence ones
                    if confidence > 0.8:
                        self.pred_result_var.set(f"Predicted: {text} (confidence: {confidence:.3f})")
                        self.last_pred = text
                        self.last_print_time = time.time()
                        print(f"High confidence prediction: {text} ({confidence:.3f})")
                    elif confidence > 0.5:
                        self.pred_result_var.set(f"Low confidence: {text} ({confidence:.3f})")
                        self.last_pred = text
                        print(f"Low confidence prediction: {text} ({confidence:.3f})")
                    else:
                        # Show idle state for very low confidence
                        self.pred_result_var.set("Idle (no clear gesture)")
                        self.last_pred = None
                        
                        # Only log occasionally to avoid spam
                        if hasattr(self, 'last_idle_log') and time.time() - self.last_idle_log > 5:
                            self.pred_log(f"[{current_time}] Idle (confidence: {confidence:.3f})")
                            self.last_idle_log = time.time()
                        elif not hasattr(self, 'last_idle_log'):
                            self.last_idle_log = time.time()
                    
                    # Clear prediction indicator
                    self.pred_indicator.config(text="⏹", fg='gray')
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                error_msg = f"Prediction error: {e}"
                self.pred_result_var.set("Prediction error!")
                print(error_msg)
                print(tb)
                self.pred_log(error_msg)
                
                # Clear prediction indicator
                self.pred_indicator.config(text="⏹", fg='gray')
        else:
            self.pred_result_var.set("Not enough data for prediction.")
            # Debug: show buffer lengths
            self.pred_log(f"Buffer lengths - EMG: {len(self.pred_emg_buffer)}, Quaternion: {len(self.pred_quaternion_buffer)}")
        # Don't clear the prediction buffers - they accumulate data continuously
        
    def pred_log(self, message):
        """Log message to prediction log"""
        timestamp = time.strftime("%H:%M:%S")
        self.pred_log_text.insert('end', f"[{timestamp}] {message}\n")
        self.pred_log_text.see('end')
        
    def log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert('end', f"[{timestamp}] {message}\n")
        self.log_text.see('end')
        
    def save_data(self):
        """Save collected data to file"""
        if not any(self.data.values()):
            messagebox.showerror("Error", "No data to save!")
            return
            
        try:
            # Ask for save location
            filename = filedialog.asksaveasfilename(
                title="Save Training Data",
                defaultextension=".npz",
                filetypes=[("NumPy files", "*.npz"), ("All files", "*.*")],
                initialdir=os.getcwd()  # Start in current directory
            )
            
            if filename:  # User didn't cancel
                # Prepare data for saving
                save_dict = {}
                for label in self.labels:
                    if self.data[label]:
                        save_dict[f'{label}_emg'] = np.array(self.data[label], dtype=object)
                        save_dict[f'{label}_quaternion'] = np.array(self.quaternion_data[label], dtype=object)
                        
                # Save the data
                np.savez(filename, **save_dict)
                
                self.log(f"✅ Data saved successfully to: {filename}")
                self.log(f"   Labels saved: {[label for label in self.labels if self.data[label]]}")
                
                # Show success message
                messagebox.showinfo("Success", f"Data saved to:\n{filename}")
                
        except Exception as e:
            self.log(f"❌ Error saving data: {e}")
            messagebox.showerror("Error", f"Failed to save data:\n{e}")

    def load_data(self):
        """Load previously saved training data"""
        try:
            # Ask for file to load
            filename = filedialog.askopenfilename(
                title="Load Training Data",
                filetypes=[("NumPy files", "*.npz"), ("All files", "*.*")],
                initialdir=os.getcwd()  # Start in current directory
            )
            
            if filename:  # User didn't cancel
                # Load the data
                loaded_data = np.load(filename, allow_pickle=True)
                
                # Clear existing data
                for label in self.labels:
                    self.data[label] = []
                    self.quaternion_data[label] = []
                    self.collected[label] = 0
                
                # Load data for each label
                for label in self.labels:
                    emg_key = f'{label}_emg'
                    quaternion_key = f'{label}_quaternion'
                    
                    if emg_key in loaded_data and quaternion_key in loaded_data:
                        # Convert to float32 arrays to avoid dtype=object issues
                        self.data[label] = [np.array(x, dtype=np.float32) for x in loaded_data[emg_key]]
                        self.quaternion_data[label] = [np.array(x, dtype=np.float32) for x in loaded_data[quaternion_key]]
                        self.collected[label] = len(self.data[label])
                        
                        self.log(f"✅ Loaded {self.collected[label]} samples for {label}")
                    else:
                        self.log(f"⚠️ No data found for {label} in file")
                        self.data[label] = []
                        self.quaternion_data[label] = []
                        self.collected[label] = 0
                
                # Update progress
                total_collected = sum(self.collected.values())
                total_needed = len(self.labels) * self.samples_per_class
                self.progress_var.set(f"Progress: {total_collected}/{total_needed}")
                
                # Update class count
                self._update_class_count(None)
                
                self.log(f"✅ Data loaded successfully from: {filename}")
                self.log(f"   Total samples loaded: {total_collected}")
                
                # Show success message
                messagebox.showinfo("Success", f"Data loaded from:\n{filename}\n\nTotal samples: {total_collected}")
                
        except Exception as e:
            self.log(f"❌ Error loading data: {e}")
            messagebox.showerror("Error", f"Failed to load data:\n{e}")

    def _schedule_plot_update(self):
        """Schedule periodic plot updates for better performance"""
        self._update_live_plot()
        self.after(50, self._schedule_plot_update)  # Update every 50ms (20 FPS)

    def _auto_load_default_data(self):
        """Automatically load the default data file if it exists"""
        default_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "data.npz")
        
        if os.path.exists(default_data_path):
            try:
                self.log("🔄 Auto-loading default data file...")
                
                # Load the data
                loaded_data = np.load(default_data_path, allow_pickle=True)
                
                # Clear existing data
                for label in self.labels:
                    self.data[label] = []
                    self.quaternion_data[label] = []
                    self.collected[label] = 0
                
                # Load data for each label
                for label in self.labels:
                    emg_key = f'{label}_emg'
                    quaternion_key = f'{label}_quaternion'
                    
                    if emg_key in loaded_data and quaternion_key in loaded_data:
                        self.data[label] = list(loaded_data[emg_key])
                        self.quaternion_data[label] = list(loaded_data[quaternion_key])
                        self.collected[label] = len(self.data[label])
                        
                        self.log(f"✅ Auto-loaded {self.collected[label]} samples for {label}")
                    else:
                        self.log(f"⚠️ No data found for {label} in file")
                        self.data[label] = []
                        self.quaternion_data[label] = []
                        self.collected[label] = 0
                
                # Update progress
                total_collected = sum(self.collected.values())
                total_needed = len(self.labels) * self.samples_per_class
                self.progress_var.set(f"Progress: {total_collected}/{total_needed}")
                
                # Update class count
                self._update_class_count(None)
                
                self.log(f"✅ Default data loaded successfully from: {default_data_path}")
                self.log(f"   Total samples loaded: {total_collected}")
                
            except Exception as e:
                self.log(f"❌ Error auto-loading default data: {e}")
        else:
            self.log("ℹ️ No default data file found at: data/data.npz")

    def _on_key_press(self, event):
        """Handle keyboard shortcuts"""
        if event.char.lower() == 'r':
            # Only allow recording if not already collecting and connected
            if not self.collecting and self.connected:
                self._start_recording()
        elif event.char.lower() == 's':
            # 'S' key to save data
            self.save_data()
        elif event.char.lower() == 't':
            # 'T' key to train model
            self.train_model()
        elif event.char.lower() == 'u':
            # 'U' key to undo last recording
            self._undo_last_recording()

    def new_training(self):
        """Clear all collected data and reset progress counters"""
        for label in self.labels:
            self.data[label] = []
            self.quaternion_data[label] = []
            self.collected[label] = 0
        
        self.progress_var.set("Progress: 0/0")
        self.log("All collected data cleared and progress counters reset")

    def reset_data(self):
        """Reset all collected data and reset progress counters"""
        # Show confirmation dialog
        total_samples = sum(self.collected.values())
        if total_samples > 0:
            result = messagebox.askyesno(
                "Confirm Reset", 
                f"This will clear all {total_samples} collected samples.\n\nAre you sure you want to reset all data?",
                icon='warning'
            )
            if not result:
                return
        
        # Clear all data
        for label in self.labels:
            self.data[label] = []
            self.quaternion_data[label] = []
            self.collected[label] = 0
        
        self.progress_var.set("Progress: 0/0")
        self.log("✅ All collected data cleared and progress counters reset")

    def _undo_last_recording(self):
        """Undo the last recorded data for the current class"""
        if not self.data[self.class_var.get()]:
            messagebox.showinfo("Info", "No data to undo for the current class")
            return
            
        # Remove the last recorded data
        self.data[self.class_var.get()].pop()
        self.quaternion_data[self.class_var.get()].pop()
        self.collected[self.class_var.get()] -= 1
        
        # Update progress
        total_collected = sum(self.collected.values())
        total_needed = len(self.labels) * self.samples_per_class
        self.progress_var.set(f"Progress: {total_collected}/{total_needed}")
        
        # Update class count
        self._update_class_count(None)
        
        # Update plot
        self._update_plot(self.class_var.get())
        
        self.log(f"✅ Last recorded data for {self.class_var.get()} removed")

    def _update_class_count(self, event):
        """Update the class count label when the class changes"""
        selected_class = self.class_var.get()
        self.class_count_var.set(f"({self.collected[selected_class]}/{self.samples_per_class})")
        
        # Update record button state
        if self.collected[selected_class] >= self.samples_per_class:
            self.record_btn.config(state='disabled')
            self.record_btn.config(text="Limit Reached")
        else:
            self.record_btn.config(state='normal')
            self.record_btn.config(text="Record")

    def clear_model(self):
        """Clear the current model and reset prediction variables"""
        self.model = None
        self.le = None
        self.pred_emg_buffer = []
        self.pred_quaternion_buffer = []
        self.prediction_window_size = 400  # Reset to default
        self.last_prediction_time = 0
        self.pred_log("Model cleared")
        self.pred_result_var.set("Prediction: None")
        self.confidence_var.set("Confidence: 0%")
        self.pred_log_text.delete('1.0', 'end')
        self.log("Model cleared")

    def show_variation_stats(self):
        """Show variation statistics for collected data"""
        self.log("📊 Variation Statistics:")
        for label in self.labels:
            if self.variation_notes[label]:
                self.log(f"  {label}: {len(self.variation_notes[label])} samples")
                # Count variations
                variation_counts = {}
                for variation in self.variation_notes[label]:
                    variation_counts[variation] = variation_counts.get(variation, 0) + 1
                
                for variation, count in variation_counts.items():
                    self.log(f"    {variation}: {count} samples")
            else:
                self.log(f"  {label}: No samples collected")

    def _toggle_sliding_windows(self):
        """Toggle sliding windows option"""
        self.use_sliding_windows = self.sliding_window_var.get()
        self.log(f"Sliding windows: {'Enabled' if self.use_sliding_windows else 'Disabled'}")

    def play_beep(self, frequency, duration_ms):
        # Use winsound instead of sounddevice for more reliable beeps
        print(f"🔊 DEBUG: play_beep called with freq={frequency}Hz, duration={duration_ms}ms")
        try:
            winsound.Beep(int(frequency), int(duration_ms))
            print("✅ DEBUG: winsound.Beep completed")
        except Exception as e:
            print(f"❌ DEBUG: winsound.Beep failed: {e}")
            # Fallback to sounddevice
            try:
                play_sine_wave_beep(frequency, duration_ms, volume=0.8)
                print("✅ DEBUG: sounddevice fallback completed")
            except Exception as e2:
                print(f"❌ DEBUG: sounddevice fallback also failed: {e2}")

    def _plot_visualizer_sample(self):
        cls = self.vis_class_var.get()
        idx = self.vis_index_var.get()
        # Try to get data
        emg_data = self.data.get(cls)
        quat_data = self.quaternion_data.get(cls)
        if emg_data is None or quat_data is None or len(emg_data)==0 or idx>=len(emg_data):
            self.vis_msg_var.set(f"No data for class {cls} at index {idx}.")
            self.vis_ax_emg.clear()
            self.vis_ax_quat.clear()
            self.vis_canvas.draw()
            return
        self.vis_msg_var.set("")
        emg = np.array(emg_data[idx], dtype=np.float32)
        quat = np.array(quat_data[idx], dtype=np.float32)
        self.vis_ax_emg.clear()
        for ch in range(emg.shape[1]):
            self.vis_ax_emg.plot(emg[:,ch], label=f'EMG {ch+1}')
        self.vis_ax_emg.set_title(f'EMG (Class {cls}, Sample {idx})')
        self.vis_ax_emg.legend(fontsize=7, ncol=4)
        self.vis_ax_emg.set_ylabel('EMG')
        self.vis_ax_quat.clear()
        for q in range(quat.shape[1]):
            self.vis_ax_quat.plot(quat[:,q], label=f'Q{q+1}')
        self.vis_ax_quat.set_title(f'Quaternion (Class {cls}, Sample {idx})')
        self.vis_ax_quat.legend(fontsize=7, ncol=4)
        self.vis_ax_quat.set_ylabel('Quaternion')
        self.vis_ax_quat.set_xlabel('Timestep')
        self.vis_canvas.draw()

    def extract_features(self, emg, quat):
        # emg: (100, 8), quat: (100, 4)
        features = []
        # EMG features
        for ch in range(emg.shape[1]):
            x = emg[:, ch].astype(np.float32)
            features += [
                np.mean(x), np.std(x), np.min(x), np.max(x), np.ptp(x),
                np.sum(np.abs(np.diff(np.sign(x))))  # zero-crossings
            ]
        # Quaternion features
        for q in range(quat.shape[1]):
            x = quat[:, q].astype(np.float32)
            features += [
                np.mean(x), np.std(x), np.min(x), np.max(x), np.ptp(x)
            ]
        return np.array(features)

    def _train_feature_classifier(self):
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.svm import SVC
            from sklearn.linear_model import LogisticRegression
            from sklearn.neural_network import MLPClassifier
            from sklearn.metrics import accuracy_score, confusion_matrix
            from sklearn.model_selection import train_test_split
        except ImportError:
            self.fc_msg_var.set("scikit-learn is not installed.")
            return
        # Gather features and labels
        X, y, feature_names = [], [], []
        class_list = self.fc_class_list
        for cls in class_list:
            emg_data = self.data.get(cls)
            quat_data = self.quaternion_data.get(cls)
            if emg_data is None or quat_data is None:
                continue
            for i in range(len(emg_data)):
                X.append(self.extract_features(np.array(emg_data[i], dtype=np.float32), np.array(quat_data[i], dtype=np.float32)))
                y.append(cls)
        if not X:
            self.fc_msg_var.set("No data loaded.")
            return
        X = np.array(X)
        y = np.array(y)
        # Feature names
        feature_names = []
        for ch in range(8):
            feature_names += [f'EMG{ch+1}_mean', f'EMG{ch+1}_std', f'EMG{ch+1}_min', f'EMG{ch+1}_max', f'EMG{ch+1}_range', f'EMG{ch+1}_zcr']
        for q in range(4):
            feature_names += [f'Q{q+1}_mean', f'Q{q+1}_std', f'Q{q+1}_min', f'Q{q+1}_max', f'Q{q+1}_range']
        self.feature_names = feature_names
        # Stratified train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        # Select classifier
        clf_name = self.fc_classifier_var.get()
        if clf_name == "RandomForest":
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
        elif clf_name == "kNN":
            clf = KNeighborsClassifier(n_neighbors=5)
        elif clf_name == "SVM":
            clf = SVC(probability=True)
        elif clf_name == "LogisticRegression":
            clf = LogisticRegression(max_iter=1000)
        elif clf_name == "MLP":
            clf = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=1000)
        else:
            self.fc_msg_var.set(f"Unknown classifier: {clf_name}")
            return
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))
        self.fc_msg_var.set(f"{clf_name} accuracy: {acc*100:.1f}% on holdout set ({len(X_test)} samples)")
        self.feature_classifier = clf
        # Plot feature importances if available
        self.fc_ax_imp.clear()
        if clf_name == "RandomForest" and hasattr(clf, 'feature_importances_'):
            importances = clf.feature_importances_
            idxs = np.argsort(importances)[::-1][:20]  # Top 20
            self.fc_ax_imp.barh([self.feature_names[i] for i in idxs][::-1], importances[idxs][::-1])
            self.fc_ax_imp.set_title('Top 20 Feature Importances')
        elif clf_name == "LogisticRegression" and hasattr(clf, 'coef_'):
            importances = np.abs(clf.coef_).mean(axis=0)
            idxs = np.argsort(importances)[::-1][:20]
            self.fc_ax_imp.barh([self.feature_names[i] for i in idxs][::-1], importances[idxs][::-1])
            self.fc_ax_imp.set_title('Top 20 Feature Importances')
        else:
            self.fc_ax_imp.text(0.5, 0.5, 'No feature importances available', ha='center', va='center')
            self.fc_ax_imp.set_title('Feature Importances')
        # Plot confusion matrix
        from sklearn.metrics import ConfusionMatrixDisplay
        self.fc_ax_cm.clear()
        y_pred = clf.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, labels=class_list)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_list)
        disp.plot(ax=self.fc_ax_cm, colorbar=False)
        self.fc_ax_cm.set_title('Confusion Matrix')
        self.fc_fig.tight_layout()
        self.fc_canvas.draw()

    def _predict_feature_sample(self):
        if self.feature_classifier is None:
            self.fc_msg_var.set("Train the classifier first.")
            return
        cls = self.fc_class_var.get()
        idx = self.fc_index_var.get()
        emg_data = self.data.get(cls)
        quat_data = self.quaternion_data.get(cls)
        if emg_data is None or quat_data is None or len(emg_data)==0 or idx>=len(emg_data):
            self.fc_msg_var.set(f"No data for class {cls} at index {idx}.")
            return
        feat = self.extract_features(np.array(emg_data[idx], dtype=np.float32), np.array(quat_data[idx], dtype=np.float32)).reshape(1, -1)
        pred = self.feature_classifier.predict(feat)[0]
        self.fc_pred_var.set(f"Prediction: {pred}")

    def build_model(self, model_type, input_shape, num_classes):
        from tensorflow.keras import layers, models
        model = models.Sequential()
        if model_type == 'LSTM':
            model.add(layers.LSTM(64, input_shape=input_shape))
        elif model_type == 'GRU':
            model.add(layers.GRU(64, input_shape=input_shape))
        elif model_type == 'SimpleRNN':
            model.add(layers.SimpleRNN(64, input_shape=input_shape))
        elif model_type == 'Conv1D':
            model.add(layers.Conv1D(32, 3, activation='relu', input_shape=input_shape))
            model.add(layers.GlobalMaxPooling1D())
        elif model_type == 'Dense':
            model.add(layers.Flatten(input_shape=input_shape))
            model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(num_classes, activation='softmax'))
        return model

if __name__ == "__main__":
    print("🚀 Starting Simplified Myo Handwriting Recognition GUI")
    print("Using EMG + Quaternion data")
    print("=" * 50)
    
    try:
        app = SimpleMyoGUI(labels=['A', 'B', 'C'], samples_per_class=300, duration_ms=2000)
        print("✅ GUI created successfully")
        app.mainloop()
    except Exception as e:
        print(f"❌ Error starting GUI: {e}")
        import traceback
        traceback.print_exc() 