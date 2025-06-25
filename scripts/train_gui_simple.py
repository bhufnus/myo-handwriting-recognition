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
from src.preprocessing import preprocess_emg
from src.model import train_model
from src.utils import get_sdk_path
import tensorflow as tf

# Try to import sounddevice for square wave beeps
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    print("⚠️ sounddevice not available, using winsound for beeps")

# Global audio stream for beeps
audio_stream = None

def play_square_wave_beep(frequency, duration_ms, volume=0.2):
    """Play a sine wave beep using sounddevice with improved envelope and audio handling"""
    global audio_stream
    
    if not SOUNDDEVICE_AVAILABLE:
        return
    
    try:
        # Generate sine wave samples with higher quality
        sample_rate = 48000  # Higher sample rate for better quality
        duration_sec = duration_ms / 1000.0
        num_samples = int(sample_rate * duration_sec)
        
        # Add some padding to prevent clipping
        padding_samples = int(sample_rate * 0.01)  # 10ms padding
        total_samples = num_samples + padding_samples
        
        t = np.linspace(0, duration_sec, num_samples, False)
        
        # Create sine wave
        sine_wave = np.sin(2 * np.pi * frequency * t)
        
        # Create smoother envelope with longer transitions
        # Envelope parameters (as fractions of total duration)
        attack_time = 0.10  # 15% of duration for attack
        release_time = 0.20  # 25% of duration for release
        
        attack_samples = max(1, int(num_samples * attack_time))
        release_samples = max(1, int(num_samples * release_time))
        sustain_samples = num_samples - attack_samples - release_samples
        
        # Ensure we have at least some sustain
        if sustain_samples < 1:
            # If duration is too short, use shorter attack/release
            attack_samples = max(1, num_samples // 4)
            release_samples = max(1, num_samples // 4)
            sustain_samples = num_samples - attack_samples - release_samples
        
        # Create envelope with smoother curves
        envelope = np.ones(total_samples)  # Include padding in envelope
        
        # Attack (fade in) - use smooth linear ramp for volume
        if attack_samples > 0:
            # Use a smooth curve that goes from 0 to 1 (not cosine which affects pitch)
            attack_curve = np.linspace(0, 1, attack_samples)
            # Apply a slight curve to make it smoother (but not cosine)
            attack_curve = np.power(attack_curve, 0.5)  # Square root curve for smooth fade
            envelope[:attack_samples] = attack_curve
        
        # Release (fade out) - use smooth linear ramp for volume
        if release_samples > 0:
            # Use a smooth curve that goes from 1 to 0
            release_curve = np.linspace(1, 0, release_samples)
            # Apply a slight curve to make it smoother
            release_curve = np.power(release_curve, 0.5)  # Square root curve for smooth fade
            envelope[num_samples-release_samples:num_samples] = release_curve
        
        # Ensure the padding is silent
        envelope[num_samples:] = 0
        
        # Extend sine wave with silence for padding
        sine_wave_padded = np.zeros(total_samples)
        sine_wave_padded[:num_samples] = sine_wave
        
        # Apply envelope to sine wave
        audio_data = (sine_wave_padded * envelope * volume).astype(np.float32)
        
        # Ensure audio doesn't clip by normalizing if needed
        max_val = np.max(np.abs(audio_data))
        if max_val > 0.95:
            audio_data = audio_data * (0.95 / max_val)
        
        # Create or reuse audio stream
        if audio_stream is None or audio_stream.closed:
            audio_stream = sd.OutputStream(
                samplerate=sample_rate,
                channels=1,
                dtype=np.float32,
                blocksize=1024
            )
            audio_stream.start()
        
        # Play the audio using the persistent stream
        audio_stream.write(audio_data)
        
    except Exception as e:
        print(f"Error playing sine wave beep: {e}")
        # Reset stream on error
        if audio_stream is not None:
            try:
                audio_stream.close()
            except:
                pass
            audio_stream = None

# Initialize Myo SDK
sdk_path = get_sdk_path()
myo.init(sdk_path=sdk_path)

class SimpleMyoGUI(tk.Tk, myo.DeviceListener):
    def __init__(self, labels=['A', 'B', 'C'], samples_per_class=60, duration_ms=2000):
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
        
        # UI setup
        self._setup_ui()
        self._setup_plots()
        
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
        
        # Setup training UI
        self._setup_training_ui()
        
        # Setup prediction UI
        self._setup_prediction_ui()
        
    def _setup_training_ui(self):
        # Control frame
        control_frame = ttk.Frame(self.train_frame)
        control_frame.pack(fill='x', padx=10, pady=5)
        
        # Status
        self.status_var = tk.StringVar(value="Status: Disconnected")
        status_label = ttk.Label(control_frame, textvariable=self.status_var, font=('Arial', 12), width=25)
        status_label.pack(side='left', padx=5)
        
        # Collection controls frame
        collection_frame = ttk.Frame(control_frame)
        collection_frame.pack(side='left', padx=20)
        
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
        self.class_count_var = tk.StringVar(value="(0/60)")
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
        
        # Save and Train buttons
        save_btn = ttk.Button(control_frame, text="Save Data", command=self.save_data)
        save_btn.pack(side='left', padx=5)
        
        load_btn = ttk.Button(control_frame, text="Load Data", command=self.load_data)
        load_btn.pack(side='left', padx=5)
        
        train_btn = ttk.Button(control_frame, text="Train Model", command=self.train_model)
        train_btn.pack(side='left', padx=5)
        
        stats_btn = ttk.Button(control_frame, text="Show Variations", command=self.show_variation_stats)
        stats_btn.pack(side='left', padx=5)
        
        # Progress
        self.progress_var = tk.StringVar(value="Progress: 0/0")
        progress_label = ttk.Label(control_frame, textvariable=self.progress_var, font=('Arial', 10))
        progress_label.pack(side='right', padx=5)
        
        # Recording indicator and countdown
        self.recording_frame = ttk.Frame(control_frame)
        self.recording_frame.pack(side='right', padx=20)
        
        self.recording_indicator = tk.Label(self.recording_frame, text="⏹", font=('Arial', 16), fg='red')
        self.recording_indicator.pack(side='left', padx=5)
        
        self.countdown_var = tk.StringVar(value="")
        self.countdown_label = tk.Label(self.recording_frame, textvariable=self.countdown_var, font=('Arial', 14, 'bold'))
        self.countdown_label.pack(side='left', padx=5)
        
        # Log frame
        log_frame = ttk.Frame(self.train_frame)
        log_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=8)
        self.log_text.pack(fill='both', expand=True)
        
        # Reset Data button in bottom right corner
        reset_frame = ttk.Frame(self.train_frame)
        reset_frame.pack(fill='x', padx=10, pady=5)
        
        reset_btn = ttk.Button(reset_frame, text="Reset Data", command=self.reset_data, 
                             style='Danger.TButton')
        reset_btn.pack(side='right', padx=5)
        
    def _setup_prediction_ui(self):
        # Control frame for prediction
        pred_control_frame = ttk.Frame(self.pred_frame)
        pred_control_frame.pack(fill='x', padx=10, pady=5)
        
        # Load model button
        load_btn = ttk.Button(pred_control_frame, text="Load Model", command=self.load_model)
        load_btn.pack(side='left', padx=5)
        
        # Clear model button
        clear_model_btn = ttk.Button(pred_control_frame, text="Clear Model", command=self.clear_model)
        clear_model_btn.pack(side='left', padx=5)
        
        # Start/Stop prediction button
        self.pred_btn = ttk.Button(pred_control_frame, text="Start Prediction", command=self.toggle_prediction)
        self.pred_btn.pack(side='left', padx=5)
        
        # Prediction result
        self.pred_result_var = tk.StringVar(value="Prediction: None")
        pred_result_label = ttk.Label(pred_control_frame, textvariable=self.pred_result_var, font=('Arial', 14, 'bold'))
        pred_result_label.pack(side='left', padx=20)
        
        # Confidence
        self.confidence_var = tk.StringVar(value="Confidence: 0%")
        confidence_label = ttk.Label(pred_control_frame, textvariable=self.confidence_var, font=('Arial', 12))
        confidence_label.pack(side='left', padx=10)
        
        # Prediction log
        pred_log_frame = ttk.Frame(self.pred_frame)
        pred_log_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.pred_log_text = scrolledtext.ScrolledText(pred_log_frame, height=6)
        self.pred_log_text.pack(fill='both', expand=True)
        
    def _setup_plots(self):
        # Plot frame
        plot_frame = ttk.Frame(self)
        plot_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Create subplots
        self.fig, self.axs = plt.subplots(2, 1, figsize=(12, 8))
        
        # EMG plot
        self.emg_lines = [self.axs[0].plot([], [], label=f'EMG {i+1}')[0] for i in range(8)]
        self.axs[0].set_ylabel('EMG')
        self.axs[0].legend(loc='upper right', fontsize=6)
        self.axs[0].set_title('EMG Data')
        
        # Quaternion plot
        self.quaternion_lines = [self.axs[1].plot([], [], label=comp)[0] for comp in ['X', 'Y', 'Z', 'W']]
        self.axs[1].set_ylabel('Quaternion')
        self.axs[1].legend(loc='upper right', fontsize=6)
        self.axs[1].set_title('Quaternion Data')
        self.axs[1].set_xlabel('Samples')
        
        self.fig.tight_layout()
        
        # Canvas
        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
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
            # Keep only last window_size samples for prediction
            if len(self.pred_emg_buffer) > self.prediction_window_size:
                self.pred_emg_buffer = self.pred_emg_buffer[-self.prediction_window_size:]
            # Make prediction much less frequently to reduce CPU usage
            if len(self.pred_emg_buffer) % 100 == 0:  # Every 100th sample (much less frequent)
                self.after(50, self.make_prediction)  # Schedule with delay
            
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
        if not self.connected:
            messagebox.showerror("Error", "Not connected to Myo!")
            return
            
        if self.collected[self.class_var.get()] >= self.samples_per_class:
            messagebox.showinfo("Info", f"Already collected {self.samples_per_class} samples for {self.class_var.get()}")
            return
            
        self.collecting = False  # Don't start collecting yet
        self.collection_emg_buffer = []
        self.collection_quaternion_buffer = []
        
        # Show preparation indicator
        self.recording_indicator.config(text="⏸", fg='orange')
        self.status_var.set(f"Get ready for {self.class_var.get()}...")
        self.log(f"Preparing to collect {self.class_var.get()} - recording starts in 1 second")
        
        # Wait 1 second then start recording
        self.after(1000, lambda: self._begin_recording(self.class_var.get()))
        
    def _begin_recording(self, label):
        """Actually start recording data"""
        self.collecting = True
        
        # Show recording indicator
        self.recording_indicator.config(text="⏺", fg='red')
        
        # Change plot background to indicate recording
        self.axs[0].set_facecolor('#fff0f0')  # Light red background
        self.axs[1].set_facecolor('#fff0f0')
        self.canvas.draw_idle()  # Use draw_idle instead of draw()
        
        # Play start recording beep in background
        threading.Thread(target=self._play_start_beep, daemon=True).start()
        
        self.status_var.set(f"Recording {label}... Move your arm!")
        self.log(f"Recording started for {label}")
        
        # Start recording countdown
        self._countdown(self.duration_ms // 1000, label)
        
    def _play_start_beep(self):
        """Play start beep in background thread"""
        try:
            play_square_wave_beep(880, 200, volume=0.1)  # 400Hz, 800ms, very quiet (soothing low tone)
        except:
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
        """Play stop beep in background thread"""
        try:
            play_square_wave_beep(440, 150, volume=0.1)  
            time.sleep(0.02)
            play_square_wave_beep(440, 150, volume=0.1)  
        except:
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
                            
                            # Concatenate EMG and quaternion for sequence input
                            # Shape: (window_size, 12) - 8 EMG channels + 4 quaternion components
                            X_win = np.concatenate([emg_proc, quaternion_win], axis=1)
                            
                            all_windows.append(X_win)
                            all_labels.append(label)
            
            if not all_windows:
                messagebox.showerror("Error", "No valid training sequences could be extracted!")
                return
                
            # Convert to numpy arrays for sequence model
            X = np.array(all_windows)  # Shape: (num_samples, window_size, 12)
            y = np.array(all_labels)
            
            self.log(f"✅ Sequence training data prepared: {X.shape[0]} samples")
            self.log(f"   Input shape: {X.shape} (samples, window_size, features)")
            self.log(f"   Window size: {window_size}")
            self.log(f"   Features per timestep: {X.shape[2]}")
            self.log(f"   Labels: {np.unique(y)}")
            
            # Check class distribution
            unique_labels, counts = np.unique(y, return_counts=True)
            self.log(f"📊 Class distribution:")
            for label, count in zip(unique_labels, counts):
                self.log(f"   {label}: {count} samples ({count/len(y)*100:.1f}%)")
            
            # Check for class imbalance
            min_samples = min(counts)
            max_samples = max(counts)
            imbalance_ratio = max_samples / min_samples if min_samples > 0 else float('inf')
            self.log(f"   Imbalance ratio: {imbalance_ratio:.2f}x (max/min)")
            
            if imbalance_ratio > 2.0:
                self.log(f"⚠️  WARNING: Significant class imbalance detected!")
                self.log(f"   Consider collecting more samples for underrepresented classes")
            
            # Check data quality
            self.log(f"🔍 Data quality check:")
            self.log(f"   EMG range: {np.min(X[:, :, :8]):.2f} to {np.max(X[:, :, :8]):.2f}")
            self.log(f"   Quaternion range: {np.min(X[:, :, 8:]):.2f} to {np.max(X[:, :, 8:]):.2f}")
            self.log(f"   Any NaN values: {np.any(np.isnan(X))}")
            self.log(f"   Any infinite values: {np.any(np.isinf(X))}")
            
            # Train model
            self.log("🧠 Training LSTM sequence model...")
            self.model, self.le = train_model(X, y, self.labels)
            
            self.log("✅ Model training completed!")
            
            # Save model automatically with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            model_filename = f"myo_model_{timestamp}.h5"
            le_filename = f"myo_model_{timestamp}_labels.pkl"
            
            # Save the model
            self.model.save(model_filename)
            
            # Save label encoder
            with open(le_filename, 'wb') as f:
                pickle.dump(self.le, f)
                
            self.log(f"💾 Model saved automatically to: {model_filename}")
            self.log(f"💾 Label encoder saved to: {le_filename}")
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
            self.pred_log("Prediction started")
            self.prediction_window_size = 100  # Match training window size
            self.pred_emg_buffer = []
            self.pred_quaternion_buffer = []
            self.last_prediction_time = 0
        else:
            self.predicting = False
            self.pred_btn.config(text="Start Prediction")
            self.pred_log("Prediction stopped")
            
    def make_prediction(self):
        """Make prediction with current data - optimized for GPU performance"""
        if not hasattr(self, 'model') or self.model is None:
            return
            
        # Rate limiting - predictions every 1 second (faster with GPU)
        now = time.time()
        if now - self.last_prediction_time < 1.0:  # 1 second between predictions
            return
        self.last_prediction_time = now
            
        if len(self.pred_emg_buffer) >= self.prediction_window_size:
            try:
                # Get the latest window of data
                emg_win = np.array(self.pred_emg_buffer[-self.prediction_window_size:])
                quaternion_win = np.array(self.pred_quaternion_buffer[-self.prediction_window_size:]) if len(self.pred_quaternion_buffer) >= self.prediction_window_size else np.zeros((self.prediction_window_size, 4))
                
                # Ensure both have the same length
                min_len = min(len(emg_win), len(quaternion_win))
                emg_win = emg_win[:min_len]
                quaternion_win = quaternion_win[:min_len]
                
                # Debug: log data shapes occasionally
                if len(self.pred_emg_buffer) % 200 == 0:  # Every 200th sample
                    self.pred_log(f"Debug: EMG shape {emg_win.shape}, Quaternion shape {quaternion_win.shape}")
                
                # Preprocess EMG
                emg_proc = preprocess_emg(emg_win)
                
                # Create sequence input for LSTM model
                # Shape: (window_size, 12) - 8 EMG channels + 4 quaternion components
                X_win = np.concatenate([emg_proc, quaternion_win], axis=1)
                
                # Reshape for model input: (1, window_size, 12)
                X_sequence = X_win.reshape(1, X_win.shape[0], X_win.shape[1])
                
                # Debug: log input shape occasionally
                if len(self.pred_emg_buffer) % 200 == 0:
                    self.pred_log(f"Debug: Model input shape {X_sequence.shape}")
                
                # Make prediction with sequence model
                pred = self.model.predict(X_sequence, verbose=0, batch_size=1)
                predicted_class = np.argmax(pred)
                confidence = np.max(pred)
                
                # Debug: log prediction values occasionally
                if len(self.pred_emg_buffer) % 200 == 0:
                    self.pred_log(f"Debug: Raw prediction {pred}, class {predicted_class}, confidence {confidence:.3f}")
                
                # Safety check for label encoder
                if self.le is not None:
                    predicted_label = self.le.inverse_transform([predicted_class])[0]
                else:
                    predicted_label = f"Class_{predicted_class}"
                
                # Update UI for all predictions (lower threshold with GPU)
                if confidence > 0.2:  # Lower threshold for more responsive predictions
                    # Update UI
                    self.pred_result_var.set(f"Prediction: {predicted_label}")
                    self.confidence_var.set(f"Confidence: {confidence:.1%}")
                    
                    # Log predictions more frequently
                    if confidence > 0.5:
                        self.pred_log(f"Predicted: {predicted_label} (confidence: {confidence:.1%})")
                else:
                    # Show low confidence predictions too
                    self.pred_result_var.set(f"Prediction: {predicted_label}")
                    self.confidence_var.set(f"Confidence: {confidence:.1%}")
                    
            except Exception as e:
                # Log prediction errors for debugging
                self.pred_log(f"Prediction error: {e}")
                import traceback
                self.pred_log(f"Traceback: {traceback.format_exc()}")
                
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
                        self.data[label] = list(loaded_data[emg_key])
                        self.quaternion_data[label] = list(loaded_data[quaternion_key])
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

if __name__ == "__main__":
    print("🚀 Starting Simplified Myo Handwriting Recognition GUI")
    print("Using EMG + Quaternion data")
    print("=" * 50)
    
    try:
        app = SimpleMyoGUI(labels=['A', 'B', 'C'], samples_per_class=60, duration_ms=2000)
        print("✅ GUI created successfully")
        app.mainloop()
    except Exception as e:
        print(f"❌ Error starting GUI: {e}")
        import traceback
        traceback.print_exc() 