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
from src.preprocessing import preprocess_emg, extract_all_features
from src.model import train_model
from src.utils import get_sdk_path

# Initialize Myo SDK
sdk_path = get_sdk_path()
myo.init(sdk_path=sdk_path)

class SimpleMyoGUI(tk.Tk, myo.DeviceListener):
    def __init__(self, labels=['A', 'B', 'C'], samples_per_class=10, duration_ms=2000):
        tk.Tk.__init__(self)
        myo.DeviceListener.__init__(self)
        
        self.title("Myo Handwriting Recognition - EMG + Quaternion")
        self.geometry("1200x800")
        
        # Parameters
        self.labels = labels
        self.samples_per_class = samples_per_class
        self.duration_ms = duration_ms
        
        # Data storage
        self.data = {label: [] for label in labels}
        self.quaternion_data = {label: [] for label in labels}
        self.collected = {label: 0 for label in labels}
        
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
        
        # UI setup
        self._setup_ui()
        self._setup_plots()
        
        # Start Myo connection
        self._start_myo()
        
        # Start plot update timer
        self._schedule_plot_update()
        
    def _setup_ui(self):
        # Control frame
        control_frame = ttk.Frame(self)
        control_frame.pack(fill='x', padx=10, pady=5)
        
        # Status
        self.status_var = tk.StringVar(value="Status: Disconnected")
        status_label = ttk.Label(control_frame, textvariable=self.status_var, font=('Arial', 12))
        status_label.pack(side='left', padx=5)
        
        # Collection buttons
        for label in self.labels:
            btn = ttk.Button(control_frame, text=f"Collect {label}", 
                           command=lambda l=label: self._start_collection(l))
            btn.pack(side='left', padx=5)
        
        # Progress
        self.progress_var = tk.StringVar(value="Progress: 0/0")
        progress_label = ttk.Label(control_frame, textvariable=self.progress_var, font=('Arial', 10))
        progress_label.pack(side='right', padx=5)
        
        # Log frame
        log_frame = ttk.Frame(self)
        log_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=8)
        self.log_text.pack(fill='both', expand=True)
        
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
            
    def _start_collection(self, label):
        if not self.connected:
            messagebox.showerror("Error", "Not connected to Myo!")
            return
            
        if self.collected[label] >= self.samples_per_class:
            messagebox.showinfo("Info", f"Already collected {self.samples_per_class} samples for {label}")
            return
            
        self.collecting = True
        self.collection_emg_buffer = []
        self.collection_quaternion_buffer = []
        
        self.status_var.set(f"Collecting {label}... Move your arm!")
        self.log(f"Starting collection for {label}")
        
        # Stop after duration
        self.after(self.duration_ms, lambda: self._finish_collection(label))
        
    def _finish_collection(self, label):
        if not self.collecting:
            return
            
        self.collecting = False
        
        # Store data
        if len(self.collection_emg_buffer) > 0:
            self.data[label].append(np.array(self.collection_emg_buffer))
            self.quaternion_data[label].append(np.array(self.collection_quaternion_buffer))
            self.collected[label] += 1
            
            self.log(f"Collected sample {self.collected[label]}/{self.samples_per_class} for {label}")
            self.log(f"  EMG samples: {len(self.collection_emg_buffer)}")
            self.log(f"  Quaternion samples: {len(self.collection_quaternion_buffer)}")
            
            # Update progress
            total_collected = sum(self.collected.values())
            total_needed = len(self.labels) * self.samples_per_class
            self.progress_var.set(f"Progress: {total_collected}/{total_needed}")
            
            # Update plot
            self._update_plot(label)
            
        self.status_var.set("Status: Ready")
        
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
        
    def log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert('end', f"[{timestamp}] {message}\n")
        self.log_text.see('end')
        
    def save_data(self):
        if not any(self.data.values()):
            messagebox.showerror("Error", "No data to save!")
            return
            
        filename = filedialog.asksaveasfilename(
            defaultextension=".npz",
            filetypes=[("NumPy files", "*.npz"), ("All files", "*.*")]
        )
        
        if filename:
            save_dict = {}
            for label in self.labels:
                if self.data[label]:
                    save_dict[f'{label}_emg'] = np.array(self.data[label], dtype=object)
                    save_dict[f'{label}_quaternion'] = np.array(self.quaternion_data[label], dtype=object)
                    
            np.savez(filename, **save_dict)
            self.log(f"Saved data to {filename}")
            messagebox.showinfo("Success", f"Data saved to {filename}")

    def _schedule_plot_update(self):
        """Schedule periodic plot updates for better performance"""
        self._update_live_plot()
        self.after(50, self._schedule_plot_update)  # Update every 50ms (20 FPS)

if __name__ == "__main__":
    print("🚀 Starting Simplified Myo Handwriting Recognition GUI")
    print("Using EMG + Quaternion data only")
    print("=" * 50)
    
    try:
        app = SimpleMyoGUI(labels=['A', 'B', 'C'], samples_per_class=10, duration_ms=2000)
        print("✅ GUI created successfully")
        app.mainloop()
    except Exception as e:
        print(f"❌ Error starting GUI: {e}")
        import traceback
        traceback.print_exc() 