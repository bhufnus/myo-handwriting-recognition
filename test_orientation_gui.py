#!/usr/bin/env python3
"""
Test script to verify orientation data works in GUI
Uses the same approach as the updated GUI but simplified for testing.
"""

import tkinter as tk
import myo
import threading
import time
import numpy as np

# Use the same SDK path
sdk_path = r"C:\Users\brian\__CODING__\MyoArmband\myo-handwriting-recognition\myo-sdk-win-0.9.0"
myo.init(sdk_path=sdk_path)

class OrientationTestApp(tk.Tk, myo.DeviceListener):
    def __init__(self):
        tk.Tk.__init__(self)
        myo.DeviceListener.__init__(self)
        
        self.title("Orientation Data Test")
        self.geometry("500x400")
        
        # Data buffers
        self.emg_buffer = []
        self.orientation_buffer = []  # Euler angles
        self.quaternion_buffer = []   # Raw quaternions
        self.imu_count = 0
        self.emg_count = 0
        
        # UI
        self.status_label = tk.Label(self, text="Status: Disconnected", font=("Arial", 12))
        self.status_label.pack(pady=10)
        
        self.imu_label = tk.Label(self, text="Orientation events: 0", font=("Arial", 10))
        self.imu_label.pack(pady=5)
        
        self.emg_label = tk.Label(self, text="EMG events: 0", font=("Arial", 10))
        self.emg_label.pack(pady=5)
        
        self.orientation_label = tk.Label(self, text="Latest Euler: None", font=("Arial", 10))
        self.orientation_label.pack(pady=5)
        
        self.quaternion_label = tk.Label(self, text="Latest Quaternion: None", font=("Arial", 10))
        self.quaternion_label.pack(pady=5)
        
        self.connect_btn = tk.Button(self, text="Connect", command=self.toggle_connect)
        self.connect_btn.pack(pady=10)
        
        self.connected = False
        self.hub_thread = None
        
    def toggle_connect(self):
        if not self.connected:
            self.connect_btn.config(text="Disconnect")
            self.status_label.config(text="Status: Connecting...")
            self.hub_thread = threading.Thread(target=self._run_hub, daemon=True)
            self.hub_thread.start()
        else:
            self.connect_btn.config(text="Connect")
            self.status_label.config(text="Status: Disconnected")
            self.connected = False
    
    def _run_hub(self):
        hub = myo.Hub()
        hub.run_forever(self)
    
    def on_connected(self, event):
        print("Connected to Myo!")
        self.connected = True
        self.status_label.config(text=f"Status: Connected to {event.device_name}")
        event.device.stream_emg(True)
    
    def on_disconnected(self, event):
        print("Disconnected from Myo!")
        self.connected = False
        self.status_label.config(text="Status: Disconnected")
    
    def on_emg(self, event):
        self.emg_count += 1
        self.emg_buffer.append(event.emg)
        self.emg_label.config(text=f"EMG events: {self.emg_count}")
    
    def on_orientation(self, event):
        self.imu_count += 1
        
        # Extract orientation data (quaternions)
        orientation = [event.orientation.x, event.orientation.y, event.orientation.z, event.orientation.w]
        
        # Convert quaternion to Euler angles for easier interpretation
        roll = np.arctan2(2*(orientation[3]*orientation[0] + orientation[1]*orientation[2]), 
                         1 - 2*(orientation[0]*orientation[0] + orientation[1]*orientation[1]))
        pitch = np.arcsin(2*(orientation[3]*orientation[1] - orientation[2]*orientation[0]))
        yaw = np.arctan2(2*(orientation[3]*orientation[2] + orientation[0]*orientation[1]), 
                        1 - 2*(orientation[1]*orientation[1] + orientation[2]*orientation[2]))
        
        # Store as [roll, pitch, yaw] in degrees
        euler_angles = [np.degrees(roll), np.degrees(pitch), np.degrees(yaw)]
        
        self.orientation_buffer.append(euler_angles)
        self.quaternion_buffer.append(orientation)
        
        self.imu_label.config(text=f"Orientation events: {self.imu_count}")
        self.orientation_label.config(text=f"Latest Euler: {[f'{x:.1f}°' for x in euler_angles]}")
        self.quaternion_label.config(text=f"Latest Quaternion: {[f'{x:.3f}' for x in orientation]}")
        
        if self.imu_count % 10 == 0:  # Print every 10th event
            print(f"Orientation {self.imu_count}: Euler={euler_angles}, Quat={orientation}")
    
    def on_imu(self, event):
        # Keep this as a fallback, but orientation should come through on_orientation
        print(f"on_imu called - but orientation should use on_orientation callback")

if __name__ == "__main__":
    app = OrientationTestApp()
    print("Starting orientation test app...")
    print("This should show orientation data from quaternions")
    app.mainloop() 