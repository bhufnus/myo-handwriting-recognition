#!/usr/bin/env python3
"""
Test script to verify IMU data collection works with the updated GUI approach.
This script uses the same patterns as the working test_myo_imu.py but with GUI elements.
"""

import tkinter as tk
import myo
import threading
import time
import numpy as np

# Use the same SDK path as the working test
sdk_path = r"C:\Users\brian\__CODING__\MyoArmband\myo-handwriting-recognition\myo-sdk-win-0.9.0"
myo.init(sdk_path=sdk_path)

class IMUTestApp(tk.Tk, myo.DeviceListener):
    def __init__(self):
        tk.Tk.__init__(self)
        myo.DeviceListener.__init__(self)
        
        self.title("IMU Test - Should Work Like test_myo_imu.py")
        self.geometry("400x300")
        
        # Data buffers
        self.emg_buffer = []
        self.accel_buffer = []
        self.gyro_buffer = []
        self.imu_count = 0
        self.emg_count = 0
        
        # UI
        self.status_label = tk.Label(self, text="Status: Disconnected", font=("Arial", 12))
        self.status_label.pack(pady=10)
        
        self.imu_label = tk.Label(self, text="IMU events: 0", font=("Arial", 10))
        self.imu_label.pack(pady=5)
        
        self.emg_label = tk.Label(self, text="EMG events: 0", font=("Arial", 10))
        self.emg_label.pack(pady=5)
        
        self.accel_label = tk.Label(self, text="Latest Accel: None", font=("Arial", 10))
        self.accel_label.pack(pady=5)
        
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
        # Create hub inside thread like the working test
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
    
    def on_imu(self, event):
        self.imu_count += 1
        accel = [event.accelerometer.x, event.accelerometer.y, event.accelerometer.z]
        gyro = [event.gyroscope.x, event.gyroscope.y, event.gyroscope.z]
        
        self.accel_buffer.append(accel)
        self.gyro_buffer.append(gyro)
        
        self.imu_label.config(text=f"IMU events: {self.imu_count}")
        self.accel_label.config(text=f"Latest Accel: {accel}")
        
        print(f"IMU {self.imu_count}: accel={accel}, gyro={gyro}")

if __name__ == "__main__":
    app = IMUTestApp()
    print("Starting IMU test app...")
    print("This should work like test_myo_imu.py but with a GUI")
    app.mainloop() 