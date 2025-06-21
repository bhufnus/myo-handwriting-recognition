#!/usr/bin/env python3
"""
Test script to determine which IMU callback method is actually called by the Myo SDK.
"""

import myo
import time

# Use the same SDK path as the working test
sdk_path = r"C:\Users\brian\__CODING__\MyoArmband\myo-handwriting-recognition\myo-sdk-win-0.9.0"
myo.init(sdk_path=sdk_path)

class CallbackTest(myo.DeviceListener):
    def __init__(self):
        self.on_imu_called = 0
        self.on_imu_data_called = 0
        self.on_accelerometer_data_called = 0
        self.on_gyroscope_data_called = 0
        
    def on_connected(self, event):
        print("Connected to Myo!")
        event.device.stream_emg(True)
    
    def on_disconnected(self, event):
        print("Disconnected from Myo!")
    
    def on_emg(self, event):
        pass  # Just to enable EMG streaming
    
    def on_imu(self, event):
        self.on_imu_called += 1
        print(f"on_imu called! Count: {self.on_imu_called}")
        print(f"  Accelerometer: {event.accelerometer}")
        print(f"  Gyroscope: {event.gyroscope}")
    
    def on_imu_data(self, event):
        self.on_imu_data_called += 1
        print(f"on_imu_data called! Count: {self.on_imu_data_called}")
        print(f"  Accelerometer: {event.accelerometer}")
        print(f"  Gyroscope: {event.gyroscope}")
    
    def on_accelerometer_data(self, event):
        self.on_accelerometer_data_called += 1
        print(f"on_accelerometer_data called! Count: {self.on_accelerometer_data_called}")
        print(f"  Accelerometer: {event.accelerometer}")
    
    def on_gyroscope_data(self, event):
        self.on_gyroscope_data_called += 1
        print(f"on_gyroscope_data called! Count: {self.on_gyroscope_data_called}")
        print(f"  Gyroscope: {event.gyroscope}")

def main():
    print("Starting callback test...")
    print("This will help us determine which IMU callback method is actually called.")
    print("Move your Myo armband around to generate IMU data.")
    print("Press Ctrl+C to stop after a few seconds.\n")
    
    listener = CallbackTest()
    hub = myo.Hub()
    
    try:
        hub.run_forever(listener)
    except KeyboardInterrupt:
        print("\n\nTest completed!")
        print(f"on_imu called: {listener.on_imu_called} times")
        print(f"on_imu_data called: {listener.on_imu_data_called} times")
        print(f"on_accelerometer_data called: {listener.on_accelerometer_data_called} times")
        print(f"on_gyroscope_data called: {listener.on_gyroscope_data_called} times")
        
        if listener.on_imu_called > 0:
            print("\n✅ on_imu is the correct callback!")
        elif listener.on_imu_data_called > 0:
            print("\n✅ on_imu_data is the correct callback!")
        elif listener.on_accelerometer_data_called > 0 or listener.on_gyroscope_data_called > 0:
            print("\n✅ Separate accelerometer/gyroscope callbacks are used!")
        else:
            print("\n❌ No IMU callbacks were called!")

if __name__ == "__main__":
    main() 