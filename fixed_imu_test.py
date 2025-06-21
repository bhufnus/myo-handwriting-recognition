#!/usr/bin/env python3
"""
Enhanced IMU Test Script - Debugs accelerometer/gyroscope issues
- Uses on_imu, on_orientation, and fallback callbacks
- Prints all event attributes for debugging
- Ensures clean hub shutdown
- Tracks EMG and IMU events
"""

import myo
import os
import sys
import time
import numpy as np

# SDK path
sdk_path = r"C:\Users\brian\__CODING__\MyoArmband\myo-handwriting-recognition\myo-sdk-win-0.9.0"

# Verify SDK path
if not os.path.exists(sdk_path):
    print(f"❌ SDK path not found: {sdk_path}")
    exit(1)

print(f"✅ Using SDK path: {sdk_path}")

# Initialize Myo SDK
try:
    myo.init(sdk_path=sdk_path)
    print("✅ Myo SDK initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize Myo SDK: {e}")
    exit(1)

class FixedIMUListener(myo.DeviceListener):
    def __init__(self):
        self.connected = False
        self.imu_count = 0
        self.emg_count = 0
        self.accel_data = []
        self.gyro_data = []
        self.orientation_data = []
        
    def on_connected(self, event):
        print(f"✅ Connected to Myo: {event.device_name}")
        self.connected = True
        event.device.stream_emg(True)  # Enable EMG for sync
        # Debug device methods
        print(f"Device methods: {dir(event.device)}")
        
    def on_disconnected(self, event):
        print("❌ Disconnected from Myo")
        self.connected = False
        
    def on_emg(self, event):
        self.emg_count += 1
        if self.emg_count % 100 == 0:
            print(f"📊 EMG events: {self.emg_count}")
            
    def on_imu(self, event):
        self.imu_count += 1
        try:
            accel = [getattr(event.accelerometer, attr, event.accelerometer[i]) for i, attr in enumerate(['x', 'y', 'z'])]
            gyro = [getattr(event.gyroscope, attr, event.gyroscope[i]) for i, attr in enumerate(['x', 'y', 'z'])]
            self.accel_data.append(accel)
            self.gyro_data.append(gyro)
            if self.imu_count % 10 == 0:
                print(f"\n📈 IMU Event #{self.imu_count}:")
                print(f"   Accelerometer (g): X={accel[0]:.3f}, Y={accel[1]:.3f}, Z={accel[2]:.3f}")
                print(f"   Gyroscope (deg/s): X={gyro[0]:.3f}, Y={gyro[1]:.3f}, Z={gyro[2]:.3f}")
        except (AttributeError, IndexError) as e:
            print(f"❌ IMU error: {e}")
            print(f"Event attributes: {dir(event)}")
            print(f"Event dict: {event.__dict__}")
            
    def on_orientation(self, event):
        try:
            orientation = [event.orientation.x, event.orientation.y, event.orientation.z, event.orientation.w]
            self.orientation_data.append(orientation)
            if self.imu_count % 10 == 0:  # Sync with IMU print frequency
                print(f"\n📈 Orientation Event:")
                print(f"   Orientation (quat): X={orientation[0]:.3f}, Y={orientation[1]:.3f}, Z={orientation[2]:.3f}, W={orientation[3]:.3f}")
        except AttributeError as e:
            print(f"❌ Orientation error: {e}")
            print(f"Event attributes: {dir(event)}")
            
    # Fallback methods
    def on_imu_data(self, event):
        print(f"🔍 on_imu_data called!")
        self.on_imu(event)
    def on_accelerometer(self, event):
        print(f"🔍 on_accelerometer called!")
        try:
            accel = [event.accelerometer.x, event.accelerometer.y, event.accelerometer.z]
            print(f"Accelerometer: {accel}")
        except AttributeError:
            print(f"Event attributes: {dir(event)}")
    def on_gyroscope(self, event):
        print(f"🔍 on_gyroscope called!")
        try:
            gyro = [event.gyroscope.x, event.gyroscope.y, event.gyroscope.z]
            print(f"Gyroscope: {gyro}")
        except AttributeError:
            print(f"Event attributes: {dir(event)}")

def run_imu_test():
    print("🚀 Starting Enhanced IMU Test")
    print("=" * 50)
    print("This script will:")
    print("1. Connect to your Myo armband")
    print("2. Print raw IMU data (accelerometer, gyroscope, orientation)")
    print("3. Debug callback methods and event attributes")
    print("4. Run for 10 seconds or until Ctrl+C")
    print("=" * 50)
    
    listener = FixedIMUListener()
    hub = myo.Hub()
    
    try:
        print("\n🔗 Connecting to Myo...")
        print("   Make sure Myo Connect is running and your armband is paired!")
        print("   Move your armband around to generate IMU data...")
        print("   Press Ctrl+C to stop\n")
        hub.run(listener.on_event, 10000)  # Run for 10 seconds
    except KeyboardInterrupt:
        print("\n\n📊 Test Results:")
        print("=" * 30)
        print(f"✅ Connected: {listener.connected}")
        print(f"📈 IMU events received: {listener.imu_count}")
        print(f"📊 EMG events received: {listener.emg_count}")
        print(f"📉 Accelerometer samples: {len(listener.accel_data)}")
        print(f"📉 Gyroscope samples: {len(listener.gyro_data)}")
        print(f"📉 Orientation samples: {len(listener.orientation_data)}")
        if listener.imu_count > 0 or len(listener.orientation_data) > 0:
            print("\n🎉 SUCCESS: IMU data is working!")
        else:
            print("\n❌ FAILURE: No IMU data received!")
            print("   Possible issues:")
            print("   - Myo Connect not running")
            print("   - Armband not paired")
            print("   - Firmware issue")
            print("   - Pylibmyo configuration")
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
    finally:
        hub.stop()
        print("\n🏁 Test completed!")

if __name__ == "__main__":
    print("🔍 Environment Check:")
    print(f"   Python: {sys.version}")
    print(f"   Working directory: {os.getcwd()}")
    print(f"   SDK path exists: {os.path.exists(sdk_path)}")
    print(f"   Listener methods: {dir(myo.DeviceListener)}")
    print()
    run_imu_test()