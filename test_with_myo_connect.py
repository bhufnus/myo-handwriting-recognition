#!/usr/bin/env python3
"""
Test script that works with Myo Connect.
Make sure Myo Connect is running before running this script.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import myo
import time
from src.utils import get_sdk_path

def test_with_myo_connect():
    print("=== Myo Connection Test (with Myo Connect) ===")
    print("IMPORTANT: Make sure Myo Connect is running!")
    print("1. Download Myo Connect from: https://support.getmyo.com/hc/en-us/articles/360018409792")
    print("2. Install and run Myo Connect")
    print("3. Pair your Myo through Myo Connect")
    print("4. Then run this test")
    print()
    
    # Initialize Myo SDK
    try:
        sdk_path = get_sdk_path()
        print(f"SDK path: {sdk_path}")
        myo.init(sdk_path=sdk_path)
        print("✅ Myo SDK initialized successfully")
    except Exception as e:
        print(f"❌ SDK initialization failed: {e}")
        return False
    
    # Create a detailed listener
    class DetailedListener(myo.DeviceListener):
        def __init__(self):
            self.connected = False
            self.emg_count = 0
            self.device_name = None
            self.arm_synced = False
            
        def on_connected(self, event):
            print(f"✅ Myo connected: {event.device_name}")
            self.connected = True
            self.device_name = event.device_name
            event.device.stream_emg(True)
            print("✅ EMG streaming enabled")
            
        def on_disconnected(self, event):
            print("❌ Myo disconnected!")
            self.connected = False
            
        def on_emg(self, event):
            self.emg_count += 1
            if self.emg_count <= 5:
                print(f"EMG sample {self.emg_count}: {event.emg}")
            elif self.emg_count == 6:
                print("... (more EMG samples coming)")
                
        def on_arm_synced(self, event):
            print(f"✅ Arm synced: {event.arm}")
            self.arm_synced = True
            
        def on_arm_unsynced(self, event):
            print("⚠️  Arm unsynced")
            self.arm_synced = False
            
        def on_pose(self, event):
            print(f"Pose detected: {event.pose}")
            
        def on_orientation(self, event):
            if self.emg_count < 3:  # Only print first few
                print(f"Orientation: {event.orientation}")
    
    listener = DetailedListener()
    hub = myo.Hub()
    
    try:
        print("\n🔍 Searching for Myo armband...")
        print("If Myo Connect is running, this should work better.")
        print()
        
        # Try to connect for 10 seconds
        hub.run(listener.on_event, 10000)
        
        if listener.connected:
            print(f"\n✅ Success! Connected to Myo armband: {listener.device_name}")
            print(f"📊 Collected {listener.emg_count} EMG samples")
            print(f"🤚 Arm synced: {listener.arm_synced}")
            
            if listener.emg_count > 0:
                print("🎉 EMG data collection is working!")
                print("You can now run the training script.")
                return True
            else:
                print("⚠️  Connected but no EMG data collected.")
                print("Try:")
                print("   - Moving your arm/fingers")
                print("   - Making sure the armband is snug")
                print("   - Checking the electrode contacts")
                return False
        else:
            print("\n❌ No Myo armband detected!")
            print("Troubleshooting steps:")
            print("1. Make sure Myo Connect is running")
            print("2. Check if Myo appears in Myo Connect")
            print("3. Try restarting Myo Connect")
            print("4. Check if Myo is paired in Myo Connect")
            print("5. Try using a different USB port for Bluetooth dongle")
            return False
            
    except Exception as e:
        print(f"❌ Error during connection: {e}")
        return False
    finally:
        hub.stop()

if __name__ == "__main__":
    success = test_with_myo_connect()
    if success:
        print("\n✅ Test passed! You can run 'python scripts/train.py'")
    else:
        print("\n❌ Test failed! Try using Myo Connect first.") 