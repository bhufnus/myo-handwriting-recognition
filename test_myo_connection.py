#!/usr/bin/env python3
"""
Simple test script to check Myo armband connection and EMG data collection.
Run this in your Anaconda prompt to troubleshoot Myo connectivity.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import myo
import time
from src.utils import get_sdk_path

class TestListener(myo.DeviceListener):
    def __init__(self):
        self.connected = False
        self.emg_count = 0
        
    def on_connected(self, event):
        print(f"✅ Myo connected: {event.device_name}")
        self.connected = True
        event.device.stream_emg(True)
        print("✅ EMG streaming enabled")
        
    def on_disconnected(self, event):
        print("❌ Myo disconnected!")
        self.connected = False
        
    def on_emg(self, event):
        self.emg_count += 1
        if self.emg_count <= 5:  # Only print first 5 samples
            print(f"EMG sample {self.emg_count}: {event.emg}")
        elif self.emg_count == 6:
            print("... (more EMG samples coming)")
            
    def on_pose(self, event):
        print(f"Pose detected: {event.pose}")
        
    def on_arm_synced(self, event):
        print(f"Arm synced: {event.arm}")
        
    def on_arm_unsynced(self, event):
        print("Arm unsynced")

def main():
    print("=== Myo Connection Test ===")
    print("Make sure your Myo armband is:")
    print("1. Charged")
    print("2. Paired with your computer via Bluetooth")
    print("3. Worn on your arm")
    print("4. Not in sleep mode (tap it to wake up)")
    print()
    
    # Initialize SDK
    try:
        sdk_path = get_sdk_path()
        print(f"SDK path: {sdk_path}")
        myo.init(sdk_path=sdk_path)
        print("✅ Myo SDK initialized successfully")
    except Exception as e:
        print(f"❌ SDK initialization failed: {e}")
        return
    
    # Create listener and hub
    listener = TestListener()
    hub = myo.Hub()
    
    print("\n🔍 Searching for Myo armband...")
    print("If no device is found, check:")
    print("- Bluetooth is enabled")
    print("- Myo is paired")
    print("- Myo is not connected to another device")
    print()
    
    try:
        # Try to connect for 10 seconds
        hub.run(listener.on_event, 10000)
        
        if listener.connected:
            print(f"\n✅ Success! Connected to Myo armband")
            print(f"📊 Collected {listener.emg_count} EMG samples")
            
            if listener.emg_count > 0:
                print("🎉 EMG data collection is working!")
            else:
                print("⚠️  No EMG data collected. Try:")
                print("   - Moving your arm/fingers")
                print("   - Making sure the armband is snug")
                print("   - Checking the electrode contacts")
        else:
            print("\n❌ No Myo armband detected!")
            print("Troubleshooting steps:")
            print("1. Make sure Myo is paired in Windows Bluetooth settings")
            print("2. Try unpairing and re-pairing the Myo")
            print("3. Check if Myo appears in Device Manager")
            print("4. Try restarting the Myo armband")
            
    except Exception as e:
        print(f"❌ Error during connection: {e}")
    finally:
        hub.stop()
        print("\nTest completed.")

if __name__ == "__main__":
    main() 