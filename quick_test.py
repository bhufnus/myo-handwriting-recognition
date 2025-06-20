#!/usr/bin/env python3
"""
Quick test to verify Myo connection before running training.
Run this in your Anaconda prompt to check if Myo is working.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import myo
import time
from src.utils import get_sdk_path

def quick_test():
    print("=== Quick Myo Connection Test ===")
    print("This will test if your Myo armband is connected and collecting data.")
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
    
    # Create a simple listener
    class TestListener(myo.DeviceListener):
        def __init__(self):
            self.connected = False
            self.emg_count = 0
            
        def on_connected(self, event):
            print(f"✅ Myo connected: {event.device_name}")
            event.device.stream_emg(True)
            self.connected = True
            
        def on_disconnected(self, event):
            print("❌ Myo disconnected!")
            self.connected = False
            
        def on_emg(self, event):
            self.emg_count += 1
            if self.emg_count <= 3:  # Only print first 3 samples
                print(f"EMG sample {self.emg_count}: {event.emg}")
    
    listener = TestListener()
    hub = myo.Hub()
    
    try:
        print("\n🔍 Searching for Myo armband...")
        print("Make sure your Myo is:")
        print("- Paired with your computer")
        print("- Worn on your arm")
        print("- Not in sleep mode")
        print()
        
        # Try to connect for 5 seconds
        hub.run(listener.on_event, 5000)
        
        if listener.connected:
            print(f"✅ Myo connected successfully!")
            print(f"📊 Collected {listener.emg_count} EMG samples")
            
            if listener.emg_count > 0:
                print("🎉 EMG data collection is working!")
                print("You can now run the training script.")
                return True
            else:
                print("⚠️  Connected but no EMG data collected.")
                print("Try moving your arm/fingers during the test.")
                return False
        else:
            print("❌ Myo armband not detected!")
            print("Check Bluetooth pairing and try again.")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    finally:
        hub.stop()

if __name__ == "__main__":
    success = quick_test()
    if success:
        print("\n✅ Test passed! You can run 'python scripts/train.py'")
    else:
        print("\n❌ Test failed! Fix the connection issues before training.") 