#!/usr/bin/env python3
"""
Direct connection test - tries different approaches to connect to Myo
without relying on Windows Bluetooth stack.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import myo
import time
from src.utils import get_sdk_path

def test_direct_connection():
    print("=== Direct Myo Connection Test ===")
    print("This test tries different connection approaches.")
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
    
    # Create a listener that tries multiple connection attempts
    class DirectListener(myo.DeviceListener):
        def __init__(self):
            self.connected = False
            self.emg_count = 0
            self.connection_attempts = 0
            
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
            if self.emg_count <= 3:
                print(f"EMG sample {self.emg_count}: {event.emg}")
            elif self.emg_count == 4:
                print("... (more EMG samples coming)")
                
        def on_arm_synced(self, event):
            print(f"✅ Arm synced: {event.arm}")
            
        def on_arm_unsynced(self, event):
            print("⚠️  Arm unsynced")
            
        def on_pose(self, event):
            print(f"Pose detected: {event.pose}")
    
    # Try different connection approaches
    approaches = [
        ("Standard connection", 5000),
        ("Extended timeout", 15000),
        ("Quick retry", 3000)
    ]
    
    for approach_name, timeout in approaches:
        print(f"\n🔍 Trying: {approach_name} (timeout: {timeout}ms)")
        
        listener = DirectListener()
        hub = myo.Hub()
        
        try:
            hub.run(listener.on_event, timeout)
            
            if listener.connected:
                print(f"✅ Success with {approach_name}!")
                print(f"📊 Collected {listener.emg_count} EMG samples")
                
                if listener.emg_count > 0:
                    print("🎉 EMG data collection is working!")
                    return True
                else:
                    print("⚠️  Connected but no EMG data.")
                    print("Try moving your arm/fingers.")
                    
            else:
                print(f"❌ {approach_name} failed - no device detected")
                
        except Exception as e:
            print(f"❌ Error with {approach_name}: {e}")
        finally:
            hub.stop()
            
        # Small delay between attempts
        time.sleep(1)
    
    print("\n❌ All connection approaches failed!")
    print("\nTroubleshooting suggestions:")
    print("1. Try unplugging and reconnecting your Bluetooth dongle")
    print("2. Disable built-in Bluetooth in Device Manager")
    print("3. Try a different USB port for the Bluetooth dongle")
    print("4. Check if Myo is paired in Windows Bluetooth settings")
    print("5. Try restarting your computer")
    print("6. As a last resort, try Myo Connect app")
    
    return False

if __name__ == "__main__":
    success = test_direct_connection()
    if success:
        print("\n✅ Test passed! You can run 'python scripts/train.py'")
    else:
        print("\n❌ Test failed! Try the troubleshooting steps above.") 