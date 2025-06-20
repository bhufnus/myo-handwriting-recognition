# Myo armband setup and data collection
# src/myo_interface.py
import myo
import os
import numpy as np
import time
import winsound
from .utils import get_sdk_path

# Initialize Myo SDK at module level (like in the notebook)
sdk_path = get_sdk_path()
if not os.path.exists(os.path.join(sdk_path, 'bin', 'myo64.dll')):
    raise FileNotFoundError(f"myo64.dll not found in {sdk_path}\\bin")

try:
    myo.init(sdk_path=sdk_path)
    print("Myo SDK initialized")
except Exception as e:
    raise RuntimeError(f"SDK init failed: {e}")

class EmgListener(myo.DeviceListener):
    def __init__(self, duration=2.0):
        self.duration = duration
        self.emg_data = []
        self.start_time = None
        self.connected = False

    def on_connected(self, event):
        print(f"Myo connected: {event.device_name}")
        event.device.stream_emg(True)
        self.start_time = time.time()
        self.connected = True

    def on_disconnected(self, event):
        print("Myo disconnected!")
        self.start_time = None
        self.connected = False

    def on_emg(self, event):
        if len(event.emg) == 8:
            self.emg_data.append(event.emg)

def init_myo():
    """Initialize Myo SDK - now handled at module level."""
    print("Myo SDK already initialized at module level")
    return True

def collect_data(label, duration_ms=2000):
    """Collect EMG data for a given label."""
    emg_data = []
    listener = EmgListener(duration=duration_ms/1000)
    hub = myo.Hub()
    
    try:
        print(f"\n=== Collecting data for {label} ===")
        print(f"Please write letter {label} on a notepad for 2 seconds...")
        print("Press Enter when ready to start...")
        time.sleep(3)
        input()
        
        for i in range(3, 0, -1):
            print(f"Starting in {i}...")
            try:
                winsound.Beep(800 + 100 * (4-i), 150)  # 800 Hz (3), 900 Hz (2), 1000 Hz (1)
            except Exception as e:
                print(f"Beep failed for {i}: {e}")
            time.sleep(1)
            
        print("Start writing!")
        try:
            winsound.Beep(1000, 200)  # 1000 Hz, 200 ms for start
        except Exception as e:
            print(f"Start beep failed: {e}. Start writing now!")
            
        # Run the hub for the specified duration
        hub.run(listener.on_event, duration_ms)
        
        # Check if we got any data
        if not listener.connected:
            print("⚠️  Warning: Myo armband not connected during data collection")
            print("   Make sure the armband is paired and worn properly")
        
        emg_data = np.array(listener.emg_data)
        print(f"Collected {len(emg_data)} EMG samples")
        
    except Exception as e:
        print(f"Error during data collection: {e}")
    finally:
        hub.stop()
        
    return emg_data, [label] * len(emg_data)