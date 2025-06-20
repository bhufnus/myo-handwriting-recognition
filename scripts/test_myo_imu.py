import myo
import os
import threading

# Path to the SDK directory (where myo64.dll is located in the bin subfolder)
sdk_path = r"C:\Users\brian\__CODING__\MyoArmband\myo-handwriting-recognition\myo-sdk-win-0.9.0"
myo.init(sdk_path=sdk_path)

class Listener(myo.DeviceListener):
    def on_connected(self, event):
        print("Connected")
    def on_imu_data(self, event):
        print("IMU:", event.accelerometer)

class MyoApp:
    def __init__(self):
        self.hub = None
        self.hub_thread = None
        self.connect_var = None
        self.status = None
        self.collecting = False

    def _run_hub(self):
        hub = myo.Hub()
        hub.run_forever(Listener())

    def toggle_connect(self):
        if self.connect_var.get():
            self.status.set("Connecting to Myo...")
            self._set_led("gray")
            if not hasattr(self, 'hub_thread') or not self.hub_thread.is_alive():
                self.hub = myo.Hub()
                self.hub_thread = threading.Thread(target=self._run_hub, daemon=True)
                self.hub_thread.start()
        else:
            self.status.set("Disconnected (manual)")
            self._set_led("gray")
            try:
                if hasattr(self, 'hub') and self.hub:
                    self.hub.stop()
                if hasattr(self, 'hub_thread') and self.hub_thread.is_alive():
                    self.hub_thread.join(timeout=1)
            except Exception as e:
                print(f"Error stopping hub: {e}")
            self.collecting = False 