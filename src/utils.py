# Helper functions (e.g., beeps, file paths)
# src/utils.py
import os

def get_sdk_path():
    """Return the Myo SDK path."""
    # Get the project root directory (parent of src directory)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, 'myo-sdk-win-0.9.0')

def get_output_dir():
    """Return the output directory for data and model."""
    output_dir = r'C:\Users\brian\MyoData'
    os.makedirs(output_dir, exist_ok=True)
    return output_dir