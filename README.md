# Myo Handwriting Recognition

A Python project for recognizing handwritten letters (A, B, C) using the Myo armband's EMG signals.

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Install `pylibmyo` manually (e.g., from GitHub).
3. Ensure Myo SDK is at `C:\Users\brian\__CODING__\MyoArmband\myo-sdk-win-0.9.0`.
4. Run Myo Connect to pair the armband.

## Usage

- Run app: `python scripts/train_gui.py`

## Structure

- `src/`: Core modules (Myo interface, preprocessing, model, GUI, utils).
- `scripts/`: Entry-point scripts for training and prediction.
- `data/`: Saved model and training data (ignored by Git).
- `notebooks/`: Jupyter notebooks for experiments (ignored by Git).
