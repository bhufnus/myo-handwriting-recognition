# Myo Handwriting Recognition

A Python project for recognizing handwritten letters (A, B, C) using the Myo armband's EMG signals.

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Install `pylibmyo` manually (e.g., from GitHub).
3. Ensure Myo SDK is at `C:\Users\brian\__CODING__\MyoArmband\myo-sdk-win-0.9.0`.
4. Run Myo Connect to pair the armband.

## Usage

- Run app: `python scripts/main.py`

## Scripts

### Main Application

- **`scripts/main.py`** - Main application with GUI for training and real-time prediction
  - Full-featured GUI with data collection, training, and prediction
  - Supports both Myo device and fallback mode (no device required)
  - Real-time prediction with visual feedback

### Training Scripts

- **`scripts/train.py`** - Command-line training script
  - Train models from saved data without GUI
  - Supports both full model and position-only model

### Prediction Scripts

- **`scripts/predict.py`** - Command-line prediction script
  - Load trained model and make predictions on new data
  - Useful for testing model performance

### Testing Scripts

- **`scripts/test_myo_imu.py`** - Test Myo IMU data collection
  - Verify Myo device connection and IMU data quality
  - Debug orientation and position data

### Test Suite

- **`scripts/run_tests.py`** - Run the complete test suite
  - Executes all unit and integration tests
  - Provides summary report of test results
  - Usage: `python scripts/run_tests.py`

### Diagnostic Scripts

- **`debug_data_format.py`** - Analyze data format and structure
  - Examine training data format and quality
  - Debug data preprocessing issues

## Structure

- `src/`: Core modules (Myo interface, preprocessing, model, GUI, utils).
- `scripts/`: Entry-point scripts for training and prediction.
- `tests/`: Test suite (unit tests, integration tests).
- `data/`: Saved model and training data (ignored by Git).
- `notebooks/`: Jupyter notebooks for experiments (ignored by Git).
- `demo/`: Demo outputs and visualizations.

## Testing

Run the test suite to ensure everything works correctly:

```bash
python scripts/run_tests.py
```

This will run all tests and provide a summary of results.
