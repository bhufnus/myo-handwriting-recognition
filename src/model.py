# Model training and loading
# src/model.py
import numpy as np
import os
import pickle
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Input, BatchNormalization, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from .utils import get_output_dir
import glob

# GPU Configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU acceleration enabled: {len(gpus)} GPU(s) available")
    except RuntimeError as e:
        print(f"⚠️ GPU configuration error: {e}")
else:
    print("ℹ️ No GPU detected, using CPU")

def train_model(X, y, labels):
    """Train and save a handwriting recognition sequence model (LSTM) for EMG+quaternion."""
    output_dir = get_output_dir()
    le = LabelEncoder()
    y = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Improved LSTM model for better performance
    model = Sequential([
        Input(shape=(X.shape[1], X.shape[2])),  # (window_size, 12)
        
        # First LSTM layer - increased capacity
        LSTM(128, return_sequences=True, dropout=0.2),
        BatchNormalization(),
        
        # Second LSTM layer - more capacity
        LSTM(64, return_sequences=False, dropout=0.2),
        BatchNormalization(),
        
        # Dense layers with moderate regularization
        Dense(128, activation='relu'),
        Dropout(0.3),
        BatchNormalization(),
        
        Dense(64, activation='relu'),
        Dropout(0.3),
        BatchNormalization(),
        
        Dense(len(le.classes_), activation='softmax')
    ])

    # Optimized optimizer for better training
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,  # Higher learning rate for faster convergence
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', 'sparse_categorical_accuracy']
    )

    # Optimized callbacks for better training
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=15,  # Reduced patience to stop earlier
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,  # Less aggressive reduction
            patience=5,   # More responsive LR reduction
            min_lr=1e-7,
            verbose=1
        )
    ]

    print(f"Training with improved model architecture")
    print(f"Model capacity increased for better performance")

    history = model.fit(
        X_train, y_train,
        epochs=200,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        batch_size=32,  # Larger batch size for better training
        verbose=1
    )

    model_path = os.path.join(output_dir, "handwriting_model.keras")
    model.save(model_path)
    np.save(os.path.join(output_dir, "training_features.npy"), X)
    np.save(os.path.join(output_dir, "training_labels.npy"), np.array(labels))
    print(f"Model and data saved to {output_dir}")
    return model, le

def load_trained_model():
    """Load the most recent trained model and LabelEncoder from the project root directory."""
    # Search for model and label files in the project root directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_files = sorted(glob.glob(os.path.join(base_dir, 'myo_model_*.h5')), reverse=True)
    label_files = sorted(glob.glob(os.path.join(base_dir, 'myo_model_*_labels.pkl')), reverse=True)
    if not model_files or not label_files:
        raise FileNotFoundError("No trained model or label encoder found in the project root directory.")
    model_path = model_files[0]
    le_path = label_files[0]

    from tensorflow.keras.models import load_model
    import pickle

    model = load_model(model_path)
    with open(le_path, "rb") as f:
        le = pickle.load(f)
    print(f"Loaded model: {model_path}")
    print(f"Loaded label encoder: {le_path}")
    return model, le

if __name__ == "__main__":
    app = UnifiedApp(labels=['A', 'B', 'C'], samples_per_class=10, duration_ms=2000)
    app.current_label = tk.StringVar(value=app.labels[0])
    app.mainloop()