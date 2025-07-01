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

    # Simplified LSTM model optimized for position-focused data
    model = Sequential([
        Input(shape=(X.shape[1], X.shape[2])),  # (window_size, 12)
        
        # First LSTM layer - smaller for position data
        LSTM(64, return_sequences=True, dropout=0.3),
        BatchNormalization(),
        
        # Second LSTM layer - reduced complexity
        LSTM(32, return_sequences=False, dropout=0.3),
        BatchNormalization(),
        
        # Dense layers with stronger regularization
        Dense(64, activation='relu'),
        Dropout(0.4),
        BatchNormalization(),
        
        Dense(32, activation='relu'),
        Dropout(0.4),
        BatchNormalization(),
        
        Dense(len(le.classes_), activation='softmax')
    ])

    # Use a more conservative optimizer for position-focused data
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.0005,  # Lower learning rate
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', 'sparse_categorical_accuracy']
    )

    # More conservative callbacks for position-focused training
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=20,  # More patience
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,  # More aggressive reduction
            patience=8,   # Less patience for LR reduction
            min_lr=1e-6,
            verbose=1
        )
    ]

    # Train with class weights to handle imbalance
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))
    
    print(f"Class weights: {class_weight_dict}")
    print(f"Training with position-focused data (80% position, 20% EMG)")
    print(f"Model architecture optimized for position data")

    history = model.fit(
        X_train, y_train,
        epochs=200,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        batch_size=16,  # Smaller batch size for better generalization
        class_weight=class_weight_dict,
        verbose=1
    )

    model_path = os.path.join(output_dir, "handwriting_model.keras")
    model.save(model_path)
    np.save(os.path.join(output_dir, "training_features.npy"), X)
    np.save(os.path.join(output_dir, "training_labels.npy"), np.array(labels))
    print(f"Model and data saved to {output_dir}")
    return model, le

def load_trained_model():
    """Load the trained model and LabelEncoder."""
    output_dir = get_output_dir()
    model_path = os.path.join(output_dir, "handwriting_model.keras")
    labels_path = os.path.join(output_dir, "training_labels.npy")
    
    try:
        model = load_model(model_path)
        print(f"Model loaded from {model_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")
    
    try:
        saved_labels = np.load(labels_path)
        le = LabelEncoder()
        le.fit(saved_labels)
        print(f"LabelEncoder initialized with classes: {le.classes_}")
    except Exception as e:
        raise RuntimeError(f"Error loading labels: {e}")
    
    return model, le

if __name__ == "__main__":
    app = UnifiedApp(labels=['A', 'B', 'C'], samples_per_class=10, duration_ms=2000)
    app.current_label = tk.StringVar(value=app.labels[0])
    app.mainloop()