# Position-Only Model for Myo Handwriting Recognition
# src/position_only_model.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os
import pickle
import time

def create_position_only_model(input_shape, num_classes):
    """Create a position-only LSTM model using only quaternion data."""
    
    model = Sequential([
        Input(shape=input_shape),  # (window_size, 4) for quaternions only
        
        # First LSTM layer - process quaternion sequences
        LSTM(64, return_sequences=True, dropout=0.2),
        BatchNormalization(),
        
        # Second LSTM layer - extract temporal patterns
        LSTM(32, return_sequences=False, dropout=0.2),
        BatchNormalization(),
        
        # Dense layers for classification
        Dense(64, activation='relu'),
        Dropout(0.3),
        BatchNormalization(),
        
        Dense(32, activation='relu'),
        Dropout(0.3),
        BatchNormalization(),
        
        Dense(num_classes, activation='softmax')
    ])
    
    return model

def preprocess_quaternion_only(quaternion_data):
    """Preprocess quaternion data for position-only model."""
    if quaternion_data is None or not hasattr(quaternion_data, '__len__') or len(quaternion_data) == 0:
        return np.zeros((100, 4))
    
    # Convert to numpy array
    quaternion_data = np.array(quaternion_data)
    
    # Check if data has correct shape (should be 2D with 4 columns)
    if len(quaternion_data.shape) != 2:
        print(f"Warning: Quaternion data has wrong shape {quaternion_data.shape}, using zeros")
        return np.zeros((100, 4))
    
    if quaternion_data.shape[1] != 4:
        print(f"Warning: Quaternion data has wrong number of columns {quaternion_data.shape[1]}, using zeros")
        return np.zeros((100, 4))
    
    # Normalize quaternions to unit length
    norms = np.linalg.norm(quaternion_data, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)  # Avoid division by zero
    quaternion_normalized = quaternion_data / norms
    
    # Remove DC offset (center around zero)
    quaternion_centered = quaternion_normalized - np.mean(quaternion_normalized, axis=0)
    
    # Resize to target length (100 samples)
    if len(quaternion_centered) != 100:
        # Use linear interpolation to resize
        old_indices = np.linspace(0, len(quaternion_centered) - 1, len(quaternion_centered))
        new_indices = np.linspace(0, len(quaternion_centered) - 1, 100)
        
        resized_data = np.zeros((100, 4))
        for col in range(4):
            resized_data[:, col] = np.interp(new_indices, old_indices, quaternion_centered[:, col])
        
        return resized_data
    
    return quaternion_centered

def train_position_only_model(quaternion_samples, labels, labels_list):
    """Train a position-only model using only quaternion data."""
    
    print("🧠 Training Position-Only Model...")
    print(f"Using {len(quaternion_samples)} samples")
    print(f"Classes: {set(labels)}")
    
    # Preprocess all quaternion samples
    processed_samples = []
    for sample in quaternion_samples:
        processed = preprocess_quaternion_only(sample)
        processed_samples.append(processed)
    
    # Convert to numpy arrays
    X = np.array(processed_samples, dtype=np.float32)  # Shape: (num_samples, 100, 4)
    y = np.array(labels)
    
    print(f"Input shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    # Label encoding
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"Train samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Create model
    model = create_position_only_model(
        input_shape=(X.shape[1], X.shape[2]),  # (100, 4)
        num_classes=len(le.classes_)
    )
    
    # Compile model
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', 'sparse_categorical_accuracy']
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train model
    print("Training position-only model...")
    history = model.fit(
        X_train, y_train,
        epochs=200,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        batch_size=32,
        verbose=1
    )
    
    # Save model
    timestamp = int(time.time())
    model_filename = f"position_only_model_{timestamp}.h5"
    le_filename = f"position_only_model_{timestamp}_labels.pkl"
    
    model.save(model_filename)
    with open(le_filename, 'wb') as f:
        pickle.dump(le, f)
    
    print(f"✅ Position-only model saved:")
    print(f"   Model: {model_filename}")
    print(f"   Labels: {le_filename}")
    
    return model, le, history

def load_position_only_model():
    """Load the most recent position-only model."""
    import glob
    
    # Search for position-only model files
    model_files = sorted(glob.glob('position_only_model_*.h5'), reverse=True)
    label_files = sorted(glob.glob('position_only_model_*_labels.pkl'), reverse=True)
    
    if not model_files or not label_files:
        raise FileNotFoundError("No position-only model found")
    
    model_path = model_files[0]
    le_path = label_files[0]
    
    model = tf.keras.models.load_model(model_path)
    with open(le_path, 'rb') as f:
        le = pickle.load(f)
    
    print(f"Loaded position-only model: {model_path}")
    print(f"Classes: {list(le.classes_)}")
    
    return model, le

def predict_position_only(quaternion_data, model, le):
    """Make prediction using position-only model."""
    
    # Preprocess quaternion data
    processed_data = preprocess_quaternion_only(quaternion_data)
    
    # Add batch dimension
    X_input = processed_data[np.newaxis, ...]  # Shape: (1, 100, 4)
    
    # Make prediction
    prediction = model.predict(X_input, verbose=0)
    print("DEBUG: prediction shape:", getattr(prediction, 'shape', None), "prediction:", prediction)
    # Ensure prediction is always 2D
    if np.isscalar(prediction):
        prediction = np.array([[prediction]])
    elif prediction.ndim == 1:
        prediction = prediction[np.newaxis, :]
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    predicted_label = le.inverse_transform([predicted_class])[0]
    return predicted_label, confidence, prediction[0]

# Test function
if __name__ == "__main__":
    # Example usage
    print("Position-Only Model Test")
    print("=" * 30)
    
    # Create dummy data for testing
    dummy_quaternions = [np.random.randn(100, 4) for _ in range(10)]
    dummy_labels = ['A', 'B', 'C', 'IDLE', 'NOISE'] * 2
    
    try:
        model, le, history = train_position_only_model(dummy_quaternions, dummy_labels, ['A', 'B', 'C', 'IDLE', 'NOISE'])
        print("✅ Position-only model training successful!")
        
        # Test prediction
        test_quaternion = np.random.randn(100, 4)
        predicted_label, confidence, probabilities = predict_position_only(test_quaternion, model, le)
        print(f"Test prediction: {predicted_label} (confidence: {confidence:.3f})")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc() 