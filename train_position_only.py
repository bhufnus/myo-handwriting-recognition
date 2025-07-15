#!/usr/bin/env python3
"""
Position-Only Model Training Script
Trains a model using only quaternion (position) data, ignoring EMG data.
"""

import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import extract_quaternion_only_features

def load_position_data(data_file):
    """Load and prepare position-only data from the data file."""
    print(f"📂 Loading data from: {data_file}")
    
    if not os.path.exists(data_file):
        print(f"❌ Error: Data file {data_file} not found!")
        return None, None, None
    
    # Load data
    data = np.load(data_file)
    
    # Extract quaternion data for each class
    quaternion_samples = []
    labels = []
    
    for key in data.keys():
        if key.endswith('_quaternion'):
            class_name = key.replace('_quaternion', '')
            quaternion_data = data[key]
            
            print(f"📊 Processing {class_name}: {len(quaternion_data)} samples")
            
            # Add each quaternion sample
            for i, quat_sample in enumerate(quaternion_data):
                if isinstance(quat_sample, np.ndarray) and quat_sample.shape == (100, 4):
                    quaternion_samples.append(quat_sample)
                    labels.append(class_name)
                else:
                    print(f"   Skipping sample {i} - invalid shape: {quat_sample.shape if hasattr(quat_sample, 'shape') else 'no shape'}")
    
    print(f"📊 Loaded {len(quaternion_samples)} quaternion samples")
    print(f"🏷️  Classes: {set(labels)}")
    
    return quaternion_samples, labels, data

def prepare_position_sequences(quaternion_samples, labels, window_size=100, overlap=0.5):
    """Prepare position-only sequences for training."""
    print(f"🔄 Preparing position-only sequences (window_size={window_size}, overlap={overlap})")
    
    sequences = []
    sequence_labels = []
    
    for i, quaternion_sample in enumerate(quaternion_samples):
        # Each sample is already a window, so extract features directly
        features = extract_quaternion_only_features(quaternion_sample)
        
        sequences.append(features)
        sequence_labels.append(labels[i])
    
    print(f"✅ Created {len(sequences)} position-only sequences")
    return np.array(sequences), np.array(sequence_labels)

def create_position_model(input_shape, num_classes):
    """Create a position-only LSTM model."""
    print(f"🏗️  Creating position-only model (input_shape={input_shape}, classes={num_classes})")
    
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("📋 Model architecture:")
    model.summary()
    
    return model

def train_position_model(X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    """Train the position-only model."""
    print(f"🚀 Training position-only model (epochs={epochs}, batch_size={batch_size})")
    
    # Create model
    input_shape = (X_train.shape[1],)
    num_classes = len(np.unique(y_train))
    model = create_position_model(input_shape, num_classes)
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    return model, history

def evaluate_position_model(model, X_test, y_test):
    """Evaluate the position-only model."""
    print("📊 Evaluating position-only model...")
    
    # Predict
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    # Calculate accuracy
    accuracy = np.mean(y_pred_classes == y_test_classes)
    print(f"✅ Test Accuracy: {accuracy:.4f}")
    
    return accuracy, y_pred_classes

def save_position_model(model, model_path):
    """Save the position-only model."""
    print(f"💾 Saving position-only model to: {model_path}")
    model.save(model_path)
    print("✅ Model saved successfully!")

def plot_training_history(history, save_path=None):
    """Plot training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"📈 Training history saved to: {save_path}")
    
    plt.show()

def main():
    """Main training function."""
    print("🎯 Position-Only Model Training")
    print("=" * 50)
    
    # Data file path
    data_file = "data/new_fixed_data.npz"
    
    # Load data
    quaternion_data, labels, data = load_position_data(data_file)
    if quaternion_data is None:
        return
    
    # Prepare sequences
    X, y = prepare_position_sequences(quaternion_data, labels)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded)
    
    print(f"🏷️  Encoded classes: {label_encoder.classes_}")
    print(f"📊 Feature shape: {X.shape}")
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_categorical, test_size=0.3, random_state=42, stratify=y_encoded
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    print(f"📊 Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    # Train model
    model, history = train_position_model(X_train, y_train, X_val, y_val)
    
    # Evaluate model
    accuracy, predictions = evaluate_position_model(model, X_test, y_test)
    
    # Save model
    model_path = "models/position_only_model.h5"
    os.makedirs("models", exist_ok=True)
    save_position_model(model, model_path)
    
    # Save label encoder
    import pickle
    encoder_path = "models/position_only_label_encoder.pkl"
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"💾 Label encoder saved to: {encoder_path}")
    
    # Plot training history
    plot_training_history(history, "position_only_training_history.png")
    
    print("\n🎉 Position-only model training complete!")
    print(f"📁 Model saved to: {model_path}")
    print(f"📁 Label encoder saved to: {encoder_path}")

if __name__ == "__main__":
    main() 