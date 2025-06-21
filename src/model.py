# Model training and loading
# src/model.py
import numpy as np
import os
import pickle
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from .utils import get_output_dir

def train_model(X, y, labels):
    """Train and save a handwriting recognition model."""
    output_dir = get_output_dir()
    le = LabelEncoder()
    y = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = Sequential([
        Input(shape=(X.shape[1], 1)),
        Conv1D(32, kernel_size=3, activation='relu'),
        Flatten(),
        Dense(16, activation='relu'),
        Dropout(0.3),
        Dense(len(le.classes_), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
    
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