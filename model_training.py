# model_training.py
"""
Pixel Coordinate Prediction using CNN
Author: Rishika Rajora
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os
import matplotlib.pyplot as plt

# -----------------------------
# 1. Dataset Generation (Meshgrid)
# -----------------------------
def generate_dataset(img_size=50):
    """
    Generate images with a single active pixel and normalized coordinates
    """
    images = []
    labels = []

    for y in range(img_size):
        for x in range(img_size):
            img = np.zeros((img_size, img_size), dtype=np.float32)
            img[y, x] = 1.0  # Active pixel
            images.append(img)
            labels.append([x/img_size, y/img_size])  # Normalized coordinates

    images = np.array(images).reshape(-1, img_size, img_size, 1)
    labels = np.array(labels, dtype=np.float32)
    return images, labels

# -----------------------------
# 2. Train-Validation Split
# -----------------------------
def split_dataset(images, labels, train_ratio=0.8):
    num_samples = images.shape[0]
    split_idx = int(num_samples * train_ratio)
    X_train, X_val = images[:split_idx], images[split_idx:]
    y_train, y_val = labels[:split_idx], labels[split_idx:]
    return X_train, X_val, y_train, y_val

# -----------------------------
# 3. CNN Model Definition
# -----------------------------
def create_model(input_shape=(50,50,1)):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(2, activation='sigmoid')  # Normalized coordinates
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# -----------------------------
# 4. Training Function
# -----------------------------
def train_model():
    # Generate dataset
    images, labels = generate_dataset()
    X_train, X_val, y_train, y_val = split_dataset(images, labels)

    # Create model
    model = create_model()

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=32
    )

    # Save model
    os.makedirs("saved_model", exist_ok=True)
    model.save("saved_model/pixel_model.h5")

    # Print final metrics
    val_loss, val_mae = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation MSE: {val_loss:.5f}")
    print(f"Validation MAE: {val_mae:.5f}")

    return model, history, X_val, y_val

# -----------------------------
# 5. Optional: Sample Predictions
# -----------------------------
def sample_predictions(model, X_val, y_val, num_samples=4):
    os.makedirs("images", exist_ok=True)

    plt.figure(figsize=(8,8))
    for i in range(num_samples):
        img = X_val[i].reshape(50,50)
        true_coord = y_val[i]
        pred_coord = model.predict(X_val[i:i+1])[0]

        true_x, true_y = int(true_coord[0]*49), int(true_coord[1]*49)
        pred_x, pred_y = int(pred_coord[0]*49), int(pred_coord[1]*49)

        plt.subplot(2,2,i+1)
        plt.imshow(img, cmap='gray', vmin=0, vmax=1)
        plt.scatter(true_x, true_y, c='lime', s=100, label='Ground Truth')
        plt.scatter(pred_x, pred_y, c='red', s=100, label='Predicted')
        plt.axis('off')
        plt.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig("images/sample_predictions.png")
    plt.close()
    print("Sample predictions saved in images/sample_predictions.png")

# -----------------------------
# 6. Main Execution
# -----------------------------
if __name__ == "__main__":
    model, history, X_val, y_val = train_model()
    sample_predictions(model, X_val, y_val)
    print("Training complete. Model saved in saved_model/pixel_model.h5")
