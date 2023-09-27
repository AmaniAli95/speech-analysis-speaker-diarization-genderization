import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from utils import load_data, split_data, create_model

def main():
    X, y = load_data()
    data = split_data(X, y, test_size=0.1, valid_size=0.1)
    model = create_model()
    callbacks = setup_callbacks()
    batch_size = 64
    epochs = 100

    model.fit(
        data["X_train"],
        data["y_train"],
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(data["X_valid"], data["y_valid"]),
        callbacks=callbacks
    )

    model.save("results/model.h5")
    evaluate_model(model, data["X_test"], data["y_test"])

def setup_callbacks():
    tensorboard = TensorBoard(log_dir="logs")
    early_stopping = EarlyStopping(mode="min", patience=5, restore_best_weights=True)
    return [tensorboard, early_stopping]

def evaluate_model(model, X_test, y_test):
    print(f"Evaluating the model using {len(X_test)} samples...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
