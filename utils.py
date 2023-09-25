import os
import numpy as np
import pandas as pd
import tqdm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.model_selection import train_test_split

label2int = {"male": 1, "female": 0}

def load_data(vector_length=128):
    """
    Load gender recognition dataset from CSV file and return features and labels.
    Args:
        vector_length (int): The length of feature vectors.
    Returns:
        X (numpy.ndarray): Features (audio data).
        y (numpy.ndarray): Labels (1 for male, 0 for female).
    """

    if not os.path.isdir("results"):
        os.mkdir("results")
    if os.path.isfile("results/features.npy") and os.path.isfile("results/labels.npy"):
        X = np.load("results/features.npy")
        y = np.load("results/labels.npy")
        return X, y
    df = pd.read_csv("balanced-all.csv")

    n_samples = len(df)
    n_male_samples = len(df[df['gender'] == 'male'])
    n_female_samples = len(df[df['gender'] == 'female'])

    X = np.zeros((n_samples, vector_length))
    y = np.zeros((n_samples, 1))

    for i, (filename, gender) in tqdm.tqdm(enumerate(zip(df['filename'], df['gender'])), "Loading data", total=n_samples):
        features = np.load(filename)
        X[i] = features
        y[i] = label2int[gender]

    np.save("results/features", X)
    np.save("results/labels", y)

    return X, y


def split_data(X, y, test_size=0.1, valid_size=0.1):
    """
    Split the dataset into training, validation, and testing sets.
    Args:
        X (numpy.ndarray): Features (audio data).
        y (numpy.ndarray): Labels (1 for male, 0 for female).
        test_size (float): Fraction of data to use for testing.
        valid_size (float): Fraction of data to use for validation.
    Returns:
        data_dict (dict): A dictionary containing the split data.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=7)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=valid_size, random_state=7)

    data_dict = {
        "X_train": X_train,
        "X_valid": X_valid,
        "X_test": X_test,
        "y_train": y_train,
        "y_valid": y_valid,
        "y_test": y_test
    }

    return data_dict


def create_model(vector_length=128):
    """
    Create a neural network model for gender recognition.
    Args:
        vector_length (int): The length of feature vectors.
    Returns:
        model (tf.keras.Model): The neural network model.
    """
    model = Sequential([
        Dense(256, input_shape=(vector_length,), activation="relu"),
        Dropout(0.3),
        Dense(256, activation="relu"),
        Dropout(0.3),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])
    model.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer="adam")
    model.summary()
    return model
