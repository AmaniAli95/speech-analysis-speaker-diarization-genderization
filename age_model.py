# age_model.py
import numpy as np
import librosa
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib

def create_age_model():
    # Load and preprocess data
    data = pd.read_csv("features.csv")
    X = data.drop(columns=["age"])
    Y_age = data["age"]

    # Apply oversampling to address class imbalance
    ros = RandomOverSampler()
    X_resampled, y_resampled = ros.fit_resample(X, Y_age)

    # Initialize and train Decision Tree Classifier
    clf_age1 = DecisionTreeClassifier(max_depth=5)
    clf_age1.fit(X_resampled, y_resampled)

    return clf_age1

# Load the pre-trained age model
age_model = create_age_model()

def predict_age(audio_file_path):
    y, sr = librosa.core.load(audio_file_path)
    # Extract audio features
    melspec = librosa.feature.melspectrogram(y=y, sr=sr)
    stft = np.abs(librosa.stft(y))
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y, sr=sr).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
    features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
    features = features.reshape(1, -1)
    pred_age = age_model.predict(features)[0]
    return pred_age
