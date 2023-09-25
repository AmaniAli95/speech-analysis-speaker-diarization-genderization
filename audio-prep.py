import glob
import os
import pandas as pd
import numpy as np
import shutil
import librosa
from tqdm import tqdm

def extract_feature(file_name, **kwargs):
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    X, sample_rate = librosa.core.load(file_name)
    stft = np.abs(librosa.stft(X))
    result = np.array([])
    
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        result = np.hstack((result, chroma))
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mel))
    if contrast:
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
        result = np.hstack((result, contrast))
    if tonnetz:
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
        result = np.hstack((result, tonnetz))
    return result

def preprocess_csv(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    new_df = df[df['gender'].isin(['female', 'male'])]
    new_df.to_csv(output_csv, index=False)

def process_audio_files(input_folder, output_folder, feature_extraction_func):
    audio_files = glob.glob(os.path.join(input_folder, '**/*.wav'), recursive=True)
    for audio_file in tqdm(audio_files, desc=f"Processing audio files in {input_folder}"):
        target_path = os.path.join(output_folder, os.path.relpath(audio_file, input_folder))
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        features = feature_extraction_func(audio_file, mel=True)
        target_filename = os.path.splitext(target_path)[0]
        np.save(target_filename, features)

if __name__ == "__main__":
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    csv_files = glob.glob("*.csv")
    for csv_file in csv_files:
        print("[+] Preprocessing", csv_file)
        folder_name, _ = os.path.splitext(csv_file)
        output_csv = os.path.join(data_dir, csv_file)
        preprocess_csv(csv_file, output_csv)
        process_audio_files(folder_name, data_dir, extract_feature)
