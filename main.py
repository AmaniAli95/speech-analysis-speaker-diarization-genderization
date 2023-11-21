import os
import librosa
import soundfile as sf
import speech_recognition as sr
import pandas as pd
from google.colab import auth
from google.cloud import storage
import tensorflow.io.gfile as gf
from sklearn.externals import joblib
from sklearn.metrics import f1_score, classification_report, confusion_matrix

# Authenticate with Google Cloud
auth.authenticate_user()
os.environ["GCLOUD_PROJECT"] = "audio-project"  # Replace with your project_id

# Initialize Google Cloud Storage
BUCKET = 'audio-bucket'  # Replace with your bucket_name
gcs = storage.Client()
bucket = gcs.get_bucket(BUCKET)

# Load the Gender and Age Models
gender_model = joblib.load("gender_model.h5")
age_model = joblib.load("age_model.pkl")

def rename_blob(bucket_name, blob_name, new_name):

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    new_blob = bucket.rename_blob(blob, new_name)

def process_audio_blob(blob):
    filename = 'gs://audio-bucket/' + blob.name
    id = blob.name.replace("bucket_par_2/", "")

    with gf.GFile(filename, "rb") as fp:
        # Gender Prediction
        X, sample_rate = librosa.core.load(fp)
        features = extract_feature(X, sample_rate).reshape(1, -1)
        male_prob = gender_model.predict(features)[0][0]
        female_prob = 1 - male_prob
        gender = "male" if male_prob > female_prob else "female"
        prob = f'{male_prob * 100:.2f}%' if gender == 'male' else f'{female_prob * 100:.2f}%'

        # Age Prediction
        # Replace with your code for calculating audio features
        features = features.reshape(1, -1)
        pred_age = age_model.predict(features)[0]

        # Speech Recognition
        sf.write('sound2.wav', X, sample_rate)
        audiowav = 'sound2.wav'
        try:
            r = sr.Recognizer()
            with sr.AudioFile(audiowav) as source:
                audio_data = r.record(source)
                text = r.recognize_google(audio_data, language="ms-MY")
        except:
            text = ''

        # Speaker Diarization
        no_speaker = speaker_diarization(audiowav, n_speakers=0)

        # Print and Store Results
        print(i)
        df_audio = df_audio.append({'id': id, 'Transcribe': text, 'Gender': gender, 'Pr_Gender': prob, 'Age': pred_age, 'No_of_Speaker': no_speaker}, ignore_index=True)

        # Move processed file
        file_pro = filename.replace("gs://audio-bucket/", "")
        new = filename.replace("gs://audio-bucket/bucket_par_2/", "audio-done/")
        rename_blob(bucket_name='audio-bucket', blob_name=file_pro, new_name=new)

def main_processing():
    df_audio = pd.DataFrame(columns=['id', 'Transcribe', 'Gender', 'Pr_Gender', 'Age', 'No_of_Speaker'])

    for i, blob in enumerate(bucket.list_blobs(prefix='bucket_par_2')):
        process_audio_blob(blob)

if __name__ == "__main__":
    main_processing()
