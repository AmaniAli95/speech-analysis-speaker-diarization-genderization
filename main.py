from utils import create_model
from test import record_audio_and_save, extract_audio_features

if __name__ == "__main__":
    file_path = "test.wav"
    record_audio_and_save(file_path)
    gender_features = extract_audio_features(file_path)
    model = create_model()
    model.load_weights("results/model.h5")
    male_prob = model.predict(gender_features.reshape(1, -1))[0][0]
    female_prob = 1 - male_prob
    gender = "male" if male_prob > female_prob else "female"
    
    print("Result:", gender)
    print(f"Probabilities: Male: {male_prob * 100:.2f}% Female: {female_prob * 100:.2f}%")
