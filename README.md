# Speech Analysis Speaker Diarization and Genderization

## Overview
This repository contains a deep learning model built using TensorFlow 2 for gender recognition based on the speaker's audio. The project's primary goal is to classify the gender of a speaker from an audio sample.

## Requirements
To run this project successfully, you need to have the following dependencies installed:
- TensorFlow 2.x.x
- Scikit-learn
- Numpy
- Pandas
- PyAudio
- Librosa

You can install these dependencies using pip:
```bash
pip3 install -r requirements.txt
```

## Getting Started

To get started, clone this repository to your local machine:
```bash
git clone https://github.com/speech-analysis-speaker-diarization-genderization
```
After cloning, navigate to the project directory and install the required libraries as mentioned above.

## File Structure
Here is a brief overview of the project's file structure:
```bash
speech-analysis-speaker-diarization-genderization/
│
├── data/
│   ├── ...
│   └── (Preprocessed audio features in .npy format)
│
├── test-samples/
│   ├── ...
│   └── (Sample audio files for testing)
│
├── utils.py
├── train.py
├── test.py
├── audio-prep.py
├── requirements.txt
└── README.md
```

## Dataset
This project utilizes Mozilla's Common Voice large dataset for training. The dataset has undergone preprocessing steps as follows:

- Invalid samples have been filtered out.
- Only samples labeled in the genre field have been used.
- The dataset has been balanced to ensure an equal number of male and female samples.
- Mel Spectrogram feature extraction technique has been used to create a fixed-length feature vector from each audio sample.
  
**Note:** The data folder contains the preprocessed features in .npy format, not the actual audio samples. If you wish to download and preprocess the dataset yourself, utilize the `audio-prep.py` script, place it in the root directory of the dataset, and execute it. This will extract features from the audio files and generate new .csv files.

## Training
For training the model, you can customize it in the `utils.py` file under the `create_model()` function. After customization, run the training script:
```bash
python train.py
```

## Testing
o test the gender recognition model, utilize the `test.py` script, which can be used to test audio files or your own voice:
```bash
python test.py
```

The script will record your voice and make gender predictions. For testing specific files, you can use:
```bash
python test.py --file "test-samples/27-124992-0002.wav"
```

## Sample Usage
Here's an example of testing a file:
```yaml
Result: male
Probabilities: Male: 96.36%    Female: 3.64%
```
There are some audio samples provided in the `test-samples` folder for you to test with, taken from the LibriSpeech dataset. When using your voice, run the test script and follow the prompts. It will stop recording when you stop speaking.
