# audio-analysis

This repository contains a deep learning model built using TensorFlow 2 for gender recognition based on speaker's audio. It aims to 
classify the gender of a speaker from an audio sample.

For a detailed tutorial on how this project works, please refer to this 
[tutorial](https://www.thepythoncode.com/article/gender-recognition-by-voice-using-tensorflow-in-python).

## Requirements

To run this project, you need the following dependencies:

- TensorFlow 2.x.x
- Scikit-learn
- Numpy
- Pandas
- PyAudio
- Librosa

## Getting Started

You can get started by cloning this repository:
```bash
git clone https://github.com/x4nth055/gender-recognition-by-voice
```

After cloning, install the required libraries using pip:
```bash
pip3 install -r requirements.txt
```

## Dataset
This project uses "Mozilla's Common Voice large dataset for training. The dataset has been preprocessed as follows:

- Invalid samples have been filtered out.
- Only samples labeled in the genre field have been used.
- The dataset has been balanced to ensure an equal number of male and female samples.
- Mel Spectrogram feature extraction technique has been used to create a fixed-length feature vector from each audio sample.
- The data folder contains the preprocessed features in .npy format, not the actual audio samples.

If you want to download and preprocess the dataset yourself, use the ```audio-prep.py``` script. Place it in the root directory of the 
dataset and run it. This will extract features from the audio files and generate new .csv files.

## Training
To train the model, you can customize it in the ```utils.py``` file under the create_model() function. After customization, run:
```bash
python train.py
```
## Testing
Use the ```test.py``` script to test your audio files or your voice for gender recognition:
```bash
python test.py
```

The script will record your voice and make gender predictions.

## Sample Usage
Here's an example of testing a file:
```bash
python test.py --file "test-samples/27-124992-0002.wav"
```

**Output:**
```yaml
Result: male
Probabilities: Male: 96.36%    Female: 3.64%
```

There are some audio samples provided in the test-samples folder for you to test with, taken from the LibriSpeech dataset.

To use your voice, run the test script and follow the prompts:
```bash
python test.py
```

Wait until you see "Please speak" and start talking; it will stop recording when you stop speaking.

Feel free to customize and extend this project for your own voice recognition applications.
