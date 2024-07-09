import numpy as np
import pandas as pd
import librosa
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import warnings
warnings.filterwarnings("ignore")

# Load the saved model
model = load_model('/Users/sreeram/PycharmProjects/pythonProject/my_model.h5')#Give datapath to trained model

# Prediction on new_text and new_audio as the new input data
# Preprocess the text data
new_text = 'Oh no-no-no, give me some specifics.'
new_audio = '/Users/sreeram/PycharmProjects/pythonProject/MELD.Raw/HateFile.mp4'# Give datapath to HateFile in the Test folder
tokenizer = Tokenizer(num_words=5000)
new_text_sequence = tokenizer.texts_to_sequences(new_text)
new_text_padded_sequence = pad_sequences(new_text_sequence, maxlen=100, padding='post', truncating='post')

new_text_sequence = tokenizer.texts_to_sequences([new_text])
new_text_padded_sequence = pad_sequences(new_text_sequence, maxlen=100, padding='post', truncating='post')

def extract_features(file_path):
    signal, sr = librosa.load(file_path, sr=None)
    stft = librosa.stft(signal, n_fft=2408, hop_length=512)
    magnitude_spectrogram = np.abs(stft)
    mel_spectrogram = librosa.feature.melspectrogram(S=magnitude_spectrogram, sr=sr, n_mels=128)
    features = np.mean(mel_spectrogram.T, axis=0)
    return features
# Set the data path
data_path = '/Users/sreeram/PycharmProjects/pythonProject/MELD.Raw/Train/Train_Data'  #Give datapath to training data in Train folder for normalisation
# Get a list of all the audio files in the data directory
audio_files = []
for root, dirs, files in os.walk(data_path):
    for file in files:
        if file.endswith('.mp4'):
            audio_files.append(os.path.join(root, file))

# Extract features for all the audio files
X = np.array([extract_features(file) for file in audio_files])

# Normalize features
X_mean = X.mean(axis=0, keepdims=True)
X_std = X.std(axis=0, keepdims=True)
X_norm = (X - X_mean) / X_std


# Preprocess the audio data
new_audio_features = extract_features(new_audio)
new_audio_norm = (new_audio_features - X_mean) / X_std
new_audio_norm_reshaped = new_audio_norm.reshape((1, 128, 1))

# Get the predicted label
predicted_label = model.predict([new_text_padded_sequence, new_audio_norm_reshaped])

if predicted_label > 0.5:
    print("The model classified the HateFile  as hate.")
else:
    print("The model classified the HateFile as non-hate.")


# Prediction on new_text and new_audio as the new input data
# Preprocess the text data

# Prediction on new_text and new_audio as the new input data
# Preprocess the text data
new_text = 'Good to know'
new_audio = '/Users/sreeram/PycharmProjects/pythonProject/MELD.Raw/Non_HateFile.mp4' # Give datapath to Non_HateFile in Test Folder
new_text_sequence = tokenizer.texts_to_sequences([new_text])
new_text_padded_sequence = pad_sequences(new_text_sequence, maxlen=100, padding='post', truncating='post')

# Preprocess the audio data
new_audio_features = extract_features(new_audio)
new_audio_norm = (new_audio_features - X_mean) / X_std
new_audio_norm_reshaped = new_audio_norm.reshape((1, 128, 1))

# Get the predicted label
predicted_label = model.predict([new_text_padded_sequence, new_audio_norm_reshaped])

if predicted_label > 0.5:
    print("The model classified the Non-HateFile as hate.")
else:
    print("The model classified the Non-HateFile as non-hate.")




