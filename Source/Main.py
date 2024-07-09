import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import librosa
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.metrics import classification_report
from sklearn.utils import resample
import string
from nltk.stem import PorterStemmer
from sklearn.utils import class_weight
import random
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings("ignore")


#random.seed(42)

# Load the MELD Text dataset
data = pd.read_csv('/Users/sreeram/PycharmProjects/pythonProject/MELD.Raw/Train/Train_csvfile.csv')


# Text preprocessing steps

# 1.Converting text to lower case
def to_lower(text):
    return text.lower()


data['Utterance'] = data['Utterance'].apply(to_lower)
stop_words = set(stopwords.words('english'))


# 2. Removing stopwords
def remove_stopwords(text):
    words = nltk.word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)


data['Utterance'] = data['Utterance'].apply(remove_stopwords)

# 3. Removing punctuations
punctuations_list = string.punctuation


def remove_punctuations(text):
    temp = str.maketrans('', '', punctuations_list)
    return text.translate(temp)


data['Utterance'] = data['Utterance'].apply(lambda x: remove_punctuations(x))

# Perform stemming to remove plural,tenses
stemmer = PorterStemmer()


def perform_stemming(text):
    tokens = nltk.word_tokenize(text)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)


data['Utterance'] = data['Utterance'].apply(lambda x: perform_stemming(x))

# Convert the labels to binary classes
data['Emotion'] = data['Emotion'].replace(
    {'anger': 1, 'disgust': 1, 'fear': 0, 'joy': 0, 'surprise': 0, 'neutral': 0, 'sadness': 0})

# Oversample the minority class (non - hate)
hate_data = data[data['Emotion'] == 1]
non_hate_data = data[data['Emotion'] == 0]
oversampled_hate_data = resample(hate_data, replace=True, n_samples=len(non_hate_data), random_state=42)

# Combine the oversampled non-hate data with the original hate data
oversampled_data = pd.concat([non_hate_data, oversampled_hate_data])

# Define the maximum number of words to keep in the vocabulary
max_words = 5000  # 10000

# Tokenize the text data
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(oversampled_data['Utterance'])
sequences = tokenizer.texts_to_sequences(oversampled_data['Utterance'])

# Pad the sequences to ensure that all sequences have the same length
max_length = 100  # max([len(seq) for seq in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# Get the count of samples in each class
class_counts = oversampled_data['Emotion'].value_counts()

# Print the count of samples in each class
print(class_counts)

# Split data into training and testing sets, ensuring stratification
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, oversampled_data['Emotion'], test_size=0.2,
                                                    stratify=oversampled_data['Emotion'], random_state=42)

# Define the CNN model
text_model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(input_dim=max_words, output_dim=32, input_length=max_length),
    tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
text_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = text_model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
text_model.save('text_model.h5')
# Plot the training and validation accuracy and loss over epochs
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].plot(history.history['accuracy'], label='Training accuracy')
ax[0].plot(history.history['val_accuracy'], label='Validation accuracy')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Accuracy')
ax[0].set_title('Accuracy over epochs')
ax[0].legend()

ax[1].plot(history.history['loss'], label='Training loss')
ax[1].plot(history.history['val_loss'], label='Validation loss')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Loss')
ax[1].set_title('Loss over epochs')
ax[1].legend()

plt.savefig('text.png')
plt.clf()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Plot training accuracy
ax1.plot(history.history['accuracy'], label='Training accuracy')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')
ax1.set_title('Training Accuracy over Epochs')
ax1.legend()

# Plot validation accuracy
ax2.plot(history.history['val_accuracy'], label='Validation accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.set_title('Validation Accuracy over Epochs')
ax2.legend()

# Adjust spacing between subplots
plt.tight_layout()

# Save the figure or display it
plt.savefig('accuracy_plots_text.png')
plt.clf()

# Predict labels for the test set
y_pred_text = text_model.predict(X_test.reshape((*X_test.shape, 1)))
y_pred_text = (y_pred_text > 0.5).astype(int)

# Calculate precision, recall, accuracy, and F1-score
report = classification_report(y_test, y_pred_text)
print(report)

# assuming y_true and y_pred are the true and predicted labels, respectively
accuracy = accuracy_score(y_test, y_pred_text)
precision = precision_score(y_test, y_pred_text)
recall = recall_score(y_test, y_pred_text)
f1score = f1_score(y_test, y_pred_text)

print("Accuracy: {:.2f}".format(accuracy))
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print("F1-score: {:.2f}".format(f1score))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_text)

# Plot the confusion matrix
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.savefig('cm_text.png')
plt.clf()

# Predicted probabilities for the test set
y_prob_text = text_model.predict(X_test.reshape((*X_test.shape, 1)))

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob_text)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.savefig('ROC_text.png')
plt.clf()


# Set the data path for audio file
data_path = '/Users/sreeram/PycharmProjects/pythonProject/MELD.Raw/Train/Train_Data'

# Load the MELD dataset annotations
meld_df = pd.read_csv('/Users/sreeram/PycharmProjects/pythonProject/MELD.Raw/Train/Train_csvfile.csv')

# Parameters for feature extraction
n_mels = 128
n_fft = 2048
hop_length = 512
max_frames = 500


# Function to extract features from audio file
def extract_features(file_path):
    signal, sr = librosa.load(file_path, sr=None)
    stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
    magnitude_spectrogram = np.abs(stft)
    mel_spectrogram = librosa.feature.melspectrogram(S=magnitude_spectrogram, sr=sr, n_mels=n_mels)
    features = np.mean(mel_spectrogram.T, axis=0)
    # print("filepath:", file_path)
    # print("signal shape:", signal.shape)
    # print("sample rate:", sr)
    return features


# Get a list of all the audio files in the data directory
audio_files = []
for root, dirs, files in os.walk(data_path):
    for file in files:
        if file.endswith('.mp4'):
            audio_files.append(os.path.join(root, file))

# Extract features for all the audio files
X = np.array([extract_features(file) for file in audio_files])

# Create label array
y = np.zeros(len(audio_files))
for i, file in enumerate(audio_files):
    file_name = os.path.basename(file)
    utterance_id = int(file_name.split('.')[0].split('_')[-1][3:])
    dialogue_id = int(file_name.split("a")[-1].split("_")[0])
    row = meld_df.loc[(meld_df['Dialogue_ID'] == dialogue_id) & (meld_df['Utterance_ID'] == utterance_id)]
    emotion = row.iloc[0]['Emotion']
    # print(file_name + str(dialogue_id) + '_' + str(utterance_id) + '_' + emotion)

    if 'anger' in emotion or 'disgust' in emotion:
        y[i] = 1

# Randomly oversample the hate class
X_resampled, y_resampled = resample(X[y == 1], y[y == 1],
                                    replace=True, n_samples=X[y == 0].shape[0],
                                    random_state=42)
# Combine the resampled minority class with the majority class
X_resampled = np.concatenate((X[y == 0], X_resampled))
y_resampled = np.concatenate((y[y == 0], y_resampled))


# Normalize features
X_mean = X_resampled.mean(axis=0, keepdims=True)
X_std = X_resampled.std(axis=0, keepdims=True)
X_norm = (X_resampled - X_mean) / X_std
print(X_norm)


# Split the data into training and testing sets using stratified sampling
train_X, test_X, train_y, test_y = train_test_split(X_norm, y_resampled, test_size=0.2, random_state=42,
                                                    stratify=y_resampled)

# Reshape the training data
train_X = np.expand_dims(train_X, axis=-1)
print('Class distribution in the test set:', np.bincount(test_y.astype(int)))

print('train_X.shape = ', train_X.shape)

for i in range(len(test_X)):
    input_text = test_X[i]
    label = test_y[i]
    # print("Input text:", input_text)
    #print("Label:", label)

# Define the CNN model
audio_model = Sequential()
audio_model.add(Conv1D(32, 5, activation='relu', input_shape=(128, 1)))
audio_model.add(MaxPooling1D(2))
audio_model.add(Flatten())
audio_model.add(Dense(128, activation='relu'))
audio_model.add(Dense(1, activation='sigmoid'))
audio_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = audio_model.fit(train_X, train_y, epochs=15, batch_size=16, validation_split=0.1)
audio_model.save('audio_model.h5')
# Plot the training and validation accuracy and loss over epochs
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].plot(history.history['accuracy'], label='Training accuracy')
ax[0].plot(history.history['val_accuracy'], label='Validation accuracy')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Accuracy')
ax[0].set_title('Accuracy over epochs')
ax[0].legend()

ax[1].plot(history.history['loss'], label='Training loss')
ax[1].plot(history.history['val_loss'], label='Validation loss')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Loss')
ax[1].set_title('Loss over epochs')
ax[1].legend()

plt.savefig('audio.png')
plt.clf()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Plot training accuracy
ax1.plot(history.history['accuracy'], label='Training accuracy')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')
ax1.set_title('Training Accuracy over Epochs')
ax1.legend()

# Plot validation accuracy
ax2.plot(history.history['val_accuracy'], label='Validation accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.set_title('Validation Accuracy over Epochs')
ax2.legend()

# Adjust spacing between subplots
plt.tight_layout()

# Save the figure or display it
plt.savefig('accuracy_plots_audio.png')
plt.clf()


# Predict labels for the test set
y_pred_audio = audio_model.predict(test_X.reshape((*test_X.shape, 1)))
y_pred_audio = (y_pred_audio > 0.5).astype(int)

# Calculate precision, recall, accuracy, and F1-score
report = classification_report(test_y, y_pred_audio)
print(report)
# assuming y_true and y_pred are the true and predicted labels, respectively
accuracy = accuracy_score(test_y, y_pred_audio)
precision = precision_score(test_y, y_pred_audio)
recall = recall_score(test_y, y_pred_audio)
f1score = f1_score(test_y, y_pred_audio)

print("Accuracy: {:.2f}".format(accuracy))
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print("F1-score: {:.2f}".format(f1score))

# Confusion matrix
cm = confusion_matrix(test_y, y_pred_audio)

# Plot the confusion matrix
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.savefig('cm_audio.png')
plt.clf()

# Predicted probabilities for the audio set
y_prob_audio = audio_model.predict(test_X.reshape((*test_X.shape, 1)))

# ROC curve
fpr, tpr, thresholds = roc_curve(test_y, y_prob_audio)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.savefig('ROC_audio.png')
plt.clf()

# Define the input layers for the multimodal model
text_input = tf.keras.layers.Input(shape=(max_length,))
audio_input = tf.keras.layers.Input(shape=(128, 1))
# Pass the text input through the CNN model
text_output = text_model(text_input)
# Pass the audio input through the CNN model
audio_output = audio_model(audio_input)
# Concatenate the outputs of the two models
merged = tf.keras.layers.concatenate([text_output, audio_output])

# Add a dense layer to combine the outputs of the two models
merged = tf.keras.layers.Dense(16, activation='relu')(merged)

# Add a dropout layer to prevent over fitting
merged = tf.keras.layers.Dropout(0.5)(merged)
# Add the output layer
merged = tf.keras.layers.Dense(1, activation='sigmoid')(merged)
# Create the multimodal model
model = tf.keras.models.Model(inputs=[text_input, audio_input], outputs=merged)
# Compile the multimodal model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# reshaping
X_train_text_split = X_train[:13248] #13248
y_train_text_split = y_train[:13248]
X_train_audio_split = train_X[:13248]
y_train_audio_split = train_y[:13248]
# Calculate class weights
class_weights = {0: 1, 1: len(y_train_audio_split) / sum(y_train_audio_split)}
# Fit the model with class weights
history = model.fit([X_train_text_split, train_X], y_train_audio_split, epochs=10, batch_size=32, validation_split=0.1)
model.save('my_model.h5')
# Plot the training and validation accuracy and loss over epochs
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].plot(history.history['accuracy'], label='Training accuracy')
ax[0].plot(history.history['val_accuracy'], label='Validation accuracy')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Accuracy')
ax[0].set_title('Accuracy over epochs')
ax[0].legend()

ax[1].plot(history.history['loss'], label='Training loss')
ax[1].plot(history.history['val_loss'], label='Validation loss')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Loss')
ax[1].set_title('Loss over epochs')
ax[1].legend()

plt.savefig('mm.png')
plt.clf()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Plot training accuracy
ax1.plot(history.history['accuracy'], label='Training accuracy')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')
ax1.set_title('Training Accuracy over Epochs')
ax1.legend()

# Plot validation accuracy
ax2.plot(history.history['val_accuracy'], label='Validation accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.set_title('Validation Accuracy over Epochs')
ax2.legend()

# Adjust spacing between subplots
plt.tight_layout()

# Save the figure or display it
plt.savefig('accuracy_plots_multimodal.png')
plt.clf()

# Predict labels for the test set using the multimodal model
print(X_test.shape)
print(test_X.shape)

X_test_reshaped = X_test[:3312]#46

print(X_test.shape)
print(X_test_reshaped.shape)
# Predict the labels using the multimodal model
y_pred = model.predict([X_test_reshaped, test_X])
y_pred = (y_pred > 0.5).astype(int)

print(test_y, y_test, y_pred)
# Print the classification report

report = classification_report(test_y, y_pred)
print(report)

# Confusion matrix
cm = confusion_matrix(test_y, y_pred)

# Plot the confusion matrix
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')

plt.savefig('cm_mm.png')
plt.clf()

# Predicted probabilities for the multimodel set
y_prob = model.predict([X_test_reshaped, test_X])

# ROC curve
fpr, tpr, thresholds = roc_curve(test_y, y_prob)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.savefig('ROC_mm.png')
plt.clf()
# assuming y_true and y_pred are the true and predicted labels, respectively
accuracy = accuracy_score(test_y, y_pred)
precision = precision_score(test_y, y_pred)
recall = recall_score(test_y, y_pred)
f1score = f1_score(test_y, y_pred)

print("Accuracy: {:.2f}".format(accuracy))
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print("F1-score: {:.2f}".format(f1score))
