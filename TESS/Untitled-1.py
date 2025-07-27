
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
import os
import matplotlib
matplotlib.use('TkAgg')  # Reliable backend for Windows
import matplotlib.pyplot as plt


# %%
# Enable interactive mode
plt.ion()

# Define the path to the TESS dataset
dataset_path = r'C:\Users\manim\OneDrive\Desktop\TESS'

# Define emotions in TESS
emotions = {
    'angry': 'angry',
    'disgust': 'disgust',
    'fear': 'fear',
    'happy': 'happy',
    'neutral': 'neutral',
    'sad': 'sad',
    'ps': 'surprise'
}

# Initialize lists to store features and labels
features = []
labels = []

# Load audio files and extract features
print(f"Checking directory: {dataset_path}")
if not os.path.exists(dataset_path):
    print(f"Error: Directory {dataset_path} does not exist!")
    exit()
else:
    for file in os.listdir(dataset_path):
        if file.endswith('.wav'):
            file_path = os.path.join(dataset_path, file)
            parts = file.split('_')
            if len(parts) > 2:
                emotion_key = parts[2].replace('.wav', '')
                if emotion_key == 'pleasant':
                    emotion_key = 'ps'
                if emotion_key in emotions:
                    emotion = emotions[emotion_key]
                    print(f"Processing file: {file} -> Emotion: {emotion}")
                    y, sr = librosa.load(file_path)
                    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                    mfcc_delta = librosa.feature.delta(mfcc)
                    mfcc_combined = np.concatenate((mfcc, mfcc_delta), axis=0)
                    if mfcc_combined.shape[1] > 100:
                        mfcc_combined = mfcc_combined[:, :100]
                    elif mfcc_combined.shape[1] < 100:
                        mfcc_combined = np.pad(mfcc_combined, ((0, 0), (0, 100 - mfcc_combined.shape[1])), mode='constant')
                    features.append(mfcc_combined)
                    labels.append(emotion)
    print(f"Total files processed: {len(features)}")
    print(f"Labels collected: {labels}")
    if len(features) == 0:
        print("Error: No valid .wav files were processed. Check file naming or directory.")
        exit()

# Convert lists to numpy arrays
features = np.array(features)  # Shape: (n_samples, 26, 100)
labels = np.array(labels)

# Encode labels
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
labels_categorical = to_categorical(labels_encoded)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels_categorical, test_size=0.2, random_state=42)

# Define the CNN model architecture
model = Sequential([
    Conv1D(64, 3, activation='relu', input_shape=(26, 100)),
    MaxPooling1D(2),
    Conv1D(32, 3, activation='relu'),
    MaxPooling1D(2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(emotions), activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Function to predict emotion for a file
def predict_emotion(file_path, model, le):
    y, sr = librosa.load(file_path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_combined = np.concatenate((mfcc, mfcc_delta), axis=0)
    if mfcc_combined.shape[1] > 100:
        mfcc_combined = mfcc_combined[:, :100]
    elif mfcc_combined.shape[1] < 100:
        mfcc_combined = np.pad(mfcc_combined, ((0, 0), (0, 100 - mfcc_combined.shape[1])), mode='constant')
    input_data = np.array([mfcc_combined])
    prediction = model.predict(input_data, verbose=0)
    predicted_emotion_index = np.argmax(prediction)
    predicted_emotion = le.inverse_transform([predicted_emotion_index])[0]
    return predicted_emotion

# Custom callback to track predictions at specific epochs
class PredictionHistory(Callback):
    def __init__(self, example_files, le, interval=10):
        super(PredictionHistory, self).__init__()
        self.example_files = example_files
        self.le = le
        self.interval = interval
        self.predictions = {file: [] for file in example_files}
        self.epochs = []

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.interval == 0 or epoch == 0:  # Every 'interval' epochs or first epoch
            self.epochs.append(epoch + 1)
            for file in self.example_files:
                if os.path.exists(file):
                    pred = predict_emotion(file, self.model, self.le)
                    self.predictions[file].append(pred)

# Example files to track
example_files = [
    r'C:\Users\manim\OneDrive\Desktop\TESS\OAF_back_angry.wav',
    r'C:\Users\manim\OneDrive\Desktop\TESS\YAF_back_happy.wav',
    r'C:\Users\manim\OneDrive\Desktop\TESS\OA_bite_neutral.wav'
    # Add more files if you want
]

# Define callbacks
early_stopping = EarlyStopping(monitor='val_accuracy', patience=20, mode='max', restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)
prediction_history = PredictionHistory(example_files, le, interval=10)

# Train the model
history = model.fit(X_train, y_train, epochs=200, batch_size=16, validation_data=(X_test, y_test), 
                    callbacks=[early_stopping, checkpoint, prediction_history], verbose=1)

# Load the best model
model.load_weights('best_model.h5')

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Final Test Accuracy: {accuracy:.3f}')
print(f'Final Test Loss: {loss:.3f}')

# Training summary
print("\nFinal Training Summary:")
print(f"Total Epochs Trained: {len(history.history['accuracy'])}")
print(f"Best Validation Accuracy: {max(history.history['val_accuracy']):.3f}")
print(f"Best Validation Loss: {min(history.history['val_loss']):.3f}")
print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.3f}")
print(f"Final Training Loss: {history.history['loss'][-1]:.3f}")

# Plot training history (non-blocking)
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show(block=False)  # Non-blocking display
plt.pause(1)  # Pause to ensure plots render

# Prediction summary during training
print("\nPrediction Summary During Training:")
for file in example_files:
    if os.path.exists(file):
        filename = os.path.basename(file)
        true_emotion = file.split('_')[2].replace('.wav', '')
        if true_emotion == 'pleasant':
            true_emotion = 'surprise'
        print(f"\nFile: {filename} | True Emotion: {true_emotion}")
        for epoch, pred in zip(prediction_history.epochs, prediction_history.predictions[file]):
            print(f"  Epoch {epoch}: Predicted Emotion: {pred}")
        final_pred = predict_emotion(file, model, le)
        print(f"  Final Prediction: Predicted Emotion: {final_pred}")
# Keep the script alive to prevent plots from closing
input("Press Enter to exit...")




