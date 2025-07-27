# 🎙️ Real-Time Emotion Detection from Speech using CNN-LSTM (TESS Dataset)

This project detects human emotions from live speech using a deep learning model trained on the [TESS dataset](https://tspace.library.utoronto.ca/handle/1807/24487). It captures audio in real-time via microphone, extracts Mel Frequency Cepstral Coefficients (MFCC) features, and predicts the emotion using a CNN-LSTM model.

---

## 🚀 Features

- 🎤 **Real-Time Audio Input** via microphone
- 🧠 **CNN + LSTM** based deep learning model
- 🎯 **7 Emotion Classes**: `angry`, `disgust`, `fear`, `happy`, `neutral`, `ps`, `sad`
- 📈 Trained on preprocessed **TESS (Toronto Emotional Speech Set)** dataset
- 🔊 MFCC-based feature extraction for robust emotion classification
- 🛠️ Easy to run locally with minimal dependencies

---

## 🧠 Emotion Classes

| Class Index | Emotion  |
|-------------|----------|
| 0           | Angry    |
| 1           | Disgust  |
| 2           | Fear     |
| 3           | Happy    |
| 4           | Neutral  |
| 5           | PS       |
| 6           | Sad      |

> `ps` stands for "pleasant surprise" — a class included in the TESS dataset.

---

## 🗂️ Project Structure

📦 RealTime-Emotion-Detection
├── model/
│ └── tess_emotion_model.h5 # Trained model
├── main.py # Main script to record and predict emotion


---

## 🛠️ Installation

### 🔁 Step 1: Clone this Repository
```bash
git clone https://github.com/yourusername/RealTime-Emotion-Detection.git
cd RealTime-Emotion-Detection
```

### 📦 Step 2: Create a Virtual Environment
python -m venv tess_emotion_env
source tess_emotion_env/bin/activate      # For macOS/Linux
tess_emotion_env\Scripts\activate         # For Windows

### 📥 Step 3: Install Dependencies
- tensorflow
- numpy
- librosa
- sounddevice
- scikit-learn

## 🤖 Future Improvements
🎚️ Add GUI using Streamlit or Tkinter

🧪 Improve model generalization with augmentation

📈 Evaluate with other datasets (e.g., RAVDESS, CREMA-D)

🌐 Deploy as a web app (Flask / FastAPI)


