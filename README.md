# Driver Fatigue Detection with Gesture-Based Activation

## 📌 Overview

This project implements a **driver fatigue detection system** using computer vision techniques.  
The system includes a **gesture-based activation mechanism** and detects fatigue using both:

- Classical computer vision methods
- Deep learning (CNN-based eye classifier)

The system simulates an **in-car monitoring scenario** using webcam or recorded video.

---

## 🎯 Features

### 1. Gesture-Based Activation
- The system starts **inactive by default**
- It activates only after a correct sequence of gestures
- Requirements:
  - Multiple gestures
  - Correct order
  - Time constraint

---

### 2. Fatigue Detection

Once activated, the system detects fatigue using:

#### 🟢 Classical Pipeline
- Eye aspect ratio (eye closure)
- Yawning detection
- Head pose estimation

#### 🔵 Deep Learning Pipeline
- CNN-based eye classifier trained on MRL Eye Dataset
- Detects open vs closed eyes

#### 🟣 Hybrid Mode (Recommended)
- Combines classical + DL outputs

---

## 📁 Project Structure

```
.
├── src/
│   ├── main.py
│   ├── fatigue/
│   ├── gestures/
│   └── utils/
├── scripts/
│   └── train_eye_classifier.py
├── data/
│   ├── raw/       # dataset (not versioned)
│   ├── videos/    # user videos (not versioned)
│   └── sample/
├── models/
│   └── eye_classifier.pt
├── tests/
├── README.md
└── requirements.txt
```

---

## ⚙️ Setup

### 1. Clone repository

```bash
git clone <repo-url>
cd Fatigued-Driver-Detector
```

---

### 2. Create environment (Python 3.11 required)

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📦 Dataset (MRL Eye Dataset)

Dataset is **NOT included in the repository**.

### Download automatically

```bash
python src/utils/download_dataset.py
```

### OR download manually:
https://www.kaggle.com/datasets/akashshingha850/mrl-eye-dataset

Extract into:

```
data/raw/
```

---

## 🧠 Train Deep Learning Model

```bash
python scripts/train_eye_classifier.py data/raw
```

This will generate:

```
models/eye_classifier.pt
```

---

## 🚀 Running the System

### 📷 Webcam

```bash
python -m src.main --mode both
```

---

### 🎥 Video input (recommended for demo)

Place your video in:

```
data/videos/
```

Run:

```bash
python -m src.main --video-path data/videos/"VIDEO_NAME".mp4 --mode both
```

---

## ⚙️ Modes

| Mode       | Description |
|-----------|------------|
| classical | Traditional CV only |
| dl        | Deep learning only |
| both      | Combined approach (recommended) |

---

## 🎬 Demo Workflow

1. System starts in **INACTIVE state**
2. Perform gesture sequence → system activates
3. Normal condition → `ALERT`
4. Fatigue simulation → `DROWSY`

---

## 📊 Model Performance

The CNN eye classifier was trained on the MRL Eye Dataset:

- Accuracy: **99.13%**
- Precision: **99.39%**
- Recall: **98.86%**
- F1-score: **99.12%**

---

## 📌 Notes

- Dataset is used only for **training and evaluation**, not runtime
- The system operates in **real-time using video/webcam**
- All testing is performed in a **stationary vehicle simulation**

---

## 🧪 Testing

```bash
python -m pytest
```

---

## 📎 Requirements

- Python 3.11
- OpenCV
- MediaPipe
- PyTorch
- Kaggle API (for dataset download)

---

## 👥 Authors

Group Project - AI: Computer Vision
- Hibba Alkamas
- Sabina Bacaoanu
- Diana Cordovez
- Sofía González
- Cayetana Hinostroza
- Irina Izquierdo
- Gabriela Vega

---

## 🎯 Conclusion

This system demonstrates a realistic approach to driver fatigue detection combining:

- Gesture-based activation
- Classical CV techniques
- Deep learning methods
