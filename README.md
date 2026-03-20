# Driver Drowsiness Detection System

A real-time driver drowsiness detection system built with MediaPipe, OpenCV, and PyTorch. The system detects drowsiness by analyzing facial landmarks in real time and fires an audio alert when the driver appears drowsy.

---

## Demo

| State | Description |
|-------|-------------|
| 🟢 ALERT | Eyes open, driver is attentive |
| 🔴 DROWSY DETECTED | Eyes closing or yawning detected |
| 🚨 WAKE UP! | Alert fires after 20 consecutive drowsy frames |

---

## How It Works

1. **MediaPipe** detects 468 facial landmarks from the webcam feed in real time
2. **EAR (Eye Aspect Ratio)** is computed from 6 eye landmarks — drops when eyes close
3. **MAR (Mouth Aspect Ratio)** is computed from 8 mouth landmarks — rises when yawning
4. A **PyTorch MLP classifier** takes EAR and MAR as input and predicts alert vs drowsy
5. If drowsiness is detected for 20+ consecutive frames, an **audio alert fires** for 15 seconds
6. The alert **stops immediately** when the driver becomes alert again

---

## Results

| Metric | Value |
|--------|-------|
| Dataset size | 1,348 labeled frames |
| Test accuracy | **90.64%** |
| Weighted F1 score | **0.91** |
| Weighted Precision | 0.91 |
| Weighted Recall | 0.91 |
| Inference latency | **0.18ms per frame** |
| Training time | 22.46 seconds (CPU) |
| Real-time FPS | 30+ FPS on CPU |

### Confusion Matrix

|  | Predicted Alert | Predicted Drowsy |
|--|----------------|-----------------|
| **Actual Alert** | 96 | 8 |
| **Actual Drowsy** | 11 | 88 |

---

## Project Structure

```
drowsiness/
    config.py           # all constants and thresholds in one place
    detector.py         # EAR/MAR feature extraction functions
    collect_data.py     # webcam tool to collect and label training data
    train_model.py      # PyTorch MLP training with full metrics
    live_inference.py   # real-time detection using trained model
    README.md
```

---

## Setup

### 1. Create a conda environment

```bash
conda create -n drowsy python=3.10
conda activate drowsy
```

### 2. Install dependencies

```bash
conda install spyder numpy matplotlib
pip install opencv-python mediapipe==0.10.14 torch scikit-learn
```

---

## Usage

### Step 1 — Collect training data

```bash
python collect_data.py
```

- Press `a` to toggle alert recording on/off
- Press `d` to toggle drowsy recording on/off
- Press `q` to quit and save
- Aim for 300+ samples per class

### Step 2 — Train the model

```bash
python train_model.py
```

Prints accuracy, F1, confusion matrix and saves `drowsiness_model.pt` and `metrics.txt`.

### Step 3 — Run live inference

```bash
python live_inference.py
```

---

## Tech Stack

- **Python 3.10**
- **MediaPipe 0.10.14** — facial landmark detection
- **OpenCV** — webcam capture and display
- **PyTorch** — MLP classifier training and inference
- **scikit-learn** — train/val/test split, metrics, normalization

---

## Key Design Decisions

- **Learned classifier over fixed thresholds** — replaced hand-tuned EAR/MAR thresholds with a PyTorch MLP trained on real labeled data, improving accuracy by 15 percentage points
- **StandardScaler normalization** — normalizing EAR and MAR to zero mean and unit variance was the single biggest accuracy improvement
- **Threaded audio alert** — alert sound runs in a background thread so the webcam loop is never blocked
- **Modular codebase** — split into config, detector, data collection, training, and inference so each file has one responsibility

---

## Author

Made as a portfolio project to demonstrate end-to-end ML pipeline development —
from data collection and feature engineering to model training, evaluation, and real-time deployment.
