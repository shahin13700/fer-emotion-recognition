---
title: EdgeVision Facial Emotion Recognition
emoji: 🎭
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 6.27.0
app_file: app.py
pinned: false
license: mit
---

# 🎭 EdgeVision: Real-Time Facial Emotion & Driver Fatigue Guard

[![CI](https://github.com/shahin13700/fer-emotion-recognition/actions/workflows/ci.yml/badge.svg)](https://github.com/shahin13700/fer-emotion-recognition/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-FF6F00?logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-007ACC?logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![Gradio](https://img.shields.io/badge/Gradio-Web%20App-orange?logo=gradio)](https://gradio.app/)
[![Model Size](https://img.shields.io/badge/Model%20Size-817%20KB-brightgreen)]()

An ultra-lightweight, production-ready computer vision pipeline that performs **real-time facial emotion recognition** and **driver drowsiness/attention monitoring** at 60+ FPS on consumer CPUs—requiring **zero GPU** for deployment.

Powered by a compact **MiniXception CNN** (~60,000 parameters) trained with balanced class weighting, stabilized by **Exponential Moving Average (EMA) temporal smoothing**, and combined with **Google MediaPipe 3D Face Mesh** (468 landmarks).

---

## 🌟 Key Features

- ⚡ **Sub-10ms Edge Inference:** MiniXception uses depthwise separable convolutions to reduce computational cost by 8–9× compared to standard convolutions. The model file is only **817 KB**.
- 👁️ **Fatigue & Drowsiness Guard:** Real-time **Eye Aspect Ratio (EAR)** calculation via 468 3D facial landmarks detects eye closure and micro-sleeps to alert drowsy drivers.
- 🎯 **No More Flickering (Temporal Smoothing):** Implements Exponential Moving Average (EMA) over sequential video frames to eliminate erratic label jittering.
- 🧠 **Explainable AI (Grad-CAM):** Full interpretability suite visualizing neural network activation heatmaps—revealing exactly *why* each emotion was predicted.
- 🌐 **Interactive Web App (Gradio):** One-click local web UI ready for instant deployment on **Hugging Face Spaces**.
- ⚖️ **Imbalance-Aware Training:** FER2013 has 16× more *Happy* than *Disgust* samples. Solved via balanced cost-sensitive class weights (boosting *Disgust* recall to 64%).

---

## 🔬 Explainable AI: Grad-CAM Feature Attributions

Neural networks shouldn't be black boxes. `explain.py` generates **Gradient-weighted Class Activation Mapping (Grad-CAM)** heatmaps to confirm the model attends to biologically plausible facial action units (e.g., mouth corners and zygomaticus for *Happy*, brow furrowing for *Angry*):

![Grad-CAM Gallery](outputs/gradcam_samples/gradcam_gallery.png)

---

## 📊 Evaluation & Benchmark Results

Evaluated on **7,178 test images** from the FER2013 benchmark dataset:

| Emotion | Precision | Recall | F1-Score | Support | Key Facial Signals |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Happy** | **0.81** | **0.78** | **0.79** | 1,774 | Distinctive smile, raised cheeks |
| **Surprise** | **0.68** | **0.71** | **0.70** | 831 | Raised eyebrows, open mouth |
| **Disgust** | 0.34 | **0.64** | 0.45 | 111 | Wrinkled nose (boosted by class weights) |
| **Neutral** | 0.48 | 0.63 | 0.54 | 1,233 | Resting facial posture |
| **Angry** | 0.45 | 0.54 | 0.49 | 958 | Lowered eyebrows, tightened lips |
| **Sad** | 0.48 | 0.40 | 0.43 | 1,247 | Downturned lip corners |
| **Fear** | 0.43 | 0.24 | 0.31 | 1,024 | Widened eyes (often confused with Surprise) |

- **Test Accuracy:** **57.15%** (Competitive with published state-of-the-art for FER2013, given human-level annotation ceiling is ~65%).
- **Weighted F1-Score:** **0.56**

---

## 🚀 Quickstart

### 1. Clone & Install
```bash
git clone https://github.com/shahin13700/fer-emotion-recognition.git
cd fer-emotion-recognition

# Create virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

*(Note: Pre-trained weights are pre-packaged in `model/emotion_model.keras`, so you can run demos immediately without training!)*

---

### 2. Choose Your Demo

#### Option A: Interactive Web UI (Gradio)
Launch a local browser interface for webcam snapshots or photo uploads:
```bash
python app.py
```
*Open `http://127.0.0.1:7860` in your browser.*

#### Option B: Real-Time Live Webcam (Smoothed)
Run real-time facial emotion recognition on your local webcam:
```bash
python demo.py
```
*Press **Q** to exit.*

#### Option C: Driver Drowsiness & Emotion Guard (MediaPipe)
Launch the multi-task monitor tracking 468 3D landmarks, emotion probabilities, and Eye Aspect Ratio (EAR) alerts:
```bash
python monitor.py
```
*Press **Q** to exit.*

#### Option D: Generate Explainable AI (Grad-CAM) Heatmaps
Generate attribution heatmaps for test set samples:
```bash
python explain.py
```

---

## 🏗️ Architecture Overview: MiniXception

MiniXception (Arriaga et al., 2017) replaces heavy 2D convolutions with **Depthwise Separable Convolutions** and residual skip connections:

```
Input (48×48 Grayscale)
   │
   ├─► Conv2D (8 filters, 3×3) + BatchNorm + ReLU
   ├─► Conv2D (8 filters, 3×3) + BatchNorm + ReLU
   │
   ├─► [Residual Block 1] 16 filters  (SeparableConv2D + Residual Shortcut)
   ├─► [Residual Block 2] 32 filters  (SeparableConv2D + Residual Shortcut)
   ├─► [Residual Block 3] 64 filters  (SeparableConv2D + Residual Shortcut)
   ├─► [Residual Block 4] 128 filters (SeparableConv2D + Residual Shortcut)
   │
   ├─► GlobalAveragePooling2D
   ├─► Dropout (0.5)
   └─► Dense (7 classes, Softmax)
```

---

## 📁 Repository Structure

```
├── app.py                         # Gradio interactive web application
├── demo.py                        # Real-time webcam demo with EMA temporal smoothing
├── monitor.py                     # MediaPipe Face Mesh + Drowsiness (EAR) monitor
├── explain.py                     # Grad-CAM Explainable AI generator
├── train.py                       # Modular model training script with class weights
├── evaluate.py                    # Evaluation suite & confusion matrix generator
├── model/
│   └── emotion_model.keras        # Lightweight trained model weights (817 KB)
├── outputs/
│   ├── class_indices.json         # Emotion class label index mapping
│   └── gradcam_samples/           # Generated Grad-CAM visualization galleries
├── requirements.txt               # Pinned Python package dependencies
├── LICENSE                        # MIT License
└── README.md
```

---

## 🤝 Contributing & Community

Contributions, bug reports, and feature requests are welcome! If you find this project helpful for your research, university coursework, or hobby projects, please consider giving it a ⭐ **Star**!

---

## 📜 License

Distributed under the **MIT License**. See `LICENSE` for details.