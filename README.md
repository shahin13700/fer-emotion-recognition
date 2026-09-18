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

An ultra-lightweight, production-ready computer vision pipeline that performs **real-time facial emotion recognition**, **gamified peak expression tracking**, and **driver drowsiness monitoring** at 60+ FPS on consumer CPUs—requiring **zero GPU** for deployment.

Powered by a compact **MiniXception CNN** (~60,000 parameters) trained with cost-sensitive class weights, stabilized by **Exponential Moving Average (EMA) temporal smoothing**, and integrated with **Google MediaPipe 3D Face Mesh** (468 landmarks).

---

## 🌟 Key Features

- ⚡ **Sub-10ms Edge Inference:** MiniXception uses depthwise separable convolutions to reduce computational cost by 8–9× compared to standard convolutions. The model file is only **817 KB**.
- 📸 **7-Emotion Photo Booth Challenge:** Gamified webcam interface that tracks your peak moments across all 7 emotions, enforces **empirical confidence gates**, and compiles a downloadable **Emotion Photo Strip**.
- 🤝 **Privacy-Preserving Active Learning:** Community data collection engine logging verified faces to append-only `metadata.jsonl` with live milestone tracking toward EdgeVision v2.0 retraining.
- 🛡️ **Guardrailed Fine-Tuning Engine:** Built-in `fine_tune.py` enforcing minimum sample gates ($\ge 10$/class), 15× user data oversampling, heavy augmentation, and automated test set evaluation with rollback protection.
- 👁️ **Fatigue & Drowsiness Guard:** Real-time **Eye Aspect Ratio (EAR)** calculation via 468 3D facial landmarks detects eye closure and micro-sleeps to alert drowsy drivers.
- 🎯 **No More Flickering (Temporal Smoothing):** Implements Exponential Moving Average (EMA) over sequential video frames to eliminate erratic label jittering.
- 🧠 **Explainable AI (Grad-CAM):** Full interpretability suite visualizing neural network activation heatmaps—revealing exactly *why* each emotion was predicted.
- 🌐 **Interactive Web App (Gradio):** Multi-modal web UI with live streaming, video processing, and photo booth, ready for instant deployment on **Hugging Face Spaces**.

---

## 🔬 Empirical Confidence Thresholds

Rather than guessing hardcoded confidence cutoffs for webcam detection, EdgeVision evaluates the **empirical confidence distribution** of correctly classified test samples across all 7,178 images in FER2013. The Photo Booth capture gates are pegged to the **25th percentile ($Q_1$)** of correct predictions:

| Emotion | Support | Test Recall | 25th %ile ($Q_1$) | Median ($Q_2$) | 75th %ile ($Q_3$) | Proposed Capture Gate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Happy** | 1,774 | 74.5% | **71.4%** | 90.1% | 97.3% | **0.70** |
| **Surprise** | 831 | 74.7% | **68.6%** | 84.5% | 94.5% | **0.65** |
| **Neutral** | 1,233 | 61.0% | **46.1%** | 59.5% | 74.7% | **0.46** |
| **Angry** | 958 | 54.4% | **46.1%** | 61.8% | 77.3% | **0.46** |
| **Fear** | 1,024 | 29.7% | **38.6%** | 49.4% | 64.5% | **0.38** |
| **Sad** | 1,247 | 42.1% | **37.5%** | 45.9% | 55.4% | **0.37** |
| **Disgust** | 111 | 61.3% | **70.7%** | 89.0% | 97.7% | **0.50*** |

*\*Note: Disgust webcam gate is set to 0.50 alongside an interactive "Fill Benchmark Face" fallback to balance accessibility with verification integrity.*

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

# Create virtual environment
python -m venv .venv
source .venv/bin/activate       # On Linux/macOS
.\.venv\Scripts\Activate.ps1    # On Windows

# Install dependencies
pip install -r requirements.txt
```

*(Pre-trained weights are pre-packaged in `model/emotion_model.keras`—ready out of the box!)*

---

### 2. Choose Your Demo

#### Option A: Interactive Web UI & Photo Booth (Gradio)
```bash
python app.py
```
*Open `http://127.0.0.1:7860` in your browser.*

#### Option B: Real-Time Live Webcam (Smoothed)
```bash
python demo.py
```
*Press **Q** to exit.*

#### Option C: Driver Drowsiness & Emotion Guard (MediaPipe)
```bash
python monitor.py
```
*Press **Q** to exit.*

#### Option D: Generate Explainable AI (Grad-CAM) Heatmaps
```bash
python explain.py
```

#### Option E: Active Learning Fine-Tuning with Guardrails
```bash
python fine_tune.py
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

## 🎯 Open-Source Roadmap & Advanced Challenges

We welcome external contributors! We have outlined 5 advanced technical challenges ready for collaboration:

1. ⚡ **Issue #1: INT8 Quantization & Sub-3ms Inference:** Convert `emotion_model.keras` to ONNX Runtime and INT8 quantized TFLite; benchmark on Raspberry Pi / edge CPUs.
2. 👁️ **Issue #2: 3D Head Pose & Gaze Vector Estimation:** Use MediaPipe FaceMesh 6-DOF landmarks to filter extreme off-angle faces before emotion inference.
3. ⏱️ **Issue #3: Temporal Micro-Expression Modeling:** Add an optional lightweight GRU or TCN head over 16-frame sliding windows to capture dynamic emotion transitions.
4. 🌐 **Issue #4: In-Browser Client-Side Inference (ONNX.js / WebGL):** Build an edge-only web interface running inference directly inside the user's browser with zero server compute.
5. 🔬 **Issue #5: Vision Transformer (MobileViT) Benchmark:** Benchmark modern edge Vision Transformers against MiniXception for parameter efficiency and attention interpretability.

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines, branch conventions, and environment setup!

---

## 📁 Repository Structure

```
├── app.py                         # Gradio web app (Photo Booth, webcam, video, active learning)
├── fine_tune.py                   # Guardrailed active learning fine-tuning engine
├── demo.py                        # Real-time webcam demo with EMA temporal smoothing
├── monitor.py                     # MediaPipe Face Mesh + Drowsiness (EAR) monitor
├── explain.py                     # Grad-CAM Explainable AI generator
├── train.py                       # Base model training script with class weights
├── evaluate.py                    # Evaluation suite & confusion matrix generator
├── CONTRIBUTING.md                # Open-source contributor guide & roadmap challenges
├── assets/
│   └── samples/                   # Benchmark sample faces for fallback UX
├── model/
│   └── emotion_model.keras        # Lightweight trained model weights (817 KB)
├── outputs/
│   ├── class_indices.json         # Emotion class label index mapping
│   ├── empirical_thresholds.json  # 25th percentile empirical confidence cutoffs
│   └── gradcam_samples/           # Generated Grad-CAM visualization galleries
├── requirements.txt               # Pinned Python package dependencies
├── LICENSE                        # MIT License
└── README.md
```

---

## 📜 License

Distributed under the **MIT License**. See `LICENSE` for details.