# 🚀 EdgeVision: Open Source Launch & Community Playbook

This playbook contains complete, copy-paste-ready distribution posts for Reddit, Hacker News, X (Twitter), and LinkedIn to maximize visibility and GitHub stars for **EdgeVision**.

---

## 📌 1. Hugging Face Spaces Deployment (1-Minute Setup)

Deploying a live interactive demo is the single highest-converting action for GitHub stars.

1. Go to [Hugging Face Spaces](https://huggingface.co/new-space).
2. Set **Space Name**: `edgevision-fer` (or `edgevision`).
3. Select **Space SDK**: `Gradio`.
4. License: `MIT`, Hardware: `Free CPU (basic)`.
5. In your local terminal, link and push the repository:
   ```bash
   git remote add space https://huggingface.co/spaces/<YOUR_HF_USERNAME>/edgevision-fer
   git push space main
   ```
6. Once deployed, add your Space URL to GitHub:
   - In GitHub repo page: click the ⚙️ icon next to "About" -> check **Use your GitHub Pages website or link a website** -> enter `https://huggingface.co/spaces/<YOUR_HF_USERNAME>/edgevision-fer`.

---

## 💬 2. Reddit Playbook

### A. r/Python
* **Title:** `I built a real-time, CPU-only emotion recognition photo booth and fatigue monitor in Python (817 KB model, zero GPU required)`
* **Flair:** `Project` or `Showcase`
* **Body:**
```markdown
Hey everyone! 👋

I wanted to share **EdgeVision**, an open-source computer vision pipeline I built to run real-time facial emotion recognition and driver fatigue monitoring entirely on everyday laptop CPUs.

Most facial emotion models rely on heavy ResNet or VGG backbones that chew through battery and struggle without a discrete GPU. I wanted to see how far a lightweight architecture could be pushed for real-time edge use.

### ⚡ Key Engineering Details:
* **817 KB MiniXception Architecture:** Depthwise separable convolutions keep total parameters down to just 51,255.
* **~2 ms CPU Inference:** The forward pass is graph-compiled (`tf.function`), costing ~2 ms per face crop and ~5 ms for a full 640×480 frame on a standard Ryzen/Intel CPU.
* **Temporal Smoothing:** Consecutive frames use Exponential Moving Average (EMA) smoothing to eliminate label jitter and flicker.
* **Driver Drowsiness Guard:** Uses Google's MediaPipe Face Landmarker (478 3D landmarks) to calculate the Eye Aspect Ratio (EAR) and sound warnings when eyes remain closed.
* **7-Emotion Photo Booth Challenge:** A gamified Gradio interface that tracks your peak expressions above a 25% floor and generates a downloadable photo strip.
* **Explainable AI (Grad-CAM):** Visualizes activation heatmaps from the final residual block to show which facial regions triggered the prediction.

The repository includes a Gradio web app (`app.py`), an OpenCV desktop webcam pipeline (`demo.py`), and a standalone fatigue monitor (`monitor.py`).

* **GitHub:** https://github.com/shahin13700/fer-emotion-recognition
* **License:** MIT

Would love to hear your feedback, feature ideas, or pull requests! If you find it useful, a star on GitHub is always appreciated! ⭐
```

---

### B. r/ComputerVision
* **Title:** `[P] EdgeVision: Real-time 817KB MiniXception CNN (51k params, 2ms CPU latency) for FER and Driver Drowsiness`
* **Flair:** `Project`
* **Body:**
```markdown
Hi r/ComputerVision,

I've open-sourced **EdgeVision**, a CPU-optimized pipeline combining real-time facial expression recognition (FER) with eye-aspect-ratio (EAR) fatigue monitoring.

### Technical Highlights:
1. **Model Architecture:** MiniXception with 4 residual depthwise-separable convolution blocks, global average pooling, and softmax output. Total size: 817 KB (51,255 weights).
2. **Training & Regularization:** Trained on FER2013 with class-balanced weighting (to address severe disgust/fear underrepresentation), batch normalization, and L2 regularization (0.01).
3. **Temporal Stability:** Rather than raw frame-by-frame argmax, predictions pass through an Exponential Moving Average (EMA) filter ($\alpha = 0.35$) with exponential confidence decay when no face is tracked.
4. **Driver Fatigue Guard:** MediaPipe FaceLandmarker Tasks API extracts 16 eyelid landmarks to measure Euclidean distances for Eye Aspect Ratio (EAR). Sustained closure for >20 frames triggers an audible/visual alarm.
5. **Grad-CAM Interpretability:** Computes gradients of the target class score with respect to the final 3×3 feature map, upsampled to the input crop for localization.

Repository & code: https://github.com/shahin13700/fer-emotion-recognition

Feedback on edge quantization (INT8/TFLite/ONNX) or alternative dataset benchmarks (RAF-DB / AffectNet) would be super welcome!
```

---

## 🍊 3. Hacker News ("Show HN")

* **Title:** `Show HN: EdgeVision – Real-time 817KB facial emotion recognition and fatigue guard`
* **URL:** `https://github.com/shahin13700/fer-emotion-recognition`
* **First Comment (Post immediately after submission):**
```markdown
Hi HN!

I built EdgeVision to explore lightweight computer vision that runs seamlessly on cheap CPUs without needing cloud APIs or discrete GPUs.

A common issue with facial emotion recognition demos is that they either require massive deep nets (50MB–200MB+) that lag on consumer laptops, or they produce frantic frame-to-frame label flicker.

EdgeVision solves this using:
1. An 817 KB MiniXception CNN (51k params) that runs in ~2ms per face on a laptop CPU.
2. Exponential Moving Average (EMA) smoothing across frames to stabilize predictions.
3. MediaPipe Face Landmarker for real-time Eye Aspect Ratio (EAR) driver drowsiness detection.
4. An interactive 7-Emotion Photo Booth challenge built with Gradio that renders a downloadable retro-style photo strip when you unlock all expressions.
5. Strict local privacy: webcam frames are processed entirely in memory; saving is disabled by default on shared servers.

Code and pre-trained weights (MIT): https://github.com/shahin13700/fer-emotion-recognition

Looking forward to your thoughts and critique!
```

---

## 💼 4. LinkedIn Post

* **Media to attach:** Short 10–15s screen recording of the Photo Booth or Driver Monitor in action.
* **Text:**
```text
Excited to open-source EdgeVision 🎭 — a lightweight, zero-GPU Computer Vision pipeline for real-time Facial Emotion Recognition and Driver Fatigue Monitoring!

Most deep learning vision models require discrete GPUs or external cloud APIs. EdgeVision is designed for edge efficiency:

🔹 817 KB model footprint (51,255 parameters using MiniXception)
🔹 ~2 ms CPU inference per face crop (tested on standard laptop CPUs)
🔹 Dual-purpose: 7-emotion expression tracking + MediaPipe Eye Aspect Ratio (EAR) driver drowsiness detection
🔹 EMA temporal smoothing to eliminate frame flicker
🔹 Gamified Gradio Photo Booth challenge with downloadable emotion photo strips
🔹 Explainable AI powered by Grad-CAM activation heatmaps

Everything is 100% open-source under the MIT license.

Check out the code, pre-trained weights, and documentation on GitHub:
👉 https://github.com/shahin13700/fer-emotion-recognition

If you like the project or are interested in edge AI and computer vision, feel free to drop a star! ⭐

#ComputerVision #MachineLearning #DeepLearning #Python #AI #EdgeAI #OpenSource #TensorFlow #MediaPipe
```

---

## 🐦 5. X (Twitter) Thread

* **Tweet 1 (with video attached):**
```text
I built an 817 KB neural network that runs real-time facial emotion recognition & driver fatigue detection on any cheap CPU with zero GPU lag. ⚡

Meet EdgeVision 🎭 — featuring an interactive 7-emotion photo booth challenge!

🧵👇 Code & weights below:
```
* **Tweet 2:**
```text
Key specs:
• 51,255 params (MiniXception architecture)
• ~2ms CPU latency per face
• MediaPipe Eye Aspect Ratio (EAR) drowsiness alerts
• EMA temporal smoothing (no label flickering)
• Grad-CAM explainability heatmaps
• Built with @Gradio & @GoogleMediaPipe

GitHub: https://github.com/shahin13700/fer-emotion-recognition
```
* **Tweet 3:**
```text
Try beating all 7 emotions in the Photo Booth challenge: Happy, Surprise, Neutral, Angry, Disgust, Sad, Fear!

It tracks your peak expressions and exports a downloadable photo strip. 📸

Star on GitHub if you find it helpful! ⭐
```
