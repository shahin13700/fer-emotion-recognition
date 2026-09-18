# Contributing to EdgeVision (Facial Emotion Recognition)

First off, thank you for considering contributing to **EdgeVision**! 🎉

EdgeVision is an open-source, ultra-lightweight real-time facial expression analysis suite engineered for edge devices, web browsers, and low-latency robotics. We welcome contributions ranging from algorithm optimization and edge model quantization to UX enhancements and documentation.

---

## 🧭 Code of Conduct

We are committed to providing a welcoming, inclusive, and harassment-free environment for everyone. Please be respectful, constructive, and collaborative in all communications, issues, and pull requests.

---

## 🚀 Quickstart Development Setup

### 1. Fork & Clone
```bash
git clone https://github.com/<your-username>/fer-emotion-recognition.git
cd fer-emotion-recognition
```

### 2. Set Up a Virtual Environment
We recommend Python 3.10 or 3.11:

```bash
# Using standard venv:
python -m venv .venv
source .venv/bin/activate       # On Linux/macOS
.\.venv\Scripts\Activate.ps1    # On Windows PowerShell

# Or using uv (ultrafast):
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 3. Verify Local Installation
Run our smoke test suite to confirm model loading, inference pipelines, and Gradio endpoints:

```bash
python -m pytest tests/ -v
# Or run direct module validation:
python -m py_compile app.py demo.py monitor.py explain.py train.py evaluate.py fine_tune.py
```

### 4. Launch the Interactive App
```bash
python app.py
```
Open `http://localhost:7860` in your web browser to test the Photo Booth, webcam stream, and video analyzer.

---

## 🎯 High-Priority Roadmap Challenges (Looking for Contributors!)

We are specifically seeking contributions across these 5 advanced challenges:

### 1. ⚡ Issue #1: INT8 Edge Quantization & Sub-3ms Inference
* **Goal:** Convert `model/emotion_model.keras` into ONNX format and INT8-quantized TFLite.
* **Benchmark:** Measure latency on Raspberry Pi 4 / 5, Intel i5 CPU, and Apple Silicon.
* **Target:** Sub-3ms per frame on CPU with $< 1\%$ drop in top-1 accuracy.
* **Tags:** `performance`, `edge-ai`, `onnx`, `quantization`

### 2. 👁️ Issue #2: 3D Head Pose & Gaze Vector Tracking
* **Goal:** Integrate MediaPipe Iris & FaceMesh 6-DOF landmarks into `monitor.py` / `app.py`.
* **Feature:** Estimate pitch, yaw, and roll to reject off-angle frames before feeding crops to the CNN, reducing False Positive rates during inattentive states.
* **Tags:** `computer-vision`, `mediapipe`, `geometry`

### 3. ⏱️ Issue #3: Temporal Micro-Expression Modeling (CNN + GRU/TCN)
* **Goal:** Implement a sliding window sequence model (16 frames) over spatial feature embeddings extracted from MiniXception's GlobalAveragePooling layer.
* **Feature:** Differentiate fleeting micro-expressions (e.g. suppressed contempt/fear) from prolonged macro-expressions.
* **Tags:** `deep-learning`, `temporal`, `recurrent-networks`

### 4. 🌐 Issue #4: Client-Side WASM Zero-Latency Inference (ONNX.js / WebGL)
* **Goal:** Build an in-browser inference tab running MiniXception entirely on client-side WebGL / WebGPU via ONNX Runtime Web.
* **Benefit:** Zero backend server cost, 100% user privacy, and zero network video latency.
* **Tags:** `frontend`, `wasm`, `webgpu`, `privacy`

### 5. 🔬 Issue #5: Edge Vision Transformer (MobileViT) Benchmark
* **Goal:** Benchmark a lightweight MobileViT or EfficientFormer against MiniXception on FER2013 and AffectNet.
* **Deliverable:** Parameter count, FLOPs comparison, latency profile, and Grad-CAM/Attention Rollout visualization.
* **Tags:** `research`, `vision-transformers`, `benchmarking`

---

## 🤝 Active Learning & Data Contributions

EdgeVision features a privacy-preserving **Active Learning Engine**:
- Contributors can run the **7-Emotion Photo Booth** (`python app.py`) to capture peak expressions.
- Contributed faces are saved locally into `dataset/user_contributed/` with append-only metadata logged to `dataset/user_contributed/metadata.jsonl`.
- Before fine-tuning, run the volume audit:
  ```bash
  python fine_tune.py
  ```
- `fine_tune.py` enforces **4 strict guardrails**:
  1. Minimum volume gate ($\ge 10$ samples per class).
  2. 15× oversampling of user data.
  3. Heavy data augmentation.
  4. Test set evaluation with automatic rollback protection if accuracy degrades.

---

## 🛠️ Pull Request Guidelines

1. **Branch Naming:**
   - Features: `feat/issue-number-short-description` (e.g., `feat/int8-onnx-quantization`)
   - Bugfixes: `fix/short-description` (e.g., `fix/webcam-scaling`)
   - Documentation: `docs/short-description`
2. **Commit Messages:** Follow conventional commits format:
   - `feat(edge): add INT8 tflite conversion script`
   - `fix(web): resolve webcam aspect ratio distortion on mobile`
   - `docs: update benchmarking table in README`
3. **Automated Testing:**
   - Ensure all tests pass (`pytest`).
   - Run `python -m py_compile` across modified files.
   - Do not commit large binary model checkpoints or raw dataset folders (`dataset/` is git-ignored).

---

## 💬 Questions & Community

Feel free to open an issue for questions, feature proposals, or architecture feedback. Thank you for making edge AI faster, lighter, and more accessible!
