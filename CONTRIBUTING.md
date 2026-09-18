# Contributing to EdgeVision

Thank you for your interest in contributing to **EdgeVision**!

EdgeVision is an ultra-lightweight real-time facial expression and driver fatigue monitoring project designed for edge devices and consumer CPUs. We welcome contributions that improve latency, robustness, test coverage, and documentation.

---

## 🚀 Quickstart Development Setup

### 1. Fork & Clone
```bash
git clone https://github.com/<your-username>/fer-emotion-recognition.git
cd fer-emotion-recognition
```

### 2. Virtual Environment Setup
```bash
# Using standard Python venv:
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
.\.venv\Scripts\Activate.ps1    # Windows PowerShell

# Install dependencies:
pip install -r requirements.txt
```

### 3. Run the Test Suite
```bash
pytest tests/ -v
```

---

## 🎯 Roadmap & Engineering Goals

We are looking for contributions across these two practical edge engineering areas:

### 1. ⚡ ONNX Runtime & INT8 Quantization
* **Goal:** Export `model/emotion_model.keras` to ONNX and INT8-quantized TFLite.
* **Benchmark:** Measure latency and memory footprint on low-power devices (Raspberry Pi, CPU edge nodes).
* **Target:** Sub-3ms inference latency on modern CPUs with minimal accuracy loss.

### 2. 👁️ 3D Head Pose Filtering
* **Goal:** Use MediaPipe FaceMesh 6-DOF landmarks to calculate pitch, yaw, and roll in `monitor.py` and `app.py`.
* **Feature:** Reject extreme off-angle faces before classification to reduce false positives in driver monitoring and photo booth sessions.

---

## 💾 Local Active Learning & Fine-Tuning

EdgeVision includes a local active learning script for tailoring the model:
1. Capture expressions in the Photo Booth (`python app.py`) and save them locally.
2. Contributed crops are saved to `dataset/user_contributed/` with metadata in `metadata.jsonl`.
3. Run `python fine_tune.py` to audit volume and train with our blended data generator.
4. If accuracy drops on the test set, weights remain protected. You can restore the pristine original baseline anytime with:
   ```bash
   python fine_tune.py --reset
   ```

---

## 🛠️ Pull Request Guidelines

1. **Branch Naming:**
   - Features: `feat/short-description` (e.g., `feat/onnx-quantization`)
   - Bugfixes: `fix/short-description` (e.g., `fix/webcam-aspect-ratio`)
2. **Commit Messages:** Follow conventional commits format:
   - `feat(edge): add onnx model conversion script`
   - `fix(booth): improve preview image scaling`
3. **Testing:**
   - Run `pytest tests/ -v` and ensure all tests pass.
   - Run `python -m py_compile app.py fine_tune.py demo.py monitor.py explain.py`.
   - Never commit raw dataset folders or private facial crops (`dataset/` is git-ignored).
