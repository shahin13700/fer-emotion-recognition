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
The suite must leave `git status` clean; CI fails if a test writes into a tracked file. Write generated files to `tmp_path`.

### 4. Optional: FER2013 for evaluation and fine-tuning
`evaluate.py`, `fine_tune.py`, `train.py` and `scripts/calc_percentiles.py` need the dataset unpacked at `dataset/train` and `dataset/test` (git-ignored):
```bash
kaggle datasets download -d msambare/fer2013
unzip -q fer2013.zip -d dataset/
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
1. Capture expressions in the Photo Booth (`python app.py`) and save them locally (≥10 per emotion).
2. Contributed crops are saved to `dataset/user_contributed/` with metadata in `metadata.jsonl`.
3. Run `python fine_tune.py` to audit volume (unique images only; duplicates are ignored) and train with the blended data generator. 20% of your faces are held out to measure whether the model actually improved on *you*.
4. The candidate is promoted only if it does not regress on your held-out faces, stays within `--max_test_drop` (default 1 pp) accuracy and macro-F1 on FER2013, and no class loses more than `--max_class_drop` (default 3 pp) recall; the previous weights are backed up to `model/emotion_model_backup.keras`. Restore the original baseline anytime with:
   ```bash
   python fine_tune.py --reset
   ```

### Claims in the README must be backed by an artifact
Accuracy numbers come from `outputs/classification_report.txt` (regenerate with `evaluate.py`), the parameter count from the model file (checked in CI), and latency figures should state the CPU they were measured on.

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
   - Never commit raw dataset folders, private facial crops, or fine-tuned/backup weights (`dataset/` and `model/*_backup|_finetuned|_original.keras` are git-ignored).
   - Keep exactly one OpenCV distribution in `requirements.txt` (`opencv-contrib-python`, which mediapipe requires); CI checks this.
   - Rebuild your venv from `requirements.txt` before trusting local test results; `pip check` must be clean.
   - Anything that imports `mediapipe` must use the Tasks API (`mediapipe.tasks.python.vision`); `mp.solutions` no longer exists in the pinned version.
