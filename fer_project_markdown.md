# Facial Emotion Recognition using FER2013
## CST8508 — Machine Vision Final Project
**Student:** Shahin (Shaheeend)
**Course:** CST8508 Machine Vision
**Date:** April 2026

---

## 1. Project Overview

This project builds a complete Facial Emotion Recognition (FER) system using deep learning. The system learns to classify human facial expressions into 7 emotion categories from a 48x48 pixel grayscale image, and then applies that knowledge in real time through a live webcam feed.

**The 7 emotion classes:**
- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

**Why this matters:** Facial emotion recognition has real-world applications in healthcare (patient monitoring), education (student engagement tracking), automotive safety (driver drowsiness detection), human-computer interaction, and market research.

---

## 2. Dataset: FER2013

**Source:** Kaggle — originally from the 2013 ICML Facial Expression Recognition Challenge (Goodfellow et al.)

**Dataset statistics:**
- Total images: 35,887
- Training set: 28,709 images
- Test set: 3,589 images
- Image format: Grayscale, 48x48 pixels
- Number of classes: 7

**Class distribution (training set):**
- Angry: 3,995 images
- Disgust: 436 images
- Fear: 4,097 images
- Happy: 7,215 images
- Neutral: 4,965 images
- Sad: 4,830 images
- Surprise: 3,171 images

**Key challenge — class imbalance:** The happy class has over 16x more samples than disgust. This means a naive model will heavily favor happy and ignore disgust entirely. This was addressed using class weights during training.

**Known limitation:** FER2013 was collected from Google Image Search and contains noisy labels — some images are mislabeled or ambiguous. This caps the theoretical accuracy ceiling. Human-level accuracy on FER2013 is approximately 65%.

**Validation strategy:** 20% of training data (5,742 images) was held out as a validation set. The test set (3,589 images) was kept completely untouched until final evaluation to prevent data leakage.

---

## 3. Preprocessing and Data Augmentation

### Normalization
Pixel values were rescaled from [0, 255] to [0, 1]. This helps gradient flow during backpropagation and prevents numerical instability.

### Data Augmentation (training only)
Augmentation was applied exclusively to the training set to artificially expand the dataset and improve generalization:
- **Horizontal flip** — faces are approximately symmetric
- **Rotation ±10°** — accounts for slight head tilts
- **Zoom ±10%** — accounts for varying camera distances

The validation and test generators used rescaling only — no augmentation — to ensure unbiased evaluation metrics.

### Class Weights
Class weights were computed using scikit-learn's `compute_class_weight('balanced')` function. This penalizes the model more heavily for misclassifying rare emotions like disgust, forcing it to learn all 7 classes rather than only the dominant ones.

Approximate class weights computed:
- Disgust: ~6.0x baseline
- Fear: ~1.2x baseline
- Happy: ~0.7x baseline (downweighted as overrepresented)

---

## 4. Model Architecture: MiniXception

### Why MiniXception?
MiniXception was proposed by Arriaga et al. (2017) specifically for real-time facial expression recognition. It uses depthwise separable convolutions to achieve competitive accuracy with very few parameters, making it fast enough to run on consumer hardware in real time.

### Architecture Details

**Input:** 48 × 48 × 1 (grayscale)

**Stage 1 — Initial feature extraction:**
- Conv2D(8, 3x3) → BatchNorm → ReLU
- Conv2D(8, 3x3) → BatchNorm → ReLU

**Stage 2 — Residual Xception modules (x4):**
Filter sizes: [16, 32, 64, 128]

For each module:
- Residual shortcut: Conv2D(filters, 1x1, stride=2)
- Main path: SeparableConv2D → BatchNorm → ReLU → SeparableConv2D → BatchNorm → MaxPool(2x2)
- Add residual + main path

**Stage 3 — Classification head:**
- GlobalAveragePooling2D
- Dropout(0.5)
- Dense(7, activation='softmax')

**Regularization:** L2 regularization (λ=0.01) on Conv2D layers.

**Total parameters:** Approximately 60,000 — deliberately lightweight for real-time inference.

### Why depthwise separable convolutions?
Standard convolutions apply a single filter across all channels simultaneously. Depthwise separable convolutions split this into two steps: (1) a depthwise convolution that filters each channel independently, then (2) a 1x1 pointwise convolution that combines channels. This reduces computational cost by 8-9x with minimal accuracy loss.

### Why residual connections?
Skip connections allow gradients to flow directly through the network during backpropagation, solving the vanishing gradient problem in deeper networks. They also allow the network to learn residual features (small corrections) rather than full transformations at each layer.

---

## 5. Training Configuration

### Optimizer
**Adam** with initial learning rate 0.001. Adam combines the benefits of momentum (RMSProp) and adaptive learning rates (Adagrad), making it well-suited for noisy gradients from image data.

### Loss Function
**Categorical crossentropy** — standard for multi-class classification with one-hot encoded labels.

### Callbacks

**EarlyStopping:**
- Monitor: validation loss
- Patience: 10 epochs
- Restores best weights on stop

**ReduceLROnPlateau:**
- Monitor: validation loss
- Factor: 0.5 (halves the learning rate)
- Patience: 5 epochs
- Minimum lr: 1e-6

**ModelCheckpoint:**
- Saves only the best model (by validation accuracy)
- Saved to model/emotion_model.keras

### Training Results
- Training stopped at epoch 65 (EarlyStopping)
- Best model restored from epoch 55
- ReduceLROnPlateau fired 4 times: 0.001 → 0.0005 → 0.00025 → 0.0000625 → 0.00003125
- Total training time: approximately 23 minutes on Google Colab T4 GPU
- Batch size: 64
- Maximum epochs set: 100

---

## 6. Development Tools and Workflow

### IDE: Google Antigravity
Google Antigravity (released November 2025) is Google's agent-first IDE built on a VS Code fork. It allows autonomous AI agents to plan, write code, run terminal commands, and verify results. Powered primarily by Gemini 3.1 Pro with support for Claude Sonnet 4.6.

### Two-Agent Development Workflow
A dual-agent review loop was used throughout the project:

**Claude Sonnet 4.6 (Coder Agent):**
- Wrote all Python scripts (train.py, evaluate.py, demo.py)
- Generated implementation plans before writing code
- Fixed runtime errors autonomously

**Gemini 3 Pro (Reviewer Agent):**
- Reviewed each script before execution
- Caught 3 critical bugs that would have produced incorrect results

**Critical bugs caught by Gemini:**
1. Missing `shuffle=False` on the test generator — would have produced a completely misaligned confusion matrix
2. `ImageDataGenerator` deprecation warning with TF 2.16 — flagged for awareness
3. Hardcoded emotion labels — replaced with dynamic loading from class_indices.json

### Training Platform: Google Colab T4 GPU
- Dataset downloaded directly from Kaggle via API (60MB zip, 5 seconds) instead of Google Drive
- Key insight: reading 35,000 images directly from Google Drive was 40x slower than local disk (38s/step vs <1s/step)
- Fix: copy dataset to Colab's /content/ local disk before training

### Version Control: GitHub
Repository: github.com/shahin13700/fer-emotion-recognition

---

## 7. Evaluation Results

### Overall Performance
- **Test accuracy: 57.15%**
- Macro F1-score: 0.53
- Weighted F1-score: 0.56
- Total test samples: 7,178

### Per-Class Performance

| Emotion | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| Angry   | 0.45      | 0.54   | 0.49     | 958     |
| Disgust | 0.34      | 0.64   | 0.45     | 111     |
| Fear    | 0.43      | 0.24   | 0.31     | 1,024   |
| Happy   | 0.81      | 0.78   | 0.79     | 1,774   |
| Neutral | 0.48      | 0.63   | 0.54     | 1,233   |
| Sad     | 0.48      | 0.40   | 0.43     | 1,247   |
| Surprise| 0.68      | 0.71   | 0.70     | 831     |

### Confusion Matrix Key Findings

**Strongest emotions:**
- Happy: 78% recall — distinctive open-mouth smile, raised cheeks
- Surprise: 71% recall — raised eyebrows and open mouth are unique
- Disgust: 64% recall — despite only 111 samples, class weights worked

**Most confused emotions:**
- Angry ↔ Neutral: angry often predicted as neutral (18% confusion)
- Sad ↔ Neutral: sad often predicted as neutral (26% confusion)
- Sad ↔ Angry: 15% confusion
- Fear → almost everything: only 24% recall, confused with all emotions

**Why fear performs worst:**
Fear in static images presents similarly to surprise (wide eyes) and sometimes neutral or sad. Without temporal information (video sequences showing the build-up of fear), static images alone are insufficient to reliably distinguish fear from other emotions. Even human annotators struggle with this class.

---

## 8. Live Webcam Demo

### How demo.py Works

1. Load trained model from model/emotion_model.keras
2. Load class index mapping from outputs/class_indices.json
3. Load OpenCV Haar cascade face detector: cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
4. Open webcam with cv2.VideoCapture(0)
5. For each frame:
   - Convert to grayscale
   - Detect faces using cascade classifier
   - For each face: crop → resize to 48x48 → normalize to [0,1] → expand dims
   - Run inference: model(inputs, training=False)
   - Draw green bounding box around face
   - Overlay predicted emotion label and confidence percentage
   - Draw 7-bar probability chart for all emotions
6. Press Q to quit

### Real-World Observations
- Happy, surprised, and neutral work reliably on camera
- Disgust bar rarely activates — consistent with the class being underrepresented in training
- Lighting conditions significantly affect face detection quality
- Model inference speed is fast enough for smooth real-time video

---

## 9. Lessons Learned

### Technical Lessons

**Data pipeline matters more than model architecture:**
The most impactful optimization was not changing the model — it was fixing the data loading. Reading images from Google Drive caused 38-second steps. Moving data to local disk reduced this to under 1 second. A 40x speedup from a single line of code.

**Review loops catch bugs that run silently:**
The missing `shuffle=False` bug would never have caused a crash — the confusion matrix would have simply been wrong. Without the Gemini review step, this would have gone unnoticed until results seemed unexplained.

**Class imbalance requires explicit handling:**
Without class weights, disgust would have had near-zero recall despite being a real emotion category. Balanced class weights are essential for any imbalanced classification problem.

**GPU setup on Windows is fragile:**
TensorFlow dropped native Windows GPU support after version 2.10. The CUDA + cuDNN manual installation process took over an hour and ultimately failed. Cloud training (Colab) was the pragmatic solution.

**FER2013 label noise is a fundamental limitation:**
The dataset's labels were generated by crowd-sourcing on images collected from Google Image Search. Many labels are ambiguous or incorrect. This places a hard ceiling on achievable accuracy regardless of model complexity.

### Process Lessons

**Agentic development accelerates without sacrificing quality:**
The Claude (coder) + Gemini (reviewer) workflow caught real bugs before they ran, saving debugging time. The agent-written code was well-structured and followed all specified requirements.

**Start with the right infrastructure:**
Using the Kaggle API to download directly to Colab took 5 seconds. Uploading via Google Drive browser interface took 54+ minutes. Infrastructure decisions have outsized impact on workflow speed.

---

## 10. Future Work

### Short-Term Improvements
- **Increase model capacity:** Replace MiniXception's 8 starting filters with 16 or 32 — published results suggest 3-5% accuracy gain with minimal latency cost
- **Data cleaning:** Remove or relabel the most ambiguous FER2013 samples using an automated confidence filter
- **Test-time augmentation:** Average predictions across multiple augmented versions of each test image

### Medium-Term
- **Transfer learning:** Fine-tune EfficientNetV2-S pretrained on ImageNet — published results show 70-74% on FER2013
- **Temporal modeling:** Use video sequences and an LSTM or 3D-CNN to incorporate facial movement cues
- **Larger dataset:** AffectNet (450,000 images) or RAF-DB (30,000 images with cleaner labels) would significantly improve generalization

### Long-Term
- **Multimodal fusion:** Combine facial expression + voice tone + body language for robust real-world emotion recognition
- **Deployment:** Flask/FastAPI web API hosted on Azure, accessible from any device
- **Privacy-preserving inference:** On-device processing so no face images leave the user's hardware

---

## 11. References

- Arriaga, O., Valdenegro-Toro, M., & Plöger, P. (2017). Real-time Convolutional Neural Networks for Emotion and Gender Classification. *arXiv:1710.07557*
- Goodfellow, I. J., Erhan, D., Carrier, P. L., & Courville, A. (2013). Challenges in Representation Learning: A report on three machine learning contests. *ICML Workshop on Challenges in Representation Learning*
- Chollet, F. (2017). Xception: Deep Learning with Depthwise Separable Convolutions. *CVPR 2017*
- FER2013 Dataset: https://www.kaggle.com/datasets/msambare/fer2013
- GitHub Repository: https://github.com/shahin13700/fer-emotion-recognition
