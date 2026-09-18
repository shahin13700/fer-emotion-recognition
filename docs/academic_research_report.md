# Facial Emotion Recognition using FER2013
## CST8508 — Machine Vision Final Project
**Student:** Shahin (Shaheeend)
**Course:** CST8508 Machine Vision
**Date:** April 2026

---

## 1. Introduction

Facial Emotion Recognition (FER) is the task of automatically identifying a person's emotional state from their facial expression. This project builds a complete FER system using deep learning that classifies facial images into seven emotion categories: **Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise**.

**Problem Statement:** Human emotions are complex and nuanced. Building a model that can reliably distinguish between seven distinct expressions from a 48×48 pixel grayscale image — while dealing with class imbalance and label noise — is a non-trivial classification challenge.

**Objectives:**
- Train a convolutional neural network on the FER2013 benchmark dataset
- Address class imbalance using weighted training
- Evaluate per-class performance and analyze common misclassification patterns
- Deploy the trained model in a live webcam demo for real-time emotion recognition

**Applications:** Patient monitoring in healthcare, student engagement tracking in education, driver drowsiness detection in automotive safety, and human-computer interaction systems.

---

## 2. Dataset

**Source:** FER2013 — Kaggle, originally from the 2013 ICML Facial Expression Recognition Challenge (Goodfellow et al., 2013)

**Structure:**
- Total images: 35,887
- Training set: 28,709 images
- Test set: 7,178 images (the Kaggle `msambare/fer2013` split merges the original public and private test sets)
- Format: Grayscale, 48×48 pixels
- Classes: 7 emotion categories

**Class Distribution (Training Set):**

| Emotion  | Images |
|----------|--------|
| Happy    | 7,215  |
| Neutral  | 4,965  |
| Sad      | 4,830  |
| Fear     | 4,097  |
| Angry    | 3,995  |
| Surprise | 3,171  |
| Disgust  | 436    |

**Key Challenge — Class Imbalance:** The Happy class has over 16× more samples than Disgust. Without intervention, a naive model will heavily favor dominant classes and ignore rare ones entirely.

**Known Limitation:** FER2013 was collected by crawling Google Image Search. Labels were generated through crowd-sourcing, resulting in noisy and ambiguous annotations. Human-level accuracy on this dataset is approximately 65%, which acts as a practical accuracy ceiling.

**Preprocessing Steps:**

- **Normalization:** Pixel values rescaled from [0, 255] to [0, 1] to support stable gradient flow during backpropagation.
- **Data Augmentation (training only):** Horizontal flip, rotation ±10°, and zoom ±10° were applied to artificially expand the training set and improve generalization.
- **Validation Split:** 20% of training data (5,742 images) was held out as a validation set. The test set remained completely untouched until final evaluation to prevent data leakage.
- **Class Weights:** Computed using scikit-learn's `compute_class_weight('balanced')` to penalize misclassification of rare emotions. Approximate weights: Disgust ~6.0×, Fear ~1.2×, Happy ~0.7× (downweighted as overrepresented).

---

## 3. Methodology

### Model Architecture: MiniXception

MiniXception was selected for this project based on Arriaga et al. (2017), who proposed it specifically for real-time facial expression recognition. It uses **depthwise separable convolutions** to achieve competitive accuracy with approximately 60,000 parameters — deliberately lightweight for real-time inference.

**Why depthwise separable convolutions?** Standard convolutions apply a single filter across all input channels simultaneously. Depthwise separable convolutions split this into two steps: (1) a depthwise convolution that filters each channel independently, then (2) a 1×1 pointwise convolution that combines channels. This reduces computational cost by 8–9× with minimal accuracy loss.

**Why residual connections?** Skip connections allow gradients to flow directly through the network during backpropagation, solving the vanishing gradient problem in deeper networks. They also allow the network to learn small corrections rather than full transformations at each layer.

**Architecture Summary:**

| Stage | Layer | Details |
|-------|-------|---------|
| Input | — | 48 × 48 × 1 (grayscale) |
| Stage 1 | Conv2D × 2 | 8 filters, 3×3, BatchNorm, ReLU |
| Stage 2 | Residual Xception Block × 4 | Filters: 16, 32, 64, 128; SeparableConv2D + shortcut |
| Stage 3 | GlobalAveragePooling2D | — |
| Stage 3 | Dropout(0.5) | Regularization |
| Output | Dense(7, softmax) | 7-class probability distribution |

L2 regularization (λ=0.01) was applied to the two stem Conv2D layers and the Dense classifier. The resulting model has 51,255 trainable parameters.

### Training Configuration

| Setting | Value |
|---------|-------|
| Optimizer | Adam (lr=0.001) |
| Loss | Categorical crossentropy |
| Batch size | 64 |
| Max epochs | 100 |

**Callbacks used:**
- **EarlyStopping** (patience=10, monitors validation loss, restores best weights)
- **ReduceLROnPlateau** (factor=0.5, patience=5, min lr=1e-6)
- **ModelCheckpoint** (saves best model by validation accuracy)

Training ran on Colab with EarlyStopping ending it well before the 100-epoch cap; the per-epoch log was not preserved, so epoch counts and the exact learning-rate schedule are not reported here.

---

## 4. Results

### Overall Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | **57.30%** |
| Macro F1-Score | 0.54 |
| Weighted F1-Score | 0.57 |
| Total Test Samples | 7,178 |

All results below are reproducible from the weights shipped in `model/emotion_model.keras` by running `evaluate.py`; they match `outputs/classification_report.txt`.

### Per-Class Performance

| Emotion  | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Angry    | 0.45      | 0.54   | 0.49     | 958     |
| Disgust  | 0.40      | 0.61   | 0.48     | 111     |
| Fear     | 0.41      | 0.30   | 0.34     | 1,024   |
| Happy    | 0.85      | 0.75   | **0.79** | 1,774   |
| Neutral  | 0.49      | 0.61   | 0.55     | 1,233   |
| Sad      | 0.48      | 0.42   | 0.45     | 1,247   |
| Surprise | 0.67      | 0.75   | **0.71** | 831     |

### Key Observations from the Confusion Matrix

**Best-performing classes:**
- **Happy (75% recall, 85% precision):** The open-mouth smile and raised cheeks are visually distinctive.
- **Surprise (75% recall):** Raised eyebrows combined with an open mouth create a unique pattern.
- **Disgust (61% recall):** Despite only 111 test samples (436 in training), class weights forced the model to learn this class, at the cost of precision (40%): a quarter of true Disgust images are predicted Angry.

**Most confused pairs (row-normalized confusion matrix):**
- Sad → Neutral: 23% of sad samples predicted as neutral; Sad → Angry another 15%
- Angry → Neutral: 16% of angry samples predicted as neutral
- Fear → Sad 21%, Fear → Angry 16%, Fear → Neutral 14%, Fear → Surprise 13%: with only 30% recall, Fear is the weakest class
- Disgust → Angry: 25% of disgust samples predicted as angry

---

## 5. Evaluation

### Performance in Context

A test accuracy of 57.3% is consistent with published MiniXception results on FER2013 and is competitive given the dataset's known label noise. Human-level accuracy on FER2013 is approximately 65%, meaning the model is performing within a reasonable margin of the theoretical ceiling for this dataset.

### Challenges and How They Were Addressed

**Challenge 1 — Class Imbalance**
Without intervention, rare emotions like Disgust (436 samples) would be completely ignored by the model. This was resolved using scikit-learn's `compute_class_weight('balanced')`, which computed per-class penalty weights. The result: Disgust achieved 61% recall despite being the smallest class.

**Challenge 2 — Fear Class Performance**
Fear achieved only 30% recall, the worst of all classes. In static images, the expression of fear closely resembles surprise (wide eyes) and sometimes neutral or sad. Without temporal cues from video sequences, distinguishing fear from other emotions is inherently difficult — even for human annotators.

**Challenge 3 — Label Noise in FER2013**
The dataset was collected through automated web scraping and crowd-sourced labeling. Many images are mislabeled or ambiguous. This is a fundamental dataset limitation that cannot be resolved through model architecture changes alone — it places a hard ceiling on achievable accuracy.

**Challenge 4 — Data Loading Bottleneck**
Reading 35,000 images directly from Google Drive produced training steps of 38 seconds each. Copying the dataset to Colab's local disk reduced this to under 1 second per step — a 40× speedup with a single change to the data pipeline.

### Live Demo Observations

The trained model was deployed in a real-time webcam demo using OpenCV's Haar cascade face detector, with the forward pass compiled via `tf.function` (about 2 ms per face on a desktop CPU versus roughly 25 ms for an eager call). Happy, surprised, and neutral were reliably recognized. Lighting conditions significantly affected face detection quality. The disgust bar rarely activated, which is consistent with its limited representation in training data relative to real-world expression frequency.

---

## 6. Lessons Learned

### Technical Lessons

**Data pipeline quality matters more than model architecture.**
The single most impactful optimization in this project was not a change to the model — it was fixing the data loading. Moving the dataset from Google Drive to local Colab disk produced a 40× training speedup. Infrastructure decisions have disproportionate impact on iteration speed.

**Class imbalance requires explicit handling.**
Without class weights, disgust would have had near-zero recall. Balanced class weights are essential for any real-world imbalanced classification task where minority classes carry meaningful signal.

**FER2013's label noise is a hard ceiling.**
Increasing model complexity beyond a certain point yields diminishing returns on this dataset. Accuracy gains beyond approximately 65% require either a cleaner dataset or additional modalities (audio, temporal video frames).

**Review cycles prevent silent bugs.**
A missing `shuffle=False` on the test data generator would never have caused a crash — the confusion matrix would simply have been silently wrong. Manual code review before execution caught this and other critical bugs before they could corrupt evaluation results.

### Future Improvements

**Short-term:**
- Increase starting filter count from 8 to 16 or 32 — published results suggest a 3–5% accuracy gain with minimal latency cost
- Apply test-time augmentation: average predictions across multiple augmented versions of each test image

**Medium-term:**
- Fine-tune EfficientNetV2-S pretrained on ImageNet — published results show 70–74% on FER2013
- Add temporal modeling using LSTMs or 3D-CNNs to capture facial movement cues across video frames
- Switch to AffectNet (450,000 images) or RAF-DB (30,000 images, cleaner labels) to improve generalization

**Long-term:**
- Multimodal fusion combining facial expression, voice tone, and body language
- Deployment as a Flask/FastAPI web API
- On-device inference for privacy-preserving processing

---

## References

- Arriaga, O., Valdenegro-Toro, M., & Plöger, P. (2017). Real-time Convolutional Neural Networks for Emotion and Gender Classification. *arXiv:1710.07557*
- Goodfellow, I. J., Erhan, D., Carrier, P. L., & Courville, A. (2013). Challenges in Representation Learning: A report on three machine learning contests. *ICML Workshop on Challenges in Representation Learning*
- Chollet, F. (2017). Xception: Deep Learning with Depthwise Separable Convolutions. *CVPR 2017*
- FER2013 Dataset: https://www.kaggle.com/datasets/msambare/fer2013
- GitHub Repository: https://github.com/shahin13700/fer-emotion-recognition
