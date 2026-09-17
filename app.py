"""
Facial Emotion Recognition (FER) - Interactive Web Application
Powered by MiniXception & Gradio. Ready for Hugging Face Spaces.
"""

import os
import json
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import gradio as gr

# -------------------------------------------------------------
# 1. Load Model & Label Mapping
# -------------------------------------------------------------
MODEL_PATH = 'model/emotion_model.keras'
INDICES_PATH = 'outputs/class_indices.json'

if not os.path.exists(INDICES_PATH):
    raise FileNotFoundError(f"Missing {INDICES_PATH}. Please train or copy outputs first.")

with open(INDICES_PATH, 'r') as f:
    class_indices = json.load(f)

# Invert {'angry': 0, ...} to {0: 'Angry', ...}
idx_to_class = {v: k.capitalize() for k, v in class_indices.items()}
emotion_labels = [idx_to_class[i] for i in range(len(idx_to_class))]

print(f"Loading model from {MODEL_PATH}...")
model = load_model(MODEL_PATH)

# Warmup model
_ = model(tf.zeros((1, 48, 48, 1)), training=False)
print("Model initialized and ready.")

# Face detector
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)


# -------------------------------------------------------------
# 2. Prediction Pipeline
# -------------------------------------------------------------
def predict_emotion(input_image):
    """
    Takes an RGB input image from webcam or upload, detects faces,
    draws bounding boxes with predictions, and returns the annotated image
    along with class probability dictionary.
    """
    if input_image is None:
        return None, {}

    # Copy image and convert to grayscale for face detection
    img_bgr = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(40, 40)
    )

    confidences = {}

    if len(faces) == 0:
        # Fallback: User might have uploaded an already-cropped facial portrait (e.g. FER dataset sample)
        roi_gray = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
        roi = roi_gray.astype('float32') / 255.0
        roi = np.expand_dims(np.expand_dims(roi, axis=0), axis=-1)
        
        preds = model(tf.convert_to_tensor(roi), training=False).numpy()[0]
        confidences = {emotion_labels[i]: float(preds[i]) for i in range(len(emotion_labels))}
        
        # Annotate full image
        top_idx = int(np.argmax(preds))
        top_label = emotion_labels[top_idx]
        top_conf = preds[top_idx]
        
        cv2.putText(img_bgr, f"{top_label} ({top_conf*100:.1f}%) [No face crop needed]",
                    (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 230, 115), 2, cv2.LINE_AA)
        output_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return output_rgb, confidences

    # Process all detected faces
    for (x, y, w, h) in faces:
        # Draw bounding box
        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (255, 170, 0), 2)

        # Crop & preprocess exactly matching training (48x48 grayscale, [0, 1])
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        roi = roi_gray.astype('float32') / 255.0
        roi = np.expand_dims(np.expand_dims(roi, axis=0), axis=-1)

        preds = model(tf.convert_to_tensor(roi), training=False).numpy()[0]
        
        # Primary face confidences for label widget
        confidences = {emotion_labels[i]: float(preds[i]) for i in range(len(emotion_labels))}
        
        top_idx = int(np.argmax(preds))
        top_label = emotion_labels[top_idx]
        top_conf = preds[top_idx]

        # Draw label badge
        label_text = f"{top_label}: {top_conf*100:.1f}%"
        cv2.rectangle(img_bgr, (x, y - 28), (x + len(label_text)*14, y), (255, 170, 0), -1)
        cv2.putText(img_bgr, label_text, (x + 4, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    output_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return output_rgb, confidences


# -------------------------------------------------------------
# 3. Gradio Interface Definition
# -------------------------------------------------------------
custom_css = """
#header { text-align: center; margin-bottom: 20px; }
#github-btn { margin-top: 10px; display: inline-block; }
"""

with gr.Blocks(title="Edge FER — Real-Time Facial Emotion Recognition") as demo:
    gr.Markdown(
        """
        # 🎭 Edge-Vision: Facial Emotion Recognition
        ### Lightweight Real-Time Deep Learning using MiniXception (~60k parameters, 817 KB)
        
        [![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/shahin13700/fer-emotion-recognition)
        [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
        [![Model Size](https://img.shields.io/badge/Model%20Size-817%20KB-brightgreen)]()
        
        Upload an image or capture from your webcam to detect faces and classify facial expressions into **7 emotion categories**:
        *Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise*.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(
                label="Input Face Image / Webcam Snapshot",
                sources=["upload", "webcam"],
                type="numpy"
            )
            analyze_btn = gr.Button("🔍 Analyze Emotion", variant="primary", size="lg")
            
        with gr.Column(scale=1):
            output_img = gr.Image(label="Annotated Detections")
            emotion_labels_output = gr.Label(num_top_classes=7, label="Emotion Probabilities")

    analyze_btn.click(
        fn=predict_emotion,
        inputs=input_img,
        outputs=[output_img, emotion_labels_output]
    )
    input_img.change(
        fn=predict_emotion,
        inputs=input_img,
        outputs=[output_img, emotion_labels_output]
    )

    gr.Markdown(
        """
        ---
        ### ⚡ Technical Details
        - **Model Backbone:** MiniXception with Depthwise Separable Convolutions & Residual Connections
        - **Dataset:** FER2013 (35,887 benchmark images) with balanced class weight penalties
        - **Inference Speed:** Sub-10ms per face on CPU
        - **Repository:** [github.com/shahin13700/fer-emotion-recognition](https://github.com/shahin13700/fer-emotion-recognition)
        """
    )

if __name__ == '__main__':
    demo.launch(css=custom_css, share=False)
