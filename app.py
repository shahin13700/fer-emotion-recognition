"""
Facial Emotion Recognition (FER) - Interactive Multi-Modal Web Application
Supports:
1. Live Streaming Webcam Video
2. Full Video File Processing (.mp4, .mov, .avi)
3. Still Image Analysis
Powered by MiniXception & Gradio. Ready for Hugging Face Spaces.
"""

import os
import json
import tempfile
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

idx_to_class = {v: k.capitalize() for k, v in class_indices.items()}
emotion_labels = [idx_to_class[i] for i in range(len(idx_to_class))]

print(f"Loading model from {MODEL_PATH}...")
model = load_model(MODEL_PATH)
_ = model(tf.zeros((1, 48, 48, 1)), training=False)
print("Model initialized and ready.")

cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)


# -------------------------------------------------------------
# 2. Core Processing Functions
# -------------------------------------------------------------
def annotate_frame(frame_bgr, smoothed_preds=None, alpha=0.65):
    """
    Detects faces on a single BGR frame, crops, predicts emotion,
    draws bounding boxes, labels, and probability bar charts.
    Returns: (annotated_bgr, confidences_dict, updated_smoothed_preds)
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(40, 40)
    )

    confidences = {}
    h, w, _ = frame_bgr.shape

    if len(faces) == 0:
        return frame_bgr, confidences, None

    for (x, y, fw, fh) in faces:
        cv2.rectangle(frame_bgr, (x, y), (x + fw, y + fh), (255, 170, 0), 2)

        roi_gray = gray[y:y+fh, x:x+fw]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        roi = roi_gray.astype('float32') / 255.0
        roi = np.expand_dims(np.expand_dims(roi, axis=0), axis=-1)

        raw_preds = model(tf.convert_to_tensor(roi), training=False).numpy()[0]

        if smoothed_preds is None:
            smoothed_preds = raw_preds
        else:
            smoothed_preds = alpha * raw_preds + (1.0 - alpha) * smoothed_preds

        confidences = {emotion_labels[i]: float(smoothed_preds[i]) for i in range(len(emotion_labels))}
        
        top_idx = int(np.argmax(smoothed_preds))
        top_label = emotion_labels[top_idx]
        top_conf = smoothed_preds[top_idx]

        # Top label badge
        label_text = f"{top_label}: {top_conf*100:.1f}%"
        cv2.rectangle(frame_bgr, (x, y - 26), (x + len(label_text)*14, y), (255, 170, 0), -1)
        cv2.putText(frame_bgr, label_text, (x + 4, y - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

        # Draw probability bar chart on right of face if space permits
        bar_x = x + fw + 10
        if bar_x + 150 < w:
            cv2.rectangle(frame_bgr, (bar_x - 5, y - 10), (bar_x + 150, y + 145), (10, 10, 10), -1)
            for i, prob in enumerate(smoothed_preds):
                y_pos = y + (i * 20) + 10
                cv2.putText(frame_bgr, f"{emotion_labels[i][:4]}:", (bar_x, y_pos), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)
                b_width = int(prob * 70)
                b_col = (0, 230, 115) if i == top_idx else (180, 180, 180)
                cv2.rectangle(frame_bgr, (bar_x + 60, y_pos - 7), (bar_x + 60 + b_width, y_pos + 1), b_col, -1)

    return frame_bgr, confidences, smoothed_preds


def process_image(input_image):
    """Processes a single uploaded image or snapshot."""
    if input_image is None:
        return None, {}
    frame_bgr = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    annotated_bgr, confs, _ = annotate_frame(frame_bgr)
    return cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB), confs


def process_live_frame(frame):
    """Processes real-time streaming webcam frames in browser."""
    if frame is None:
        return None
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    annotated_bgr, _, _ = annotate_frame(frame_bgr)
    return cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)


def process_video_file(video_path, progress=gr.Progress()):
    """Processes an uploaded video file frame-by-frame and returns annotated MP4."""
    if not video_path:
        return None

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 100
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_out.close()
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_out.name, fourcc, fps, (width, height))

    smoothed_preds = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        annotated_bgr, _, smoothed_preds = annotate_frame(frame, smoothed_preds=smoothed_preds)
        out.write(annotated_bgr)
        
        frame_idx += 1
        if frame_idx % 10 == 0:
            progress(min(1.0, frame_idx / total_frames), desc=f"Processing video frame {frame_idx}/{total_frames}")

    cap.release()
    out.release()
    return temp_out.name


# -------------------------------------------------------------
# 3. Gradio Interface Definition
# -------------------------------------------------------------
custom_css = """
#header { text-align: center; margin-bottom: 20px; }
"""

with gr.Blocks(title="EdgeVision — Real-Time Facial Emotion Recognition") as demo:
    gr.Markdown(
        """
        # 🎭 EdgeVision: Facial Emotion Recognition
        ### Real-Time Deep Learning using MiniXception (817 KB, 60+ FPS)
        
        [![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/shahin13700/fer-emotion-recognition)
        [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
        [![Model Size](https://img.shields.io/badge/Model%20Size-817%20KB-brightgreen)]()
        """
    )

    with gr.Tabs():
        # TAB 1: Live Streaming Webcam
        with gr.TabItem("🎥 Live Streaming Webcam"):
            gr.Markdown("Click the webcam feed below to start **continuous, real-time live emotion detection** right in your browser:")
            with gr.Row():
                with gr.Column(scale=1):
                    webcam_stream = gr.Image(sources=["webcam"], streaming=True, label="Live Camera Feed")
                with gr.Column(scale=1):
                    webcam_out = gr.Image(label="Live Emotion Detection Stream")
            
            webcam_stream.stream(
                fn=process_live_frame,
                inputs=webcam_stream,
                outputs=webcam_out
            )

        # TAB 2: Upload Video File
        with gr.TabItem("🎬 Upload Video File"):
            gr.Markdown("Upload any video clip (`.mp4`, `.mov`, `.avi`) to track emotions frame-by-frame with temporal smoothing:")
            with gr.Row():
                with gr.Column(scale=1):
                    video_input = gr.Video(label="Input Video")
                    video_btn = gr.Button("🚀 Process Video", variant="primary", size="lg")
                with gr.Column(scale=1):
                    video_output = gr.Video(label="Annotated Video with Emotion HUD")
            
            video_btn.click(
                fn=process_video_file,
                inputs=video_input,
                outputs=video_output
            )

        # TAB 3: Photo / Single Image Upload
        with gr.TabItem("🖼️ Photo / Image Upload"):
            gr.Markdown("Upload a photo or capture a snapshot for detailed probability bar breakdown:")
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(sources=["upload", "webcam"], type="numpy", label="Input Photo")
                    image_btn = gr.Button("🔍 Analyze Photo", variant="primary", size="lg")
                with gr.Column(scale=1):
                    image_output = gr.Image(label="Annotated Detection")
                    label_output = gr.Label(num_top_classes=7, label="Emotion Probabilities")

            image_btn.click(
                fn=process_image,
                inputs=image_input,
                outputs=[image_output, label_output]
            )
            image_input.change(
                fn=process_image,
                inputs=image_input,
                outputs=[image_output, label_output]
            )

    gr.Markdown(
        """
        ---
        ### ⚡ Technical Details
        - **Model:** MiniXception with Depthwise Separable Convolutions & Residual Connections
        - **Parameters:** ~60,000 (817 KB model file)
        - **Inference Speed:** Sub-10ms per face on CPU
        - **Repository:** [github.com/shahin13700/fer-emotion-recognition](https://github.com/shahin13700/fer-emotion-recognition)
        """
    )

if __name__ == '__main__':
    demo.launch(css=custom_css, share=False)
