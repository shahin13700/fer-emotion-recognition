"""
Facial Emotion Recognition (FER) - Interactive Multi-Modal Web Application
Optimized for high-speed live streaming & large readable HUD overlays.
Supports:
1. Live Streaming Webcam Video (Fast & Big HUD)
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
# 2. Core Processing with Fast Detection & Big Readable HUD
# -------------------------------------------------------------
def annotate_frame(frame_bgr, smoothed_preds=None, alpha=0.70):
    """
    Detects faces on a single BGR frame with 4x downsampled speedup,
    predicts emotion, and renders LARGE, high-contrast badges & probability HUD.
    """
    h, w, _ = frame_bgr.shape
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # 4x Speedup: Downsample high-res frames for face detection
    scale_factor = 2 if max(h, w) > 480 else 1
    if scale_factor > 1:
        small_gray = cv2.resize(gray, (w // scale_factor, h // scale_factor))
    else:
        small_gray = gray

    faces = face_cascade.detectMultiScale(
        small_gray,
        scaleFactor=1.25,
        minNeighbors=4,
        minSize=(25, 25)
    )

    if scale_factor > 1:
        faces = [(fx * scale_factor, fy * scale_factor, fw * scale_factor, fh * scale_factor) for (fx, fy, fw, fh) in faces]

    confidences = {}
    if len(faces) == 0:
        return frame_bgr, confidences, None

    for (x, y, fw, fh) in faces:
        box_color = (0, 220, 255) # High-visibility Amber/Cyan
        cv2.rectangle(frame_bgr, (x, y), (x + fw, y + fh), box_color, 3)

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

        # -------------------------------------------------------------
        # 1. BIG, BOLD BADGE ABOVE FACE (Easily readable)
        # -------------------------------------------------------------
        font_scale = max(0.9, fw / 150.0)
        font_thick = max(2, int(font_scale * 2.2))
        label_text = f"{top_label.upper()} {top_conf*100:.0f}%"

        (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
        
        badge_top = max(0, y - text_h - 18)
        badge_bottom = y
        # Solid backdrop
        cv2.rectangle(frame_bgr, (x, badge_top), (x + text_w + 20, badge_bottom), box_color, -1)
        # Bold black text on bright background
        cv2.putText(frame_bgr, label_text, (x + 10, badge_bottom - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thick, cv2.LINE_AA)

        # -------------------------------------------------------------
        # 2. LARGE PROBABILITY HUD (Clear, full names, visible bars)
        # -------------------------------------------------------------
        hud_w = 270
        hud_h = 245
        
        # Position to the right of face if space allows, otherwise top-right corner
        if x + fw + hud_w + 15 < w:
            hud_x = x + fw + 15
            hud_y = max(10, y)
        else:
            hud_x = max(10, w - hud_w - 15)
            hud_y = 15

        # Draw dark translucent card
        overlay = frame_bgr.copy()
        cv2.rectangle(overlay, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (12, 12, 12), -1)
        cv2.addWeighted(overlay, 0.88, frame_bgr, 0.12, 0, frame_bgr)
        cv2.rectangle(frame_bgr, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (60, 60, 60), 1)

        # Title
        cv2.putText(frame_bgr, "EMOTION METRICS", (hud_x + 14, hud_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 1, cv2.LINE_AA)

        for i, prob in enumerate(smoothed_preds):
            y_pos = hud_y + 48 + (i * 26)
            is_top = (i == top_idx)
            
            lbl_color = (0, 255, 128) if is_top else (210, 210, 210)
            lbl_font = 0.52 if is_top else 0.45
            lbl_thick = 2 if is_top else 1
            
            # Full emotion name
            cv2.putText(frame_bgr, f"{emotion_labels[i][:7]}", (hud_x + 14, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, lbl_font, lbl_color, lbl_thick, cv2.LINE_AA)
            
            # Progress bar
            bar_x = hud_x + 95
            bar_max_w = 110
            bar_w = int(prob * bar_max_w)
            
            cv2.rectangle(frame_bgr, (bar_x, y_pos - 11), (bar_x + bar_max_w, y_pos + 1), (40, 40, 40), -1)
            bar_col = (0, 220, 255) if is_top else (150, 150, 150)
            if bar_w > 0:
                cv2.rectangle(frame_bgr, (bar_x, y_pos - 11), (bar_x + bar_w, y_pos + 1), bar_col, -1)
            
            cv2.putText(frame_bgr, f"{prob*100:.0f}%", (bar_x + bar_max_w + 8, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, lbl_color, 1, cv2.LINE_AA)

    return frame_bgr, confidences, smoothed_preds


def process_image(input_image):
    """Processes a single uploaded image or snapshot."""
    if input_image is None:
        return None, {}
    frame_bgr = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    annotated_bgr, confs, _ = annotate_frame(frame_bgr)
    return cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB), confs


def process_live_frame(frame):
    """
    Processes real-time streaming webcam frames in browser.
    Resizes incoming frames to standard 640px width to cut WebSocket latency by 4x.
    """
    if frame is None:
        return None

    h, w, _ = frame.shape
    # Downsample high-res webcam frames to 640px width for fast WebSocket roundtrip
    if w > 640:
        new_w = 640
        new_h = int(h * (640.0 / w))
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

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
