"""
Facial Emotion Recognition (FER) - Interactive Multi-Modal Web Application
Includes:
1. 📸 Emotion Photo Booth (Tracks peak expressions, 7-emotion challenge, downloadable photo strip)
2. 🎥 Live Streaming Webcam (Clean face tracking + live side panel bars)
3. 🎬 Full Video File Processing (.mp4, .mov, .avi)
4. 🖼️ Still Image Analysis
Powered by MiniXception & Gradio. Ready for Hugging Face Spaces.
"""

import os
import json
import tempfile
import datetime
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
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

# Color mappings for emotions (BGR for OpenCV, RGB for PIL)
EMOTION_COLORS_RGB = {
    "Happy": (46, 204, 113),      # Emerald Green
    "Surprise": (52, 152, 219),   # Sky Blue
    "Neutral": (149, 165, 166),   # Silver Gray
    "Sad": (41, 128, 185),        # Deep Blue
    "Angry": (231, 76, 60),       # Coral Red
    "Fear": (155, 89, 182),       # Amethyst Purple
    "Disgust": (230, 126, 34),    # Amber Orange
}

print(f"Loading model from {MODEL_PATH}...")
model = load_model(MODEL_PATH)
_ = model(tf.zeros((1, 48, 48, 1)), training=False)
print("Model initialized and ready.")

cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)


# -------------------------------------------------------------
# 2. State & Processing Helpers
# -------------------------------------------------------------
def init_booth_state():
    """Initializes empty session dictionary for the 7 emotion slots."""
    return {
        emotion: {
            "score": 0.0,
            "crop": None  # RGB numpy array
        }
        for emotion in emotion_labels
    }


def annotate_frame(frame_bgr, smoothed_preds=None, alpha=0.70):
    """
    Detects faces on a single BGR frame, draws clean bounding box +
    top emotion label above the forehead, and returns confidences dict.
    """
    h, w, _ = frame_bgr.shape
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

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
        return frame_bgr, confidences, None, []

    detected_faces_info = []

    for (x, y, fw, fh) in faces:
        box_color = (0, 230, 255) # High-visibility Cyan
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

        # Clean, bold badge above forehead
        font_scale = max(0.85, fw / 180.0)
        font_thick = max(2, int(font_scale * 2.2))
        label_text = f"{top_label.upper()} {top_conf*100:.0f}%"

        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
        
        badge_top = max(0, y - text_h - 16)
        badge_bottom = y
        cv2.rectangle(frame_bgr, (x, badge_top), (x + text_w + 18, badge_bottom), box_color, -1)
        cv2.putText(frame_bgr, label_text, (x + 8, badge_bottom - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thick, cv2.LINE_AA)

        detected_faces_info.append({
            "box": (x, y, fw, fh),
            "label": top_label,
            "score": top_conf
        })

    return frame_bgr, confidences, smoothed_preds, detected_faces_info


# -------------------------------------------------------------
# 3. Photo Booth Logic & Collage Generator
# -------------------------------------------------------------
def process_booth_frame(frame, booth_state):
    """
    Analyzes frame in Photo Booth mode, saves new peak expression crops,
    and updates the challenge counter.
    """
    if frame is None:
        return None, gr.skip(), gr.skip(), booth_state

    if booth_state is None:
        booth_state = init_booth_state()

    h, w, _ = frame.shape
    if w > 640:
        new_w = 640
        new_h = int(h * (640.0 / w))
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    annotated_bgr, confs, _, face_infos = annotate_frame(frame_bgr)

    new_capture = False
    for info in face_infos:
        label = info["label"]
        score = info["score"]
        # Only record if score is confident (>= 40%) and beats previous personal best
        if score >= 0.40 and score > booth_state[label]["score"]:
            x, y, fw, fh = info["box"]
            pad_x = int(fw * 0.25)
            pad_y = int(fh * 0.25)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(frame_bgr.shape[1], x + fw + pad_x)
            y2 = min(frame_bgr.shape[0], y + fh + pad_y)
            
            crop_bgr = frame_bgr[y1:y2, x1:x2]
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            
            booth_state[label] = {
                "score": float(score),
                "crop": crop_rgb
            }
            new_capture = True

    # Challenge counter text
    unlocked = [e for e in emotion_labels if booth_state[e]["score"] >= 0.40]
    missing = [e for e in emotion_labels if booth_state[e]["score"] < 0.40]

    if len(missing) == 0:
        status_text = "🎉 **CHALLENGE COMPLETE!** You unlocked all 7 emotions! Click **Generate Photo Strip** below!"
    else:
        missing_str = ", ".join(missing)
        status_text = f"🎯 **Challenge:** {len(unlocked)} / 7 Emotions Captured! *(Try making a face for: **{missing_str}**)*"

    # Format gallery items
    gallery_items = []
    for e in emotion_labels:
        if booth_state[e]["crop"] is not None:
            gallery_items.append((booth_state[e]["crop"], f"{e}: {booth_state[e]['score']*100:.0f}%"))

    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    if new_capture:
        return annotated_rgb, status_text, gallery_items, booth_state
    else:
        return annotated_rgb, gr.skip(), gr.skip(), booth_state


def generate_photo_strip(booth_state):
    """
    Renders a high-resolution 2x4 Photo Booth strip / collage image
    containing all captured peak expressions.
    """
    if booth_state is None:
        booth_state = init_booth_state()

    tile_w, tile_h = 280, 280
    pad = 16
    header_h = 75
    total_w = (tile_w * 4) + (pad * 5)
    total_h = header_h + (tile_h * 2) + (pad * 3)

    canvas = Image.new('RGB', (total_w, total_h), color=(18, 22, 28))
    draw = ImageDraw.Draw(canvas)

    # Header
    draw.text((pad + 10, 18), "🎭 EDGEVISION EMOTION PHOTO BOOTH", fill=(0, 230, 255))
    date_str = datetime.datetime.now().strftime("%B %d, %Y • %H:%M")
    unlocked_count = sum(1 for e in emotion_labels if booth_state[e]["score"] >= 0.40)
    draw.text((pad + 10, 44), f"Session Highlights • Score: {unlocked_count}/7 Emotions Unlocked • {date_str}", fill=(180, 180, 180))

    # Render 7 Emotion Tiles + 1 Summary Tile in a 4x2 grid
    positions = [
        (0, 0), (1, 0), (2, 0), (3, 0), # Row 1: Happy, Surprise, Neutral, Sad
        (0, 1), (1, 1), (2, 1), (3, 1)  # Row 2: Angry, Fear, Disgust, Summary
    ]

    for idx, emotion in enumerate(emotion_labels):
        col, row = positions[idx]
        tx = pad + col * (tile_w + pad)
        ty = header_h + pad + row * (tile_h + pad)

        slot = booth_state[emotion]
        bg_col = (30, 36, 46)
        draw.rectangle([tx, ty, tx + tile_w, ty + tile_h], fill=bg_col, outline=(55, 65, 80), width=2)

        if slot["crop"] is not None:
            # Resize crop to fill tile while maintaining aspect ratio
            face_img = Image.fromarray(slot["crop"])
            face_img.thumbnail((tile_w - 8, tile_h - 45))
            # Center image
            ix = tx + (tile_w - face_img.width) // 2
            iy = ty + 6 + ((tile_h - 45) - face_img.height) // 2
            canvas.paste(face_img, (ix, iy))

            # Bottom Badge
            badge_color = EMOTION_COLORS_RGB.get(emotion, (0, 230, 255))
            draw.rectangle([tx, ty + tile_h - 36, tx + tile_w, ty + tile_h], fill=badge_color)
            label_caption = f"{emotion.upper()}: {slot['score']*100:.0f}%"
            draw.text((tx + 12, ty + tile_h - 26), label_caption, fill=(255, 255, 255))
        else:
            # Locked slot placeholder
            draw.text((tx + 30, ty + tile_h // 2 - 20), f"🔒 {emotion.upper()}", fill=(120, 130, 145))
            draw.text((tx + 30, ty + tile_h // 2 + 6), "Not captured yet", fill=(80, 90, 105))

    # Slot 8: Summary / Signature Card
    scol, srow = positions[7]
    stx = pad + scol * (tile_w + pad)
    sty = header_h + pad + srow * (tile_h + pad)
    draw.rectangle([stx, sty, stx + tile_w, sty + tile_h], fill=(25, 32, 42), outline=(0, 230, 255), width=2)
    draw.text((stx + 20, sty + 40), "EDGEVISION AI", fill=(0, 230, 255))
    draw.text((stx + 20, sty + 80), f"Result: {unlocked_count} of 7", fill=(255, 255, 255))
    grade = "PERFECT!" if unlocked_count == 7 else "EXPRESSIVE!" if unlocked_count >= 5 else "NICE TRY!"
    draw.text((stx + 20, sty + 115), f"Grade: {grade}", fill=(46, 204, 113))
    draw.text((stx + 20, sty + 170), "MiniXception 817 KB", fill=(150, 150, 150))
    draw.text((stx + 20, sty + 195), "60+ FPS Real-Time Edge", fill=(100, 100, 100))

    temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    temp_out.close()
    canvas.save(temp_out.name, "PNG")
    return temp_out.name


def reset_booth():
    """Resets the Photo Booth state and clear gallery."""
    new_state = init_booth_state()
    status_text = "🎯 **Challenge:** 0 / 7 Emotions Captured! Start the camera and make your best expressions!"
    return new_state, status_text, []


# -------------------------------------------------------------
# 4. Standard Video & Image Handlers
# -------------------------------------------------------------
def process_live_frame(frame):
    """Clean stream mode: outputs frame and updates side panel."""
    if frame is None:
        return None, {}
    h, w, _ = frame.shape
    if w > 640:
        new_w = 640
        new_h = int(h * (640.0 / w))
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    annotated_bgr, confs, _, _ = annotate_frame(frame_bgr)
    return cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB), confs


def process_image(input_image):
    """Processes a single uploaded image."""
    if input_image is None:
        return None, {}
    frame_bgr = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    annotated_bgr, confs, _, _ = annotate_frame(frame_bgr)
    return cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB), confs


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
            
        annotated_bgr, _, smoothed_preds, _ = annotate_frame(frame, smoothed_preds=smoothed_preds)
        out.write(annotated_bgr)
        
        frame_idx += 1
        if frame_idx % 10 == 0:
            progress(min(1.0, frame_idx / total_frames), desc=f"Processing video frame {frame_idx}/{total_frames}")

    cap.release()
    out.release()
    return temp_out.name


# -------------------------------------------------------------
# 5. Gradio Interface Definition
# -------------------------------------------------------------
custom_css = """
#header { text-align: center; margin-bottom: 20px; }
"""

with gr.Blocks(title="EdgeVision — Real-Time Facial Emotion Recognition") as demo:
    booth_state = gr.State(value=init_booth_state)

    gr.Markdown(
        """
        # 🎭 EdgeVision: Facial Emotion Recognition & Photo Booth
        ### Real-Time Deep Learning using MiniXception (817 KB, 60+ FPS on CPU)
        
        [![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/shahin13700/fer-emotion-recognition)
        [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
        [![Model Size](https://img.shields.io/badge/Model%20Size-817%20KB-brightgreen)]()
        """
    )

    with gr.Tabs():
        # TAB 1: 📸 Emotion Photo Booth (New Gamified Feature!)
        with gr.TabItem("📸 Emotion Photo Booth"):
            gr.Markdown(
                """
                ### 🎮 The 7-Emotion Photo Booth Challenge
                Turn on your camera and make your best facial expressions! The AI tracks your peak moments in real time, collects your highest-confidence portraits, and generates a downloadable **Emotion Photo Strip**!
                """
            )
            challenge_badge = gr.Markdown("🎯 **Challenge:** 0 / 7 Emotions Captured! Start the camera and make your best faces!")

            with gr.Row():
                with gr.Column(scale=3):
                    booth_stream = gr.Image(sources=["webcam"], streaming=True, label="Live Camera")
                    reset_btn = gr.Button("🔄 Reset Session", variant="secondary", size="sm")

                with gr.Column(scale=3):
                    gallery_output = gr.Gallery(label="✨ Unlocked Emotion Portraits", columns=4, height="auto")
                    generate_strip_btn = gr.Button("📸 Generate My Emotion Photo Strip", variant="primary", size="lg")
                    strip_output = gr.Image(label="Your Downloadable Emotion Photo Strip")

            booth_stream.stream(
                fn=process_booth_frame,
                inputs=[booth_stream, booth_state],
                outputs=[booth_stream, challenge_badge, gallery_output, booth_state]
            )

            generate_strip_btn.click(
                fn=generate_photo_strip,
                inputs=booth_state,
                outputs=strip_output
            )

            reset_btn.click(
                fn=reset_booth,
                inputs=[],
                outputs=[booth_state, challenge_badge, gallery_output]
            )

        # TAB 2: Live Streaming Webcam (Clean Stream)
        with gr.TabItem("🎥 Live Streaming Webcam"):
            gr.Markdown("Continuous, real-time emotion recognition with probability bars in the sidebar:")
            with gr.Row():
                with gr.Column(scale=3):
                    webcam_stream = gr.Image(sources=["webcam"], streaming=True, label="Live Camera Input")
                    webcam_out = gr.Image(label="Live Tracking Video")
                with gr.Column(scale=2):
                    live_labels = gr.Label(num_top_classes=7, label="📊 Live Emotion Probabilities")
            
            webcam_stream.stream(
                fn=process_live_frame,
                inputs=webcam_stream,
                outputs=[webcam_out, live_labels]
            )

        # TAB 3: Upload Video File
        with gr.TabItem("🎬 Upload Video File"):
            gr.Markdown("Upload any video clip (`.mp4`, `.mov`, `.avi`) to track emotions frame-by-frame with temporal smoothing:")
            with gr.Row():
                with gr.Column(scale=1):
                    video_input = gr.Video(label="Input Video")
                    video_btn = gr.Button("🚀 Process Video", variant="primary", size="lg")
                with gr.Column(scale=1):
                    video_output = gr.Video(label="Annotated Video with Emotion Tracking")
            
            video_btn.click(
                fn=process_video_file,
                inputs=video_input,
                outputs=video_output
            )

        # TAB 4: Photo / Single Image Upload
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
