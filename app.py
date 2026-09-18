"""
Facial Emotion Recognition (FER) - Interactive Multi-Modal Web Application
Includes:
1. 📸 Emotion Photo Booth (Tracks peak expressions, 7-emotion challenge, downloadable photo strip)
2. 🎥 Live Streaming Webcam (Clean face tracking + live side panel bars)
3. 🎬 Full Video File Processing (.mp4, .mov, .avi)
4. 🖼️ Still Image Analysis
5. 💾 Local Active Learning (personal dataset builder, append-only metadata.jsonl, benchmark fallbacks)
Powered by MiniXception & Gradio. Ready for Hugging Face Spaces (face saving is disabled there by default).
"""

import os
import json
import time
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
THRESHOLDS_PATH = 'outputs/empirical_thresholds.json'
USER_CONTRIB_DIR = 'dataset/user_contributed'
METADATA_JSONL = os.path.join(USER_CONTRIB_DIR, 'metadata.jsonl')

# Saving face crops writes biometric data to the disk of whatever machine runs this app.
# That is fine on your own laptop, but on a shared deployment (Hugging Face Spaces sets
# SPACE_ID) every visitor's faces would land on the server. Saving is therefore disabled
# automatically on Spaces unless EDGEVISION_ALLOW_SAVE=1 is set explicitly.
ALLOW_LOCAL_SAVE = os.environ.get("EDGEVISION_ALLOW_SAVE", "0" if os.environ.get("SPACE_ID") else "1") == "1"

# Uploaded-video limits: uploads are processed frame-by-frame on the CPU, so cap the work
# a single request can demand (roughly 60 s at 30 fps) and the upload size.
MAX_VIDEO_FRAMES = int(os.environ.get("EDGEVISION_MAX_VIDEO_FRAMES", "1800"))
MAX_UPLOAD_SIZE = os.environ.get("EDGEVISION_MAX_UPLOAD", "50mb")
VIDEO_OUT_DIR = os.path.join(tempfile.gettempdir(), "edgevision_videos")
VIDEO_OUT_TTL_SECONDS = 3600
# Gradio keeps its own copy of every uploaded and returned file; purge those on the same schedule.
GRADIO_CACHE_TTL = (VIDEO_OUT_TTL_SECONDS, VIDEO_OUT_TTL_SECONDS)

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

# Typical confidence per emotion: the 25th percentile of the model's confidence on
# CORRECT FER2013 test predictions (outputs/empirical_thresholds.json, produced by
# scripts/calc_percentiles.py). These are NOT capture gates; they are shown as a reference
# for how confident the model usually is when it is right, and used as the nominal score
# of benchmark fallback faces.
def load_empirical_thresholds(path=THRESHOLDS_PATH):
    fallback = {"Happy": 0.71, "Surprise": 0.69, "Neutral": 0.46, "Angry": 0.46,
                "Fear": 0.39, "Sad": 0.37, "Disgust": 0.71}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        loaded = {k: round(float(v["p25"]), 2) for k, v in data.items() if "p25" in v}
        if set(loaded) == set(fallback):
            return loaded
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return fallback


EMPIRICAL_THRESHOLDS = load_empirical_thresholds()

# Photo Booth capture floor: any emotion above 25% (vs. 14.3% for a random guess) is
# captured, and the slot is upgraded whenever a higher personal best comes along.
MIN_CAPTURE_FLOOR = 0.25

# Whole-image fallback in the photo tab is only for tiny pre-cropped faces
# (FER2013 samples are 48x48). Larger images without a detected face are NOT classified.
MAX_FALLBACK_SIZE = 128

print(f"Loading model from {MODEL_PATH}...")
model = load_model(MODEL_PATH)


@tf.function(input_signature=[tf.TensorSpec(shape=(None, 48, 48, 1), dtype=tf.float32)])
def _infer(batch):
    return model(batch, training=False)


def predict_probs(roi):
    """
    Runs the classifier on a (1, 48, 48, 1) float32 array and returns the 7 probabilities.
    Uses a traced tf.function: eager model(...) calls cost ~25 ms per face on a desktop CPU,
    the compiled graph ~2 ms, which is what makes real-time framerates possible.
    """
    return _infer(tf.convert_to_tensor(roi, dtype=tf.float32)).numpy()[0]


_ = predict_probs(np.zeros((1, 48, 48, 1), dtype=np.float32))  # warm-up / trace
print("Model initialized and ready.")

cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)


# -------------------------------------------------------------
# 2. State & Processing Helpers
# -------------------------------------------------------------
def empty_slot():
    return {
        "score": 0.0,
        "crop": None,           # Padded RGB crop for display (gallery / photo strip)
        "train_crop": None,     # Tight Haar-box grayscale crop: exactly what the model classified
        "is_sample": False,     # Benchmark fallback face rather than a live capture
        "label_source": None,   # "model_argmax" or "user_corrected"
        "saved": False,         # Already written to the local dataset (prevents duplicate saves)
    }


def init_booth_state():
    """Initializes empty session dictionary for the 7 emotion slots."""
    return {emotion: empty_slot() for emotion in emotion_labels}


def slot_caption(emotion, slot):
    """Gallery / strip caption. Sample faces carry no number: the model does not necessarily
    predict that emotion for them, so printing the reference score would read as a prediction."""
    if slot.get("is_sample"):
        return f"{emotion} (Sample)"
    fixed = " (relabelled)" if slot.get("label_source") == "user_corrected" else ""
    return f"{emotion}: {slot['score']*100:.0f}%{fixed}"


def gallery_items_from_state(booth_state):
    return [(booth_state[e]["crop"], slot_caption(e, booth_state[e]))
            for e in emotion_labels if booth_state[e]["crop"] is not None]


def challenge_status(booth_state):
    unlocked = [e for e in emotion_labels if booth_state[e]["score"] >= MIN_CAPTURE_FLOOR]
    missing = [e for e in emotion_labels if booth_state[e]["score"] < MIN_CAPTURE_FLOOR]
    if not missing:
        return "🎉 **CHALLENGE COMPLETE!** You unlocked all 7 emotions! Click **Generate Photo Strip** below!"
    return f"🎯 **Challenge:** {len(unlocked)} / 7 Emotions Captured! *(Next up: **{', '.join(missing)}** — make your best face!)*"


def get_local_dataset_stats():
    """
    Returns current count, per-class counts, and local active learning status banner.
    Prefers physical files in dataset/user_contributed/<emotion>/ for 100% parity with fine_tune.py,
    falling back to metadata.jsonl if directory folders are not yet populated.
    """
    per_class = {e: 0 for e in emotion_labels}
    has_physical_dirs = False

    if os.path.exists(USER_CONTRIB_DIR):
        for emotion in emotion_labels:
            emo_folder = os.path.join(USER_CONTRIB_DIR, emotion.lower())
            if os.path.isdir(emo_folder):
                valid_files = [f for f in os.listdir(emo_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if len(valid_files) > 0:
                    has_physical_dirs = True
                per_class[emotion] = len(valid_files)

    if has_physical_dirs:
        total_count = sum(per_class.values())
    else:
        # Fall back to metadata.jsonl for mock/test environments or unindexed setups
        total_count = 0
        if os.path.exists(METADATA_JSONL):
            try:
                with open(METADATA_JSONL, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                record = json.loads(line)
                                total_count += 1
                                emo = record.get("emotion", "").capitalize()
                                if emo in per_class:
                                    per_class[emo] += 1
                            except Exception:
                                pass
            except Exception:
                pass

    ready_classes = sum(1 for cnt in per_class.values() if cnt >= 10)
    if ALLOW_LOCAL_SAVE:
        status_md = (
            f"### 💾 Local Active Learning: Personal Dataset Builder\n"
            f"**`{total_count} Verified Faces Saved Locally`** • **`{ready_classes}/7 Classes Ready for Fine-Tuning (≥10 per class required)`**\n\n"
            f"*Save your expressions from the Photo Booth to build a personal dataset, then run `python fine_tune.py` to adapt the model to your camera!*"
        )
    else:
        status_md = (
            "### 💾 Local Active Learning: Personal Dataset Builder\n"
            "**Face saving is disabled on this shared deployment.** Nothing you capture here is written to disk. "
            "Run the app on your own machine (`python app.py`) to build a personal dataset and fine-tune."
        )
    return total_count, per_class, status_md

# Backward compatibility alias
get_community_counter_stats = get_local_dataset_stats



def annotate_frame(frame_bgr, smoothed_preds=None, alpha=0.70):
    """
    Detects faces on a single BGR frame, draws clean bounding box +
    top emotion label above the forehead, and returns confidences dict.
    Preserves input frame_bgr without in-place drawing mutation.
    """
    h, w, _ = frame_bgr.shape
    annotated = frame_bgr.copy()
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
        # Keep the smoother through detection dropouts (a raw, unsmoothed frame right after a
        # miss is exactly the spike EMA exists to suppress). Callers reset it explicitly.
        return annotated, confidences, smoothed_preds, []

    # Sort detected faces by area descending so primary face is index 0
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    detected_faces_info = []

    for idx, (x, y, fw, fh) in enumerate(faces):
        box_color = (0, 230, 255) # High-visibility Cyan
        cv2.rectangle(annotated, (x, y), (x + fw, y + fh), box_color, 3)

        roi_gray_full = gray[y:y+fh, x:x+fw]
        roi_gray = cv2.resize(roi_gray_full, (48, 48), interpolation=cv2.INTER_AREA)
        roi = roi_gray.astype('float32') / 255.0
        roi = np.expand_dims(np.expand_dims(roi, axis=0), axis=-1)

        raw_preds = predict_probs(roi)

        # Apply EMA smoothing ONLY to primary face (idx == 0) to avoid multi-person cross-talk
        if idx == 0:
            if smoothed_preds is None:
                smoothed_preds = raw_preds
            else:
                smoothed_preds = alpha * raw_preds + (1.0 - alpha) * smoothed_preds
            face_preds = smoothed_preds
            confidences = {emotion_labels[i]: float(face_preds[i]) for i in range(len(emotion_labels))}
        else:
            face_preds = raw_preds

        top_idx = int(np.argmax(face_preds))
        top_label = emotion_labels[top_idx]
        top_conf = face_preds[top_idx]

        # Clean, bold badge above forehead
        font_scale = max(0.85, fw / 180.0)
        font_thick = max(2, int(font_scale * 2.2))

        # Responsive feedback: Emerald Green when confident enough to capture, Cyan otherwise
        if top_conf >= MIN_CAPTURE_FLOOR:
            badge_color = (113, 204, 46) # Emerald Green in BGR
        else:
            badge_color = box_color # Cyan

        label_text = f"{top_label.upper()} {top_conf*100:.0f}%"

        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)

        badge_top = max(0, y - text_h - 16)
        badge_bottom = y
        cv2.rectangle(annotated, (x, badge_top), (x + text_w + 18, badge_bottom), badge_color, -1)
        cv2.putText(annotated, label_text, (x + 8, badge_bottom - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thick, cv2.LINE_AA)

        detected_faces_info.append({
            "box": (x, y, fw, fh),
            "label": top_label,
            "score": top_conf,
            "train_crop": roi_gray_full.copy()   # what the model actually saw, before the 48x48 resize
        })

    return annotated, confidences, smoothed_preds, detected_faces_info


# -------------------------------------------------------------
# 3. Photo Booth Logic & Active Learning Handlers
# -------------------------------------------------------------
def process_booth_frame(frame, booth_state, ema_state=None):
    """
    Analyzes frame in Photo Booth mode, tracks the primary face's peak expression
    per emotion (EMA-smoothed across frames), and updates the challenge counter.
    Returns (annotated_frame, status_md, gallery_items, booth_state, ema_state).
    """
    if frame is None:
        return None, gr.skip(), gr.skip(), booth_state, ema_state

    if booth_state is None:
        booth_state = init_booth_state()

    h, w, _ = frame.shape
    if w > 640:
        new_w = 640
        new_h = int(h * (640.0 / w))
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    clean_bgr = frame_bgr.copy()
    annotated_bgr, confs, ema_state, face_infos = annotate_frame(frame_bgr, smoothed_preds=ema_state)

    new_capture = False
    # Only the primary (largest) face can fill the booth: bystanders in the background must
    # not end up in your gallery or, worse, in your personal fine-tuning dataset.
    for info in face_infos[:1]:
        label = info["label"]
        score = info["score"]

        # Dynamic Peak Expression Tracker: record & upgrade whenever score exceeds noise floor
        # and beats your previous personal best. A benchmark sample never blocks a live capture.
        slot = booth_state[label]
        if score >= MIN_CAPTURE_FLOOR and (score > slot["score"] or slot["is_sample"]):
            x, y, fw, fh = info["box"]
            pad_x = int(fw * 0.25)
            pad_y = int(fh * 0.25)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(clean_bgr.shape[1], x + fw + pad_x)
            y2 = min(clean_bgr.shape[0], y + fh + pad_y)

            crop_bgr = clean_bgr[y1:y2, x1:x2]
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

            booth_state[label] = {
                **empty_slot(),
                "score": float(score),
                "crop": crop_rgb,
                "train_crop": info.get("train_crop"),
                "label_source": "model_argmax",
            }
            new_capture = True

    status_text = challenge_status(booth_state)
    gallery_items = gallery_items_from_state(booth_state)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    if new_capture:
        return annotated_rgb, status_text, gallery_items, booth_state, ema_state
    else:
        return annotated_rgb, gr.skip(), gr.skip(), booth_state, ema_state


def use_sample_face(emotion, booth_state):
    """
    UX Fail-Safe: Fills an emotion slot (e.g. Disgust or Fear) with a benchmark
    sample face from assets/samples/ so the user can complete their photo strip.
    """
    if booth_state is None:
        booth_state = init_booth_state()

    sample_path = os.path.join("assets", "samples", f"{emotion.lower()}.jpg")
    if os.path.exists(sample_path):
        img_bgr = cv2.imread(sample_path)
        # Smoothly upscale to standard portrait dimensions so it renders cleanly
        img_bgr = cv2.resize(img_bgr, (240, 240), interpolation=cv2.INTER_CUBIC)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # The slot counts as unlocked (score at the floor) but displays no number: the model
        # does not necessarily predict this emotion for the sample face.
        booth_state[emotion] = {**empty_slot(), "score": float(MIN_CAPTURE_FLOOR), "crop": img_rgb, "is_sample": True}

    return booth_state, challenge_status(booth_state), gallery_items_from_state(booth_state)


def relabel_slot(booth_state, from_emotion, to_emotion):
    """
    Lets the user correct the model's label for a captured face. Without this the local
    dataset is pure self-training on the model's own argmax; with it, saved records carry
    label_source="user_corrected". The corrected face moves to the target slot (overwriting
    whatever the model had put there) and the source slot is cleared.
    """
    if booth_state is None:
        booth_state = init_booth_state()
    if not from_emotion or not to_emotion or from_emotion == to_emotion:
        return booth_state, gr.skip(), gr.skip(), "ℹ️ Pick a captured slot and a *different* correct emotion."
    src = booth_state.get(from_emotion)
    if src is None or src["crop"] is None or src["is_sample"]:
        return booth_state, gr.skip(), gr.skip(), f"⚠️ **{from_emotion}** has no live capture to relabel."
    booth_state[to_emotion] = {**src, "label_source": "user_corrected", "saved": False}
    booth_state[from_emotion] = empty_slot()
    msg = f"✏️ Moved the face captured as **{from_emotion}** into the **{to_emotion}** slot (label corrected by you)."
    return booth_state, challenge_status(booth_state), gallery_items_from_state(booth_state), msg


def save_and_contribute(booth_state, consent_given):
    """
    Local Active Learning: Appends verified facial crops to dataset/user_contributed/
    and logs metadata using append-only JSON Lines format.
    """
    if not ALLOW_LOCAL_SAVE:
        return ("🔒 Saving is disabled on this shared deployment, so your faces were **not** written anywhere. "
                "Run `python app.py` on your own machine to build a personal dataset."), gr.skip()

    if not consent_given:
        return "⚠️ Please check the confirmation box to save your expressions locally.", gr.skip()

    if booth_state is None:
        return "⚠️ No expressions captured yet in this session.", gr.skip()

    saved_count = 0
    already_saved = 0
    os.makedirs(USER_CONTRIB_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    for emotion, data in booth_state.items():
        # Only save real captured faces (exclude benchmark fallback samples)
        if data["crop"] is None or data.get("is_sample") or data["score"] < MIN_CAPTURE_FLOOR:
            continue
        # Each capture is written once. Clicking Save again must not duplicate it, otherwise the
        # fine-tune volume gate and its held-out split can be satisfied with copies of one face.
        if data.get("saved"):
            already_saved += 1
            continue

        emo_dir = os.path.join(USER_CONTRIB_DIR, emotion.lower())
        os.makedirs(emo_dir, exist_ok=True)

        img_filename = f"{timestamp}_{emotion.lower()}_{saved_count}.png"
        img_path = os.path.join(emo_dir, img_filename)

        # Train on the tight grayscale box the model actually classified, not the padded
        # display crop; otherwise fine-tuning learns the padding instead of the face.
        train_crop = data.get("train_crop")
        if train_crop is None:
            train_crop = cv2.cvtColor(data["crop"], cv2.COLOR_RGB2GRAY)
        Image.fromarray(train_crop).save(img_path)

        log_entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "emotion": emotion.lower(),
            "score": round(float(data["score"]), 4),
            "image_path": img_path.replace("\\", "/"),
            "label_source": data.get("label_source") or "model_argmax",
            "consent_given": True
        }

        with open(METADATA_JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

        data["saved"] = True
        saved_count += 1

    _, _, updated_banner = get_local_dataset_stats()

    if saved_count > 0:
        feedback = f"✅ Saved **{saved_count}** new expression(s) locally to `dataset/user_contributed/`."
        if already_saved:
            feedback += f" ({already_saved} already saved earlier in this session were skipped.)"
    elif already_saved:
        feedback = "ℹ️ Everything captured in this session is already saved; capture new expressions to add more."
    else:
        feedback = "ℹ️ No new live facial expressions were ready to save (benchmark samples are excluded from fine-tuning)."

    return feedback, updated_banner



def _load_font(size):
    """A real TrueType font where one is available; PIL's bitmap default is 11 px."""
    candidates = [
        "DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "arial.ttf", "C:/Windows/Fonts/arial.ttf", "/System/Library/Fonts/Helvetica.ttc",
    ]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def generate_photo_strip(booth_state):
    """
    Renders a high-resolution 2x4 Photo Booth strip / collage image
    containing all captured peak expressions.
    """
    if booth_state is None:
        booth_state = init_booth_state()

    font_title = _load_font(22)
    font_body = _load_font(15)
    font_small = _load_font(13)

    tile_w, tile_h = 280, 280
    pad = 16
    header_h = 75
    total_w = (tile_w * 4) + (pad * 5)
    total_h = header_h + (tile_h * 2) + (pad * 3)

    canvas = Image.new('RGB', (total_w, total_h), color=(18, 22, 28))
    draw = ImageDraw.Draw(canvas)

    # Header
    draw.text((pad + 10, 14), "EDGEVISION EMOTION PHOTO BOOTH", fill=(0, 230, 255), font=font_title)
    date_str = datetime.datetime.now().strftime("%B %d, %Y - %H:%M")
    unlocked_count = sum(1 for e in emotion_labels if booth_state[e]["score"] >= MIN_CAPTURE_FLOOR)
    draw.text((pad + 10, 46), f"Session Highlights | Score: {unlocked_count}/7 Emotions Unlocked | {date_str}", fill=(180, 180, 180), font=font_body)

    # Render 7 Emotion Tiles + 1 Summary Tile in a 4x2 grid
    positions = [
        (0, 0), (1, 0), (2, 0), (3, 0), # Row 1: Happy, Surprise, Neutral, Sad
        (0, 1), (1, 1), (2, 1), (3, 1)  # Row 2: Angry, Fear, Disgust, Summary
    ]

    for i, emotion in enumerate(emotion_labels):
        col, row = positions[i]
        tx = pad + col * (tile_w + pad)
        ty = header_h + pad + row * (tile_h + pad)

        slot = booth_state[emotion]
        bg_col = (30, 36, 46)
        draw.rectangle([tx, ty, tx + tile_w, ty + tile_h], fill=bg_col, outline=(55, 65, 80), width=2)

        if slot["crop"] is not None:
            face_img = Image.fromarray(slot["crop"])
            face_img.thumbnail((tile_w - 8, tile_h - 45))
            ix = tx + (tile_w - face_img.width) // 2
            iy = ty + 6 + ((tile_h - 45) - face_img.height) // 2
            canvas.paste(face_img, (ix, iy))

            # Bottom Badge
            badge_color = EMOTION_COLORS_RGB.get(emotion, (0, 230, 255))
            draw.rectangle([tx, ty + tile_h - 36, tx + tile_w, ty + tile_h], fill=badge_color)
            draw.text((tx + 12, ty + tile_h - 28), slot_caption(emotion, slot).upper(), fill=(255, 255, 255), font=font_body)
        else:
            # Locked slot placeholder
            draw.text((tx + 30, ty + tile_h // 2 - 20), f"[LOCKED] {emotion.upper()}", fill=(120, 130, 145), font=font_body)
            draw.text((tx + 30, ty + tile_h // 2 + 6), "Not captured yet", fill=(80, 90, 105), font=font_small)

    # Slot 8: Summary / Signature Card
    scol, srow = positions[7]
    stx = pad + scol * (tile_w + pad)
    sty = header_h + pad + srow * (tile_h + pad)
    draw.rectangle([stx, sty, stx + tile_w, sty + tile_h], fill=(25, 32, 42), outline=(0, 230, 255), width=2)
    draw.text((stx + 20, sty + 36), "EDGEVISION AI", fill=(0, 230, 255), font=font_title)
    draw.text((stx + 20, sty + 82), f"Result: {unlocked_count} of 7", fill=(255, 255, 255), font=font_body)
    grade = "PERFECT!" if unlocked_count == 7 else "EXPRESSIVE!" if unlocked_count >= 5 else "NICE TRY!"
    draw.text((stx + 20, sty + 112), f"Grade: {grade}", fill=(46, 204, 113), font=font_body)
    draw.text((stx + 20, sty + 176), "MiniXception 51k params / 817 KB", fill=(150, 150, 150), font=font_small)
    draw.text((stx + 20, sty + 200), "Real-Time CPU Inference", fill=(100, 100, 100), font=font_small)

    return canvas


def reset_booth():
    """Resets the Photo Booth state, the EMA smoother, and clears the gallery."""
    new_state = init_booth_state()
    status_text = "🎯 **Challenge:** 0 / 7 Emotions Captured! Start the camera and make your best expressions!"
    return new_state, status_text, [], None


# -------------------------------------------------------------
# 4. Standard Video & Image Handlers
# -------------------------------------------------------------
def process_live_frame(frame, ema_state=None):
    """
    Clean stream mode: outputs the annotated frame, the side-panel probabilities, and the
    updated EMA state. The smoother lives in a per-session gr.State because each stream
    callback is otherwise stateless; without it the label flickers frame to frame.
    """
    if frame is None:
        return None, {}, ema_state
    h, w, _ = frame.shape
    if w > 640:
        new_w = 640
        new_h = int(h * (640.0 / w))
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    annotated_bgr, confs, ema_state, _ = annotate_frame(frame_bgr, smoothed_preds=ema_state)
    return cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB), confs, ema_state


def process_image(input_image):
    """
    Processes a single uploaded image.
    Returns (annotated_rgb, confidences, status_markdown).
    """
    if input_image is None:
        return None, {}, ""
    frame_bgr = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    annotated_bgr, confs, _, face_infos = annotate_frame(frame_bgr)
    status = f"✅ {len(face_infos)} face(s) detected." if face_infos else "❌ No face detected."

    # Fallback ONLY for tiny pre-cropped faces (e.g. FER2013 48x48 samples) where Haar cannot fire.
    # Larger images are not classified: the model has no "not a face" class and will happily
    # assign an emotion to a cat, a logo, or random noise.
    if len(confs) == 0:
        h, w, _ = frame_bgr.shape
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        aspect_ratio = w / float(h)
        if 0.65 <= aspect_ratio <= 1.55 and float(np.std(gray)) > 15.0 and max(h, w) <= MAX_FALLBACK_SIZE:
            roi_gray = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
            roi = roi_gray.astype('float32') / 255.0
            roi = np.expand_dims(np.expand_dims(roi, axis=0), axis=-1)
            raw_preds = predict_probs(roi)
            confs = {emotion_labels[i]: float(raw_preds[i]) for i in range(len(emotion_labels))}
            status = (f"⚠️ No face detected by the Haar cascade; the whole {w}×{h} image was treated as a "
                      f"pre-cropped face (fallback only applies to images ≤ {MAX_FALLBACK_SIZE}px).")

            top_idx = int(np.argmax(raw_preds))
            top_label = emotion_labels[top_idx]
            top_conf = raw_preds[top_idx]

            annotated_bgr = frame_bgr.copy()
            box_color = (0, 165, 255)  # Orange: fallback path, not a detection
            cv2.rectangle(annotated_bgr, (0, 0), (w - 1, h - 1), box_color, 2)
            label_text = f"{top_label.upper()} {top_conf*100:.0f}%"
            font_scale = max(0.45, w / 160.0)
            font_thick = 1 if font_scale < 0.6 else 2
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
            badge_top = max(0, h - th - 8)
            cv2.rectangle(annotated_bgr, (0, badge_top), (min(w, tw + 10), h), box_color, -1)
            cv2.putText(annotated_bgr, label_text, (4, h - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thick, cv2.LINE_AA)

    return cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB), confs, status


def predict_emotion(input_image):
    """Backward-compatible (annotated, confidences) wrapper used by CI smoke tests."""
    annotated, confs, _ = process_image(input_image)
    return annotated, confs


def _cleanup_old_videos(out_dir=VIDEO_OUT_DIR, ttl=VIDEO_OUT_TTL_SECONDS):
    """Deletes annotated videos older than ttl seconds so the temp dir does not grow forever."""
    try:
        now = time.time()
        for name in os.listdir(out_dir):
            path = os.path.join(out_dir, name)
            if os.path.isfile(path) and now - os.path.getmtime(path) > ttl:
                os.remove(path)
    except OSError:
        pass


def process_video_file(video_path, progress=gr.Progress()):
    """
    Processes an uploaded video file frame-by-frame and returns an annotated MP4.
    Work per request is bounded: frames are downscaled to <=640px wide and at most
    MAX_VIDEO_FRAMES are processed (the rest of the clip is dropped).
    """
    if not video_path:
        return None

    cap = None
    out = None
    temp_out = None

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise gr.Error("Could not open the uploaded video.")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 100
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if src_w <= 0 or src_h <= 0:
            raise gr.Error("Could not read the video dimensions.")
        scale = min(1.0, 640.0 / src_w)
        width, height = int(src_w * scale), int(src_h * scale)
        frames_to_process = min(total_frames, MAX_VIDEO_FRAMES)

        os.makedirs(VIDEO_OUT_DIR, exist_ok=True)
        _cleanup_old_videos()
        temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", dir=VIDEO_OUT_DIR)
        temp_out.close()

        # Try browser-friendly AVC/H.264 codecs first, fallback to mp4v
        codecs_to_try = ['avc1', 'H264', 'mp4v']
        out = None
        for c in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*c)
                w_candidate = cv2.VideoWriter(temp_out.name, fourcc, fps, (width, height))
                if w_candidate.isOpened():
                    out = w_candidate
                    break
                else:
                    w_candidate.release()
            except Exception:
                continue

        if out is None or not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_out.name, fourcc, fps, (width, height))

        smoothed_preds = None
        frame_idx = 0

        while frame_idx < MAX_VIDEO_FRAMES:
            ret, frame = cap.read()
            if not ret:
                break
            if scale < 1.0:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

            annotated_bgr, _, smoothed_preds, _ = annotate_frame(frame, smoothed_preds=smoothed_preds)
            out.write(annotated_bgr)

            frame_idx += 1
            if frame_idx % 10 == 0:
                progress(min(1.0, frame_idx / frames_to_process), desc=f"Processing video frame {frame_idx}/{frames_to_process}")

        if total_frames > MAX_VIDEO_FRAMES:
            gr.Warning(f"Only the first {MAX_VIDEO_FRAMES} frames were processed (upload limit).")

        return temp_out.name
    except Exception as e:
        if temp_out and os.path.exists(temp_out.name):
            try:
                os.remove(temp_out.name)
            except Exception:
                pass
        raise e
    finally:
        if cap is not None:
            cap.release()
        if out is not None:
            out.release()



# -------------------------------------------------------------
# 5. Gradio Interface Definition
# -------------------------------------------------------------
custom_css = """
#header { text-align: center; margin-bottom: 20px; }
.community-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 14px;
    margin-bottom: 12px;
}
"""

with gr.Blocks(title="EdgeVision — Real-Time Facial Emotion Recognition", delete_cache=GRADIO_CACHE_TTL) as demo:
    booth_state = gr.State(value=init_booth_state)
    booth_ema = gr.State(value=None)
    live_ema = gr.State(value=None)

    gr.Markdown(
        """
        # 🎭 EdgeVision: Facial Emotion Recognition & Photo Booth
        ### Real-Time Deep Learning using MiniXception (51k parameters, 817 KB, CPU-only)

        [![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/shahin13700/fer-emotion-recognition)
        [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
        [![Model Size](https://img.shields.io/badge/Model%20Size-817%20KB-brightgreen)]()
        """
    )

    with gr.Tabs():
        # TAB 1: 📸 Emotion Photo Booth & Active Learning
        with gr.TabItem("📸 Emotion Photo Booth"):
            # Dynamic Local Active Learning Status
            _, _, banner_text = get_local_dataset_stats()
            community_banner = gr.Markdown(banner_text, elem_classes=["community-card"])

            gr.Markdown(
                """
                ### 🎮 The 7-Emotion Photo Booth Challenge
                Turn on your camera and make your best facial expressions! Any emotion the model scores above **25%** is captured, and each slot is upgraded whenever you beat your personal best. Then compile your **Emotion Photo Strip**!
                """
            )
            challenge_badge = gr.Markdown("🎯 **Challenge:** 0 / 7 Emotions Captured! Start the camera and make your best faces!")

            with gr.Row():
                with gr.Column(scale=3):
                    booth_stream = gr.Image(sources=["webcam"], streaming=True, label="Live Camera")
                    with gr.Row():
                        reset_btn = gr.Button("🔄 Reset Session", variant="secondary", size="sm")

                    # UX Fail-Safe Section for Disgust & Fear
                    gr.Markdown("💡 **Can't trigger elusive expressions?** *(Camera angle and lighting can make subtle micro-expressions difficult)*")
                    with gr.Row():
                        disgust_sample_btn = gr.Button("🤢 Fill Disgust (Benchmark Face)", variant="secondary", size="sm")
                        fear_sample_btn = gr.Button("😱 Fill Fear (Benchmark Face)", variant="secondary", size="sm")

                    # Label correction: without it the saved dataset is pure self-training on the model's argmax
                    with gr.Accordion("✏️ Wrong label? Relabel a captured face", open=False):
                        gr.Markdown("Slots are labelled by the model's own top prediction. If it captured your *fear* face as *surprise*, move it here before saving.")
                        with gr.Row():
                            relabel_from = gr.Dropdown(choices=emotion_labels, label="Captured as", value=None)
                            relabel_to = gr.Dropdown(choices=emotion_labels, label="Actually", value=None)
                            relabel_btn = gr.Button("Move", size="sm")
                        relabel_feedback = gr.Markdown("")

                    # Helpful Emoji Tips Accordion
                    with gr.Accordion("🎭 Facial Action Unit (AU) Guide & Tips", open=False):
                        gr.Markdown(
                            "Capture floor is 25% for every emotion. The percentage after each emotion is the model's "
                            "*typical* confidence when it is right on FER2013 (25th percentile), so you know what a strong score looks like.\n\n"
                            + "\n".join([
                                f"- 😊 **Happy (typical: {EMPIRICAL_THRESHOLDS['Happy']*100:.0f}%):** Smile broadly, show teeth, and raise your cheeks.",
                                f"- 😲 **Surprise (typical: {EMPIRICAL_THRESHOLDS['Surprise']*100:.0f}%):** Drop jaw open wide and raise both eyebrows high.",
                                f"- 😐 **Neutral (typical: {EMPIRICAL_THRESHOLDS['Neutral']*100:.0f}%):** Relax all facial muscles with mouth gently closed.",
                                f"- 😡 **Angry (typical: {EMPIRICAL_THRESHOLDS['Angry']*100:.0f}%):** Furrow and pull eyebrows down/together, compress lips.",
                                f"- 😢 **Sad (typical: {EMPIRICAL_THRESHOLDS['Sad']*100:.0f}%):** Lower the corners of your mouth and cast gaze downward.",
                                f"- 😱 **Fear (typical: {EMPIRICAL_THRESHOLDS['Fear']*100:.0f}%):** Widen eyes, raise inner eyebrows, pull head back slightly.",
                                f"- 🤢 **Disgust (typical: {EMPIRICAL_THRESHOLDS['Disgust']*100:.0f}%):** Wrinkle the bridge of your nose and raise upper lip.",
                            ])
                        )

                with gr.Column(scale=3):
                    gallery_output = gr.Gallery(label="✨ Unlocked Emotion Portraits", columns=4, height="auto")
                    generate_strip_btn = gr.Button("📸 Generate My Emotion Photo Strip", variant="primary", size="lg")
                    strip_output = gr.Image(label="Your Downloadable Emotion Photo Strip")

                    # Active Learning Contribution Box
                    with gr.Group():
                        gr.Markdown("#### 💾 Personal Dataset Builder for Fine-Tuning")
                        gr.Markdown(
                            "🔒 **Privacy:** face crops are written to `dataset/user_contributed/` on the machine running this app "
                            "and never uploaded anywhere by EdgeVision. Delete that folder to erase them. "
                            "Labels are the model's own predictions unless you relabel them above; each capture is saved once. "
                            + ("" if ALLOW_LOCAL_SAVE else "**Saving is disabled on this shared deployment.**")
                        )
                        consent_box = gr.Checkbox(
                            label="I understand my face crops will be saved to disk on this machine for fine_tune.py",
                            value=False,
                            interactive=ALLOW_LOCAL_SAVE
                        )
                        contribute_btn = gr.Button("💾 Save Captured Faces to Local Dataset", variant="secondary", interactive=ALLOW_LOCAL_SAVE)
                        contribute_feedback = gr.Markdown("")


            # Event bindings
            booth_stream.stream(
                fn=process_booth_frame,
                inputs=[booth_stream, booth_state, booth_ema],
                outputs=[booth_stream, challenge_badge, gallery_output, booth_state, booth_ema]
            )

            disgust_sample_btn.click(
                fn=lambda s: use_sample_face("Disgust", s),
                inputs=[booth_state],
                outputs=[booth_state, challenge_badge, gallery_output]
            )

            fear_sample_btn.click(
                fn=lambda s: use_sample_face("Fear", s),
                inputs=[booth_state],
                outputs=[booth_state, challenge_badge, gallery_output]
            )

            relabel_btn.click(
                fn=relabel_slot,
                inputs=[booth_state, relabel_from, relabel_to],
                outputs=[booth_state, challenge_badge, gallery_output, relabel_feedback]
            )

            generate_strip_btn.click(
                fn=generate_photo_strip,
                inputs=booth_state,
                outputs=strip_output
            )

            reset_btn.click(
                fn=reset_booth,
                inputs=[],
                outputs=[booth_state, challenge_badge, gallery_output, booth_ema]
            )

            contribute_btn.click(
                fn=save_and_contribute,
                inputs=[booth_state, consent_box],
                outputs=[contribute_feedback, community_banner]
            )

        # TAB 2: Live Streaming Webcam (Clean Stream)
        with gr.TabItem("🎥 Live Streaming Webcam"):
            gr.Markdown("Continuous, real-time emotion recognition with EMA temporal smoothing and probability bars in the sidebar:")
            with gr.Row():
                with gr.Column(scale=3):
                    webcam_stream = gr.Image(sources=["webcam"], streaming=True, label="Live Camera Input")
                    webcam_out = gr.Image(label="Live Tracking Video")
                with gr.Column(scale=2):
                    live_labels = gr.Label(num_top_classes=7, label="📊 Live Emotion Probabilities")

            webcam_stream.stream(
                fn=process_live_frame,
                inputs=[webcam_stream, live_ema],
                outputs=[webcam_out, live_labels, live_ema]
            )

        # TAB 3: Upload Video File
        with gr.TabItem("🎬 Upload Video File"):
            gr.Markdown(f"Upload a video clip (`.mp4`, `.mov`, `.avi`, up to {MAX_UPLOAD_SIZE}) to track emotions frame-by-frame with temporal smoothing. Clips are downscaled to 640px and the first {MAX_VIDEO_FRAMES} frames (~{MAX_VIDEO_FRAMES // 30} s at 30 fps) are processed:")
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
                    image_status = gr.Markdown("")
                    label_output = gr.Label(num_top_classes=7, label="Emotion Probabilities")

            image_btn.click(
                fn=process_image,
                inputs=image_input,
                outputs=[image_output, label_output, image_status]
            )
            image_input.change(
                fn=process_image,
                inputs=image_input,
                outputs=[image_output, label_output, image_status]
            )

    gr.Markdown(
        """
        ---
        ### ⚡ Technical Details
        - **Model:** MiniXception with Depthwise Separable Convolutions & Residual Connections
        - **Parameters:** 51,255 (817 KB model file) · 57.3% accuracy on the FER2013 test set
        - **Inference:** ~2 ms per face on a desktop CPU (graph-compiled), plus Haar face detection
        - **Active Learning:** Local, opt-in personal dataset (model-labelled unless you relabel) with append-only JSONL logging
        - **Repository:** [github.com/shahin13700/fer-emotion-recognition](https://github.com/shahin13700/fer-emotion-recognition)
        """
    )

if __name__ == '__main__':
    demo.launch(css=custom_css, share=False, max_file_size=MAX_UPLOAD_SIZE)
