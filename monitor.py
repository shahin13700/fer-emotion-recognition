"""
Edge-Vision: Real-Time Facial Emotion & Driver Drowsiness Guard
Combines the MediaPipe Face Landmarker (Tasks API, 478 landmarks) with the MiniXception CNN.
Tracks 7 emotion categories and computes Eye Aspect Ratio (EAR) to detect driver fatigue.

The landmark model (face_landmarker.task, ~3.7 MB, Apache-2.0, published by Google) is not
bundled; it is downloaded to model/ on first run or read from EDGEVISION_LANDMARKER_PATH.
"""

import os
import json
import time
import urllib.request
import warnings
warnings.filterwarnings('ignore')
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# -------------------------------------------------------------
# 1. Landmark Constants for Eye Aspect Ratio (EAR)
# -------------------------------------------------------------
# Eye landmark indices (MediaPipe face mesh topology; unchanged in the Tasks API)
LEFT_EYE_POINTS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_POINTS = [362, 385, 387, 263, 373, 380]

# EAR Thresholds
EAR_THRESHOLD = 0.21         # Eye is considered closed below this ratio
DROWSY_CONSEC_FRAMES = 15    # Consecutive closed-eye frames before alerting (~0.5 s at 30 fps)
EMA_RESET_AFTER_MISSES = 30  # Frames without a face before the emotion smoother is reset (~1 s)

LANDMARKER_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                  "face_landmarker/float16/1/face_landmarker.task")
LANDMARKER_PATH = os.environ.get("EDGEVISION_LANDMARKER_PATH", os.path.join("model", "face_landmarker.task"))


def ensure_landmarker_model(path=LANDMARKER_PATH, url=LANDMARKER_URL):
    """Returns the path to face_landmarker.task, downloading it once if it is missing."""
    if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
        return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    print(f"Downloading MediaPipe Face Landmarker model (~3.7 MB) to {path} ...")
    tmp = path + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, path)
    return path


def create_landmarker(path=None, running_mode=mp_vision.RunningMode.VIDEO, num_faces=1):
    """Builds a MediaPipe FaceLandmarker (Tasks API replacement for the removed mp.solutions.face_mesh)."""
    path = ensure_landmarker_model(path or LANDMARKER_PATH)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=path),
        running_mode=running_mode,
        num_faces=num_faces,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


def detect_landmarks(landmarker, rgb_frame, timestamp_ms):
    """Runs the landmarker on an RGB frame; returns the list of landmark lists (one per face)."""
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_frame))
    result = landmarker.detect_for_video(mp_image, int(timestamp_ms))
    return result.face_landmarks


def calculate_ear(landmarks, eye_indices, img_w, img_h):
    """
    Computes Eye Aspect Ratio (EAR) for a given eye using 6 facial landmarks.
    EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)
    `landmarks` is any sequence of objects with normalized .x / .y attributes.
    """
    pts = [np.array([landmarks[idx].x * img_w, landmarks[idx].y * img_h]) for idx in eye_indices]

    # Vertical Euclidean distances
    dist_v1 = np.linalg.norm(pts[1] - pts[5])
    dist_v2 = np.linalg.norm(pts[2] - pts[4])
    # Horizontal Euclidean distance
    dist_h = np.linalg.norm(pts[0] - pts[3])

    if dist_h == 0:
        return 0.3
    ear = (dist_v1 + dist_v2) / (2.0 * dist_h)
    return ear


def average_ear(landmarks, img_w, img_h):
    """Mean EAR over both eyes for one face."""
    left = calculate_ear(landmarks, LEFT_EYE_POINTS, img_w, img_h)
    right = calculate_ear(landmarks, RIGHT_EYE_POINTS, img_w, img_h)
    return (left + right) / 2.0


def main():
    print("===============================================================")
    print("  EdgeVision: Real-Time Emotion & Drowsiness Guard")
    print("  Powered by MediaPipe Face Landmarker + MiniXception")
    print("===============================================================")

    # 1. Load Emotion Model & Labels
    model_path = 'model/emotion_model.keras'
    indices_path = 'outputs/class_indices.json'

    if not os.path.exists(model_path) or not os.path.exists(indices_path):
        print("Error: Model or class indices missing.")
        return

    with open(indices_path, 'r') as f:
        class_indices = json.load(f)
    idx_to_class = {v: k.capitalize() for k, v in class_indices.items()}
    emotion_labels = [idx_to_class[i] for i in range(len(idx_to_class))]

    print(f"Loading emotion model from {model_path}...")
    model = load_model(model_path)

    # Graph-compiled forward pass (~2 ms/face vs ~25 ms eager on a desktop CPU)
    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 48, 48, 1), dtype=tf.float32)])
    def infer(batch):
        return model(batch, training=False)

    _ = infer(tf.zeros((1, 48, 48, 1)))  # Warmup / trace

    # 2. Initialize MediaPipe Face Landmarker
    landmarker = create_landmarker()

    # 3. Setup Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access webcam.")
        landmarker.close()
        return

    print("\n[ACTIVE] Webcam initialized. Press 'Q' to quit anytime.")

    # State variables
    drowsy_counter = 0
    smoothed_preds = None
    missed_frames = 0
    alpha = 0.65  # EMA smoothing factor
    prev_time = time.time()
    start_time = prev_time

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Calculate FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 30.0
        prev_time = curr_time

        # Landmark inference (VIDEO mode needs monotonically increasing timestamps)
        faces = detect_landmarks(landmarker, rgb_frame, (curr_time - start_time) * 1000.0)
        avg_ear = None

        if faces:
            missed_frames = 0
            for landmarks in faces:
                # -----------------------------------------------------
                # A. Eye Aspect Ratio & Drowsiness Tracking
                # -----------------------------------------------------
                avg_ear = average_ear(landmarks, w, h)

                is_drowsy = False
                if avg_ear < EAR_THRESHOLD:
                    drowsy_counter += 1
                    if drowsy_counter >= DROWSY_CONSEC_FRAMES:
                        is_drowsy = True
                else:
                    drowsy_counter = max(0, drowsy_counter - 1)

                # -----------------------------------------------------
                # B. Crop Face for Emotion Recognition
                # -----------------------------------------------------
                x_coords = [int(lm.x * w) for lm in landmarks]
                y_coords = [int(lm.y * h) for lm in landmarks]

                xmin, xmax = max(0, min(x_coords)), min(w, max(x_coords))
                ymin, ymax = max(0, min(y_coords)), min(h, max(y_coords))

                face_w = xmax - xmin
                face_h = ymax - ymin

                if face_w > 20 and face_h > 20:
                    face_roi = frame[ymin:ymax, xmin:xmax]
                    gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                    resized_roi = cv2.resize(gray_roi, (48, 48), interpolation=cv2.INTER_AREA)
                    norm_roi = resized_roi.astype('float32') / 255.0
                    tensor_inp = tf.convert_to_tensor(np.expand_dims(np.expand_dims(norm_roi, axis=0), axis=-1))

                    raw_preds = infer(tensor_inp).numpy()[0]
                    if smoothed_preds is None:
                        smoothed_preds = raw_preds
                    else:
                        smoothed_preds = alpha * raw_preds + (1.0 - alpha) * smoothed_preds

                    max_idx = int(np.argmax(smoothed_preds))
                    top_emotion = emotion_labels[max_idx]
                    top_conf = smoothed_preds[max_idx] * 100

                    # -------------------------------------------------
                    # C. Visualization & HUD
                    # -------------------------------------------------
                    box_color = (0, 0, 255) if is_drowsy else (0, 220, 100)
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), box_color, 2)

                    tag = f"{top_emotion} ({top_conf:.0f}%)"
                    cv2.putText(frame, tag, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2, cv2.LINE_AA)

                    bar_x = xmax + 10
                    if bar_x + 160 < w:
                        cv2.rectangle(frame, (bar_x - 5, ymin - 10), (bar_x + 165, ymin + 150), (15, 15, 15), -1)
                        for i, prob in enumerate(smoothed_preds):
                            y_pos = ymin + (i * 20) + 10
                            cv2.putText(frame, f"{emotion_labels[i][:4]}:", (bar_x, y_pos),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 230, 230), 1)
                            b_width = int(prob * 80)
                            b_col = (0, 220, 100) if i == max_idx else (180, 180, 180)
                            cv2.rectangle(frame, (bar_x + 65, y_pos - 7), (bar_x + 65 + b_width, y_pos + 1), b_col, -1)

                # Drowsiness Alert Overlay Banner
                if is_drowsy:
                    cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 200), -1)
                    cv2.putText(frame, "CRITICAL ALERT: DROWSINESS DETECTED!", (int(w * 0.1), 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3, cv2.LINE_AA)
        else:
            # Keep the smoother through brief detection dropouts; a single missed frame followed
            # by a raw, unsmoothed prediction is exactly the spike EMA exists to suppress.
            missed_frames += 1
            if missed_frames >= EMA_RESET_AFTER_MISSES:
                smoothed_preds = None

        # Top HUD: EAR & FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        if avg_ear is not None:
            cv2.putText(frame, f"Eye Aspect Ratio (EAR): {avg_ear:.2f}", (w - 280, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

        cv2.imshow("EdgeVision: Real-Time Emotion & Drowsiness Guard", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
