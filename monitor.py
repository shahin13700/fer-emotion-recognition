"""
Edge-Vision: Real-Time Facial Emotion & Driver Drowsiness Guard
Combines Google MediaPipe Face Mesh (468 landmarks) with MiniXception CNN.
Tracks 7 emotion categories and computes Eye Aspect Ratio (EAR) to detect driver fatigue.
"""

import os
import json
import time
import warnings
warnings.filterwarnings('ignore')
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# MediaPipe
import mediapipe as mp

# -------------------------------------------------------------
# 1. MediaPipe Landmark Constants for Eye Aspect Ratio (EAR)
# -------------------------------------------------------------
# Left eye landmark indices (MediaPipe 468 mesh)
LEFT_EYE_POINTS = [33, 160, 158, 133, 153, 144]
# Right eye landmark indices
RIGHT_EYE_POINTS = [362, 385, 387, 263, 373, 380]

# EAR Thresholds
EAR_THRESHOLD = 0.21         # Eye is considered closed below this ratio
DROWSY_CONSEC_FRAMES = 15    # Number of consecutive frames eyes must be closed to sound alert (~0.5s)


def calculate_ear(landmarks, eye_indices, img_w, img_h):
    """
    Computes Eye Aspect Ratio (EAR) for a given eye using 6 facial landmarks.
    EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)
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


def main():
    print("===============================================================")
    print("  EdgeVision: Real-Time Emotion & Drowsiness Guard")
    print("  Powered by MediaPipe Face Mesh + MiniXception")
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
    _ = model(tf.zeros((1, 48, 48, 1)), training=False) # Warmup

    # 2. Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 128))

    # 3. Setup Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access webcam.")
        return

    print("\n[ACTIVE] Webcam initialized. Press 'Q' to quit anytime.")

    # State variables
    drowsy_counter = 0
    smoothed_preds = None
    alpha = 0.65  # EMA smoothing factor
    prev_time = time.time()

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

        # MediaPipe Mesh Inference
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark

                # -----------------------------------------------------
                # A. Eye Aspect Ratio & Drowsiness Tracking
                # -----------------------------------------------------
                left_ear = calculate_ear(landmarks, LEFT_EYE_POINTS, w, h)
                right_ear = calculate_ear(landmarks, RIGHT_EYE_POINTS, w, h)
                avg_ear = (left_ear + right_ear) / 2.0

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
                
                # Bounding box with padding
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

                    raw_preds = model(tensor_inp, training=False).numpy()[0]
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

                    # Draw Top Emotion Tag
                    tag = f"{top_emotion} ({top_conf:.0f}%)"
                    cv2.putText(frame, tag, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2, cv2.LINE_AA)

                    # Draw Emotion Probability Bars
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
            smoothed_preds = None

        # Top HUD: EAR & FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        if results.multi_face_landmarks:
            cv2.putText(frame, f"Eye Aspect Ratio (EAR): {avg_ear:.2f}", (w - 280, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

        cv2.imshow("EdgeVision: Real-Time Emotion & Drowsiness Guard", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
