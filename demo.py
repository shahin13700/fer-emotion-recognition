import cv2
import numpy as np
import json
import os
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.keras.models import load_model

def main():
    print("Initializing Live Emotion Demo...")

    # -------------------------------------------------------------
    # 1. Load Dynamic Class Labels
    # -------------------------------------------------------------
    indices_path = 'outputs/class_indices.json'
    if not os.path.exists(indices_path):
        print(f"Error: {indices_path} not found. Please train the model first.")
        return

    with open(indices_path, 'r') as f:
        class_indices = json.load(f)

    # Invert {'angry': 0, 'disgust': 1} -> {0: 'Angry', 1: 'Disgust'}
    idx_to_class = {v: k for k, v in class_indices.items()}
    emotion_labels = [idx_to_class[i].capitalize() for i in range(len(idx_to_class))]
    print(f"Loaded emotion labels: {emotion_labels}")

    # -------------------------------------------------------------
    # 2. Load Model & Warm Up
    # -------------------------------------------------------------
    model_path = 'model/emotion_model.keras'
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}.")
        return

    print(f"Loading model from {model_path}... (Optimizing for speed)")
    model = load_model(model_path)

    # Trace the forward pass once into a graph. An eager model(...) call costs ~25 ms per
    # face on a desktop CPU; the compiled function ~2 ms, which is what makes 30+ FPS possible.
    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 48, 48, 1), dtype=tf.float32)])
    def infer(batch):
        return model(batch, training=False)

    # Warmup / trace so the very first frame doesn't freeze the webcam
    _ = infer(tf.zeros((1, 48, 48, 1)))

    # -------------------------------------------------------------
    # 3. Setup OpenCV
    # -------------------------------------------------------------
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print(f"Error: Failed to load cascade from {cascade_path}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("\nWebcam is active!")
    print("Press 'Q' to quit anytime.")

    # Temporal smoothing buffer to eliminate real-time prediction flicker
    smoothed_preds = None
    missed_frames = 0
    ema_reset_after_misses = 30  # ~1 s without a face before the smoother is reset
    alpha = 0.65  # Weight for current frame (0.65 balances responsiveness and stability)

    # -------------------------------------------------------------
    # 4. Live Video Loop
    # -------------------------------------------------------------
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1) # Mirror image for natural user feel
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(48, 48) # Minimum size matching our model input
        )

        if len(faces) == 0:
            # Keep the smoother through brief detection dropouts; only reset once the face
            # has really left, otherwise the first frame back is a raw, unsmoothed spike.
            missed_frames += 1
            if missed_frames >= ema_reset_after_misses:
                smoothed_preds = None
        else:
            missed_frames = 0

        # Sort detected faces by area descending so primary face is index 0
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)

        # Process every detected face
        for idx, (x, y, w, h) in enumerate(faces):
            # Draw primary face rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 191, 0), 2)  # Deep blue box

            # Crop exactly as training images
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            # Normalize pixel limits [0, 1] exactly matching train.py rescale=1./255
            roi = roi_gray.astype('float32') / 255.0

            # Reshape for tf input (Batch, Height, Width, Channels) => (1, 48, 48, 1)
            roi = np.expand_dims(roi, axis=0)
            roi = np.expand_dims(roi, axis=-1)
            tensor_input = tf.convert_to_tensor(roi)

            raw_preds = infer(tensor_input).numpy()[0]

            # Apply Exponential Moving Average (EMA) temporal smoothing ONLY to primary face
            if idx == 0:
                if smoothed_preds is None:
                    smoothed_preds = raw_preds
                else:
                    smoothed_preds = alpha * raw_preds + (1.0 - alpha) * smoothed_preds
                preds_display = smoothed_preds
            else:
                preds_display = raw_preds

            max_idx = int(np.argmax(preds_display))
            max_conf = preds_display[max_idx]
            top_emotion = emotion_labels[max_idx]

            # Draw main label above the bounding box
            label_text = f"{top_emotion} ({max_conf*100:.1f}%)"
            cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            # Draw probability bar chart anchored to the right of the face
            bar_start_x = x + w + 10
            bar_start_y = y

            # Only draw chart if it fits on screen
            if bar_start_x + 160 < frame.shape[1]:
                cv2.rectangle(frame, (bar_start_x - 5, bar_start_y - 15),
                              (bar_start_x + 160, bar_start_y + (len(emotion_labels)*20) + 5),
                              (0, 0, 0), -1) # Dark background for visibility

                for i, prob in enumerate(preds_display):
                    y_pos = bar_start_y + (i * 20)
                    cv2.putText(frame, f"{emotion_labels[i]}:", (bar_start_x, y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

                    bar_width = int(prob * 80)
                    color = (0, 255, 0) if i == max_idx else (200, 200, 200)
                    cv2.rectangle(frame, (bar_start_x + 70, y_pos - 8),
                                  (bar_start_x + 70 + bar_width, y_pos + 2),
                                  color, -1)

        cv2.imshow("FER2013 Live Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
