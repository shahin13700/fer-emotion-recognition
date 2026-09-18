"""
EdgeVision Active Learning: Fine-Tuning Engine with Production Guardrails
Enforces 4 strict guardrails:
1. Minimum Volume Gate (>= 10 samples per emotion class before training)
2. Oversampled Blended Data Generator (actually mixes user data into every gradient step)
3. Heavy Augmentation Pipeline (rotation, flip, zoom, brightness)
4. Empirical Before/After Test Evaluation with Immutable Baseline Rollback
"""

import os
import sys

# Ensure UTF-8 output handling on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import json
import argparse
import datetime
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, classification_report

USER_CONTRIB_DIR = 'dataset/user_contributed'
METADATA_JSONL = os.path.join(USER_CONTRIB_DIR, 'metadata.jsonl')
BASE_TRAIN_DIR = 'dataset/train'
TEST_DIR = 'dataset/test'
MODEL_PATH = 'model/emotion_model.keras'
ORIGINAL_MODEL_PATH = 'model/emotion_model_original.keras'
BACKUP_MODEL_PATH = 'model/emotion_model_backup.keras'
FINETUNED_MODEL_PATH = 'model/emotion_model_finetuned.keras'
REPORT_PATH = 'outputs/fine_tune_report.json'
INDICES_PATH = 'outputs/class_indices.json'

MIN_SAMPLES_PER_CLASS = 10
USER_OVERSAMPLE_FACTOR = 15


def ensure_original_baseline():
    """Ensures an immutable original baseline model checkpoint exists on disk."""
    if not os.path.exists(ORIGINAL_MODEL_PATH) and os.path.exists(MODEL_PATH):
        shutil.copyfile(MODEL_PATH, ORIGINAL_MODEL_PATH)
        print(f"[INIT] Preserved immutable ground truth model at: {ORIGINAL_MODEL_PATH}")


def check_guardrails(force=False):
    """
    Guardrail 1: Verifies dataset/user_contributed has at least
    MIN_SAMPLES_PER_CLASS images per emotion.
    """
    if not os.path.exists(INDICES_PATH):
        print(f"Error: Missing {INDICES_PATH}.")
        sys.exit(1)

    with open(INDICES_PATH, 'r') as f:
        class_indices = json.load(f)

    emotions = list(class_indices.keys())
    counts = {emo: 0 for emo in emotions}

    if os.path.exists(USER_CONTRIB_DIR):
        for emo in emotions:
            emo_folder = os.path.join(USER_CONTRIB_DIR, emo)
            if os.path.isdir(emo_folder):
                valid_files = [f for f in os.listdir(emo_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                counts[emo] = len(valid_files)

    print("\n" + "=" * 60)
    print("[GUARDRAIL 1] LOCAL USER DATA VOLUME AUDIT")
    print("=" * 60)
    print(f"{'Emotion':<15} | {'Collected':<12} | {'Requirement':<12} | {'Status'}")
    print("-" * 60)

    failing_classes = []
    for emo in emotions:
        cnt = counts[emo]
        status = "[PASS]" if cnt >= MIN_SAMPLES_PER_CLASS else "[DEFICIT]"
        if cnt < MIN_SAMPLES_PER_CLASS:
            failing_classes.append((emo, cnt))
        print(f"{emo:<15} | {cnt:<12} | {MIN_SAMPLES_PER_CLASS:<12} | {status}")
    print("-" * 60)

    total_contributed = sum(counts.values())
    print(f"Total Collected Faces: {total_contributed}")

    if failing_classes and not force:
        print("\n[BLOCKED] Fine-Tuning Blocked by Active Learning Guardrails!")
        print("To protect model generalization and prevent overfitting on small samples,")
        print(f"each class requires at least {MIN_SAMPLES_PER_CLASS} verified local samples.")
        print(f"Missing quota for: {', '.join([f'{e} ({c}/{MIN_SAMPLES_PER_CLASS})' for e, c in failing_classes])}")
        print("\n>> Action: Open the Photo Booth (`python app.py`) to capture additional expressions,")
        print("or pass `--force` if executing a smoke test.")
        return False, counts

    if failing_classes and force:
        print("\n[OVERRIDE] Proceeding despite volume deficit because `--force` flag was provided.")
        return True, counts

    print("\n[PASSED] Volume audit passed! All classes satisfy guardrail requirements.")
    return True, counts


def evaluate_model_on_test(model_instance, class_labels):
    """Evaluates a model instance on the unaugmented FER2013 test set."""
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_gen = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(48, 48),
        color_mode='grayscale',
        batch_size=64,
        class_mode='categorical',
        shuffle=False
    )
    preds = model_instance.predict(test_gen, verbose=0)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_gen.classes
    acc = accuracy_score(y_true, y_pred)
    return float(acc), y_true, y_pred


def load_and_preprocess_user_data(class_indices):
    """
    Loads user-contributed images as (48, 48, 1) normalized tensors
    along with one-hot labels.
    """
    x_user = []
    y_user = []
    num_classes = len(class_indices)

    if not os.path.exists(USER_CONTRIB_DIR):
        return np.empty((0, 48, 48, 1)), np.empty((0, num_classes))

    for emo, class_idx in class_indices.items():
        emo_folder = os.path.join(USER_CONTRIB_DIR, emo)
        if os.path.isdir(emo_folder):
            for fname in os.listdir(emo_folder):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    fpath = os.path.join(emo_folder, fname)
                    img = tf.keras.preprocessing.image.load_img(
                        fpath, color_mode='grayscale', target_size=(48, 48)
                    )
                    arr = tf.keras.preprocessing.image.img_to_array(img) / 255.0
                    one_hot = np.zeros(num_classes, dtype=np.float32)
                    one_hot[class_idx] = 1.0

                    x_user.append(arr)
                    y_user.append(one_hot)

    if len(x_user) == 0:
        return np.empty((0, 48, 48, 1)), np.empty((0, num_classes))

    return np.array(x_user, dtype=np.float32), np.array(y_user, dtype=np.float32)


def create_blended_generator(base_gen, x_user, y_user, user_datagen=None, batch_size=32, user_ratio=0.25):
    """
    Guardrail 2: Blended Generator that actively injects augmented user data
    into every training batch alongside base FER2013 samples.
    
    NOTE: x_user is already normalized to [0, 1] by load_and_preprocess_user_data.
    We use user_datagen WITHOUT rescale so user images are NOT divided by 255 twice.
    """
    if len(x_user) == 0:
        return base_gen

    if user_datagen is None:
        user_datagen = ImageDataGenerator(
            rotation_range=15,
            zoom_range=0.15,
            width_shift_range=0.10,
            height_shift_range=0.10,
            brightness_range=[0.85, 1.15],
            horizontal_flip=True
            # No rescale! x_user is already in [0, 1]
        )

    user_batch_size = max(1, int(batch_size * user_ratio))
    base_batch_size = max(1, batch_size - user_batch_size)

    user_flow = user_datagen.flow(
        x_user, y_user,
        batch_size=user_batch_size,
        shuffle=True
    )


    def _generator():
        while True:
            base_x, base_y = next(base_gen)
            user_x, user_y = next(user_flow)

            bx = base_x[:base_batch_size]
            by = base_y[:base_batch_size]

            merged_x = np.concatenate([bx, user_x], axis=0)
            merged_y = np.concatenate([by, user_y], axis=0)

            # Shuffle combined batch
            indices = np.arange(len(merged_x))
            np.random.shuffle(indices)

            yield merged_x[indices], merged_y[indices]

    return _generator()


def main():
    parser = argparse.ArgumentParser(description="EdgeVision Fine-Tuning Engine with Guardrails")
    parser.add_argument("--force", action="store_true", help="Bypass the minimum volume gate for testing")
    parser.add_argument("--reset", action="store_true", help="Restore production model from pristine original baseline")
    parser.add_argument("--epochs", type=int, default=15, help="Number of fine-tuning epochs (default: 15)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for fine-tuning (default: 5e-5)")
    args = parser.parse_args()

    ensure_original_baseline()

    if args.reset:
        if os.path.exists(ORIGINAL_MODEL_PATH):
            shutil.copyfile(ORIGINAL_MODEL_PATH, MODEL_PATH)
            print(f"[SUCCESS] Restored {MODEL_PATH} from immutable original baseline: {ORIGINAL_MODEL_PATH}")
            return
        else:
            print(f"[ERROR] Original baseline {ORIGINAL_MODEL_PATH} not found.")
            sys.exit(1)

    os.makedirs('outputs', exist_ok=True)
    os.makedirs('model', exist_ok=True)

    passed, counts = check_guardrails(force=args.force)
    if not passed:
        sys.exit(1)

    with open(INDICES_PATH, 'r') as f:
        class_indices = json.load(f)
    class_labels = list(class_indices.keys())

    print(f"\nLoading baseline model from {MODEL_PATH}...")
    baseline_model = load_model(MODEL_PATH)

    # -------------------------------------------------------------
    # Guardrail 4a: Evaluate Baseline Accuracy Before Training
    # -------------------------------------------------------------
    print("\nEvaluating Baseline Model on Test Set (`dataset/test`)...")
    base_acc, _, _ = evaluate_model_on_test(baseline_model, class_labels)
    print(f"[METRIC] Baseline Test Accuracy: {base_acc*100:.2f}%")

    # -------------------------------------------------------------
    # Guardrail 2: Oversample & Prepare User Data
    # -------------------------------------------------------------
    x_user, y_user = load_and_preprocess_user_data(class_indices)
    print(f"\nUser Data Loaded: {len(x_user)} verified portraits.")

    if len(x_user) > 0:
        print(f"Applying Guardrail 2: {USER_OVERSAMPLE_FACTOR}x Oversampling on user data...")
        x_user_oversampled = np.repeat(x_user, USER_OVERSAMPLE_FACTOR, axis=0)
        y_user_oversampled = np.repeat(y_user, USER_OVERSAMPLE_FACTOR, axis=0)
        print(f"Effective User Training Volume: {len(x_user_oversampled)} samples.")
    else:
        x_user_oversampled = np.empty((0, 48, 48, 1))
        y_user_oversampled = np.empty((0, len(class_indices)))

    # -------------------------------------------------------------
    # Guardrail 3: Heavy Data Augmentation Pipeline
    # -------------------------------------------------------------
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        zoom_range=0.15,
        width_shift_range=0.10,
        height_shift_range=0.10,
        brightness_range=[0.85, 1.15],
        horizontal_flip=True,
        validation_split=0.15
    )

    # Separate unaugmented datagen for clean validation metrics without noisy distortions
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.15
    )

    base_train_gen = train_datagen.flow_from_directory(
        BASE_TRAIN_DIR,
        target_size=(48, 48),
        color_mode='grayscale',
        batch_size=args.batch_size,
        class_mode='categorical',
        subset='training',
        seed=42,
        shuffle=True
    )

    val_gen = val_datagen.flow_from_directory(
        BASE_TRAIN_DIR,
        target_size=(48, 48),
        color_mode='grayscale',
        batch_size=args.batch_size,
        class_mode='categorical',
        subset='validation',
        seed=42,
        shuffle=False
    )

    # Actively blend user-contributed images with base images
    if len(x_user_oversampled) > 0:
        print("Blending user data into active training pipeline (25% user / 75% base per batch)...")
        train_pipeline = create_blended_generator(
            base_train_gen,
            x_user_oversampled,
            y_user_oversampled,
            user_datagen=None,
            batch_size=args.batch_size,
            user_ratio=0.25
        )
    else:
        train_pipeline = base_train_gen

    # Compile with low learning rate for gentle fine-tuning
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    baseline_model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)
    ]

    print("\n" + "=" * 60)
    print("STARTING GUARDRAILED FINE-TUNING")
    print(f"Epochs: {args.epochs} | LR: {args.lr} | Batch Size: {args.batch_size}")
    print("=" * 60)

    steps_per_epoch = len(base_train_gen)

    # Train for specified epochs using blended pipeline
    history = baseline_model.fit(
        train_pipeline,
        steps_per_epoch=steps_per_epoch,
        epochs=args.epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )

    # -------------------------------------------------------------
    # Guardrail 4b: Evaluate Fine-Tuned Accuracy on Test Set
    # -------------------------------------------------------------
    print("\nEvaluating Fine-Tuned Model on Test Set (`dataset/test`)...")
    new_acc, y_true, y_pred = evaluate_model_on_test(baseline_model, class_labels)
    delta = new_acc - base_acc

    print("\n" + "=" * 60)
    print("FINE-TUNING EVALUATION SUMMARY & SAFETY VERIFICATION")
    print("=" * 60)
    print(f"Baseline Test Accuracy:   {base_acc*100:.2f}%")
    print(f"Fine-Tuned Test Accuracy: {new_acc*100:.2f}%")
    print(f"Accuracy Delta:           {'+' if delta >= 0 else ''}{delta*100:.2f}%")

    report_data = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "baseline_accuracy": base_acc,
        "finetuned_accuracy": new_acc,
        "accuracy_delta": delta,
        "user_samples_count": counts,
        "oversample_factor": USER_OVERSAMPLE_FACTOR,
        "epochs": args.epochs,
        "learning_rate": args.lr
    }

    with open(REPORT_PATH, 'w') as f:
        json.dump(report_data, f, indent=2)

    # Safety Promotion & Rollback Guard
    baseline_model.save(FINETUNED_MODEL_PATH)
    print(f"Candidate model weights saved to: {FINETUNED_MODEL_PATH}")

    if delta >= 0:
        print("\n[SUCCESS] Model maintained/improved test accuracy! Promoting weights...")
        shutil.copyfile(MODEL_PATH, BACKUP_MODEL_PATH)
        baseline_model.save(MODEL_PATH)
        print(f"Previous production weights backed up to: {BACKUP_MODEL_PATH}")
        print(f"New production weights promoted to: {MODEL_PATH}")
    else:
        print(f"\n[ROLLBACK SAFETY] Accuracy dropped by {abs(delta)*100:.2f}%.")
        print(f"Production weights in {MODEL_PATH} remain UNTOUCHED.")
        print(f"Fine-tuned candidate is preserved in {FINETUNED_MODEL_PATH} for inspection.")
        print(f"Note: You can restore pristine baseline anytime via: python fine_tune.py --reset")

    print(f"Audit report written to {REPORT_PATH}.\n")


if __name__ == '__main__':
    main()
