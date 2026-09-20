"""
EdgeVision Active Learning: Fine-Tuning Engine with Guardrails
Enforces 4 guardrails:
1. Minimum Volume Gate (>= 10 samples per emotion class before training)
2. Blended Data Generator (mixes a fixed share of user data into every batch)
3. Augmentation Pipeline (rotation, flip, zoom, brightness) applied to raw 0-255 pixels
4. Before/After evaluation on BOTH a held-out slice of the user's own faces and the
   FER2013 test set; promotion requires no user-holdout regression, a bounded overall
   and macro-F1 regression on FER2013, and no single class losing more than
   --max_class_drop recall (guards against the majority-class drift that an
   unweighted fine-tune produces). The previous production weights are always backed up.

Honesty note: using the FER2013 test set as a promotion guard is a mild form of model
selection on the test set. For a personal model that is acceptable; do not quote the
post-fine-tune test accuracy as a benchmark result.

Requires the FER2013 dataset unpacked at dataset/train and dataset/test
(see README "Retraining Base Model from Scratch" for the download command).
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
import hashlib
import shutil
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.utils.class_weight import compute_class_weight

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
USER_HOLDOUT_FRACTION = 0.2    # Share of user faces held out to measure personal adaptation
MAX_TEST_DROP_DEFAULT = 0.01   # Allowed FER2013 accuracy / macro-F1 regression (1.0 percentage point)
MAX_CLASS_DROP_DEFAULT = 0.03  # Allowed per-class recall regression on FER2013 (3 percentage points)
IMAGE_EXTS = ('.png', '.jpg', '.jpeg')


def list_unique_user_images(emo_folder):
    """
    Image paths in a class folder with byte-identical duplicates removed. Saving the same
    booth capture twice (or copying files) must not count twice toward the volume gate,
    and must never place a copy of a training image into the held-out split.
    """
    seen, unique = set(), []
    if not os.path.isdir(emo_folder):
        return unique
    for fname in sorted(os.listdir(emo_folder)):
        if not fname.lower().endswith(IMAGE_EXTS):
            continue
        fpath = os.path.join(emo_folder, fname)
        with open(fpath, 'rb') as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        unique.append(fpath)
    return unique


def check_dataset_dirs():
    """Fails fast with a useful message if the FER2013 base dataset is missing."""
    missing = [d for d in (BASE_TRAIN_DIR, TEST_DIR) if not os.path.isdir(d)]
    if missing:
        print("[ERROR] Fine-tuning blends your faces with the FER2013 base data, which is not in the repo.")
        print(f"        Missing: {', '.join(missing)}")
        print("        Download and unpack it first:")
        print("          kaggle datasets download -d msambare/fer2013")
        print("          unzip -q fer2013.zip -d dataset/")
        sys.exit(1)


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

    duplicates = 0
    if os.path.exists(USER_CONTRIB_DIR):
        for emo in emotions:
            emo_folder = os.path.join(USER_CONTRIB_DIR, emo)
            if os.path.isdir(emo_folder):
                all_files = [f for f in os.listdir(emo_folder) if f.lower().endswith(IMAGE_EXTS)]
                unique_files = list_unique_user_images(emo_folder)
                counts[emo] = len(unique_files)
                duplicates += len(all_files) - len(unique_files)

    print("\n" + "=" * 60)
    print("[GUARDRAIL 1] LOCAL USER DATA VOLUME AUDIT (unique images)")
    print("=" * 60)
    if duplicates:
        print(f"Ignoring {duplicates} byte-identical duplicate file(s).")
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


def test_metrics(y_true, y_pred, num_classes):
    """Accuracy, macro-F1 and per-class recall: the promotion gate looks at all three."""
    labels = list(range(num_classes))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)),
        "recall": [float(r) for r in recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)],
    }


def promotion_decision(base, new, base_user_acc, new_user_acc, class_labels,
                       max_test_drop=MAX_TEST_DROP_DEFAULT, max_class_drop=MAX_CLASS_DROP_DEFAULT):
    """
    Returns (promote, reasons). The candidate must not get worse on the user's held-out
    faces, may lose at most max_test_drop in FER2013 accuracy and macro-F1, and no class
    may lose more than max_class_drop recall.
    """
    reasons = []
    acc_delta = new["accuracy"] - base["accuracy"]
    f1_delta = new["macro_f1"] - base["macro_f1"]
    if acc_delta < -max_test_drop:
        reasons.append(f"FER2013 accuracy dropped {abs(acc_delta)*100:.2f} pp (limit {max_test_drop*100:.2f} pp)")
    if f1_delta < -max_test_drop:
        reasons.append(f"FER2013 macro-F1 dropped {abs(f1_delta)*100:.2f} pp (limit {max_test_drop*100:.2f} pp)")
    for label, r_base, r_new in zip(class_labels, base["recall"], new["recall"]):
        if r_new - r_base < -max_class_drop:
            reasons.append(f"{label} recall dropped {(r_base - r_new)*100:.1f} pp (limit {max_class_drop*100:.1f} pp)")
    if base_user_acc is not None and new_user_acc is not None and new_user_acc < base_user_acc:
        reasons.append(f"accuracy on your held-out faces dropped {(base_user_acc - new_user_acc)*100:.2f} pp")
    return len(reasons) == 0, reasons


def load_and_preprocess_user_data(class_indices):
    """
    Loads user-contributed images as (48, 48, 1) float32 tensors in the RAW 0-255
    range (NOT normalized) along with one-hot labels.

    Keeping raw pixel values matters: Keras' legacy ImageDataGenerator applies
    brightness_range via PIL and truncates any array whose max is <= 1.0 to
    0/1 integers, turning pre-normalized images into black squares. Rescaling
    is therefore applied by the generator (rescale=1./255), exactly like the
    base FER2013 pipeline.
    """
    x_user = []
    y_user = []
    num_classes = len(class_indices)

    if not os.path.exists(USER_CONTRIB_DIR):
        return np.empty((0, 48, 48, 1)), np.empty((0, num_classes))

    for emo, class_idx in class_indices.items():
        for fpath in list_unique_user_images(os.path.join(USER_CONTRIB_DIR, emo)):
            # Same preprocessing as inference (app.py annotate_frame): grayscale + INTER_AREA.
            # PIL's default nearest-neighbour resize would introduce train/serve skew.
            gray = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            arr = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA).astype(np.float32)[..., np.newaxis]
            one_hot = np.zeros(num_classes, dtype=np.float32)
            one_hot[class_idx] = 1.0

            x_user.append(arr)
            y_user.append(one_hot)

    if len(x_user) == 0:
        return np.empty((0, 48, 48, 1)), np.empty((0, num_classes))

    return np.array(x_user, dtype=np.float32), np.array(y_user, dtype=np.float32)


def build_user_datagen():
    """Augmentation for raw 0-255 user faces. rescale runs AFTER the PIL-based brightness shift."""
    return ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        zoom_range=0.15,
        width_shift_range=0.10,
        height_shift_range=0.10,
        brightness_range=[0.85, 1.15],
        horizontal_flip=True
    )


def split_user_holdout(x_user, y_user, fraction=USER_HOLDOUT_FRACTION, seed=42):
    """
    Splits user data into train/holdout per class so the holdout measures whether
    fine-tuning actually improved recognition of THIS user's faces.
    Classes with fewer than 5 samples contribute nothing to the holdout.
    """
    rng = np.random.default_rng(seed)
    train_idx, hold_idx = [], []
    labels = np.argmax(y_user, axis=1) if len(y_user) else np.array([], dtype=int)
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        n_hold = int(len(idx) * fraction) if len(idx) >= 5 else 0
        hold_idx.extend(idx[:n_hold])
        train_idx.extend(idx[n_hold:])
    train_idx, hold_idx = np.array(train_idx, dtype=int), np.array(hold_idx, dtype=int)
    return x_user[train_idx], y_user[train_idx], x_user[hold_idx], y_user[hold_idx]


def evaluate_on_user_holdout(model_instance, x_hold, y_hold):
    """Accuracy on the raw 0-255 user holdout (normalized here, no augmentation)."""
    if len(x_hold) == 0:
        return None
    preds = model_instance.predict(x_hold / 255.0, verbose=0)
    return float(accuracy_score(np.argmax(y_hold, axis=1), np.argmax(preds, axis=1)))


TRAIN_SCOPES = ("head", "last_block", "all")


def set_train_scope(model_instance, scope="head"):
    """
    Limits which layers a fine-tune may change. The base model is converged, and unfreezing
    everything lets a few hundred steps of personal data drift the minority classes (Sad
    recall fell ~10 pp in every full-network trial). "head" trains only the classifier;
    "last_block" also unfreezes the final residual block; "all" is the previous behaviour.
    BatchNormalization layers stay frozen in every scope so running statistics do not move.
    Returns the number of trainable layers.
    """
    if scope not in TRAIN_SCOPES:
        raise ValueError(f"scope must be one of {TRAIN_SCOPES}")
    layers = model_instance.layers
    if scope == "all":
        cutoff = 0
    elif scope == "head":
        cutoff = len(layers) - 1                      # Dense only
    else:
        # Last residual block starts after the previous Add layer
        add_indices = [i for i, l in enumerate(layers) if l.__class__.__name__ == "Add"]
        cutoff = (add_indices[-2] + 1) if len(add_indices) >= 2 else 0
    n_trainable = 0
    for i, layer in enumerate(layers):
        is_bn = layer.__class__.__name__ == "BatchNormalization"
        layer.trainable = (i >= cutoff) and not is_bn
        n_trainable += int(layer.trainable and bool(layer.weights))
    return n_trainable


def sample_weights_for(y_batch, class_weight):
    """Per-sample weights from a {class_index: weight} dict (Keras 3 rejects class_weight for generators)."""
    idx = np.argmax(y_batch, axis=1)
    return np.array([class_weight.get(int(i), 1.0) for i in idx], dtype=np.float32)


def create_blended_generator(base_gen, x_user, y_user, user_datagen=None, batch_size=32, user_ratio=0.25,
                             class_weight=None):
    """
    Guardrail 2: Blended Generator that injects augmented user data into every
    training batch alongside base FER2013 samples.

    x_user must be RAW 0-255 pixels (see load_and_preprocess_user_data); the user
    datagen rescales to [0, 1] after augmentation, matching the base pipeline.
    With class_weight, batches are yielded as (x, y, sample_weight) so the FER2013 share
    keeps the class balancing the base model was trained with. User samples always get
    weight 1.0: the user set is balanced by construction (>= 10 per class), and applying
    the ~9x Disgust weight to it yanks the minority classes hard on a converged model.
    """
    if len(x_user) == 0:
        if class_weight is None:
            return base_gen

        def _weighted_base():
            while True:
                bx, by = next(base_gen)
                yield bx, by, sample_weights_for(by, class_weight)
        return _weighted_base()

    if user_datagen is None:
        user_datagen = build_user_datagen()

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

            if class_weight is None:
                yield merged_x[indices], merged_y[indices]
            else:
                merged_w = np.concatenate([sample_weights_for(by, class_weight),
                                           np.ones(len(user_x), dtype=np.float32)], axis=0)
                yield merged_x[indices], merged_y[indices], merged_w[indices]

    return _generator()


def main():
    parser = argparse.ArgumentParser(description="EdgeVision Fine-Tuning Engine with Guardrails")
    parser.add_argument("--force", action="store_true", help="Bypass the minimum volume gate for testing")
    parser.add_argument("--reset", action="store_true", help="Restore production model from pristine original baseline")
    parser.add_argument("--epochs", type=int, default=15, help="Number of fine-tuning epochs (default: 15)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for fine-tuning (default: 1e-5; the base model is already converged)")
    parser.add_argument("--max_test_drop", type=float, default=MAX_TEST_DROP_DEFAULT,
                        help="Max allowed FER2013 accuracy / macro-F1 drop before rollback, as a fraction (default: 0.01)")
    parser.add_argument("--max_class_drop", type=float, default=MAX_CLASS_DROP_DEFAULT,
                        help="Max allowed per-class recall drop on FER2013, as a fraction (default: 0.03)")
    parser.add_argument("--train_scope", choices=TRAIN_SCOPES, default="head",
                        help="Layers to fine-tune: head (classifier only, default), last_block, or all")
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
    check_dataset_dirs()

    with open(INDICES_PATH, 'r') as f:
        class_indices = json.load(f)
    class_labels = list(class_indices.keys())

    print(f"\nLoading baseline model from {MODEL_PATH}...")
    baseline_model = load_model(MODEL_PATH)

    # -------------------------------------------------------------
    # Guardrail 2: Load user data and hold out a slice for evaluation
    # -------------------------------------------------------------
    x_user, y_user = load_and_preprocess_user_data(class_indices)
    print(f"\nUser Data Loaded: {len(x_user)} verified portraits.")
    x_user_train, y_user_train, x_user_hold, y_user_hold = split_user_holdout(x_user, y_user)
    print(f"User split: {len(x_user_train)} for training, {len(x_user_hold)} held out for evaluation.")

    # -------------------------------------------------------------
    # Guardrail 4a: Evaluate Baseline Before Training
    # -------------------------------------------------------------
    print("\nEvaluating Baseline Model on Test Set (`dataset/test`)...")
    base_acc, y_true, y_pred = evaluate_model_on_test(baseline_model, class_labels)
    base_metrics = test_metrics(y_true, y_pred, len(class_labels))
    print(f"[METRIC] Baseline FER2013 Test Accuracy: {base_acc*100:.2f}%  (macro-F1 {base_metrics['macro_f1']*100:.2f}%)")
    base_user_acc = evaluate_on_user_holdout(baseline_model, x_user_hold, y_user_hold)
    if base_user_acc is not None:
        print(f"[METRIC] Baseline Accuracy on YOUR held-out faces: {base_user_acc*100:.2f}%")

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

    # When blending, the base generator yields exactly the base share of each batch so that
    # no FER2013 samples are sliced off and silently discarded.
    user_ratio = 0.25
    blending = len(x_user_train) > 0
    base_batch_size = max(1, args.batch_size - int(args.batch_size * user_ratio)) if blending else args.batch_size

    base_train_gen = train_datagen.flow_from_directory(
        BASE_TRAIN_DIR,
        target_size=(48, 48),
        color_mode='grayscale',
        batch_size=base_batch_size,
        class_mode='categorical',
        subset='training',
        seed=42,
        shuffle=True
    )

    # Same class-balanced weights the base model was trained with; without them a fine-tune
    # drifts toward Happy/Neutral and quietly loses Sad/Fear/Disgust recall.
    base_classes = base_train_gen.classes
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(base_classes), y=base_classes)
    class_weight_dict = {int(c): float(w) for c, w in zip(np.unique(base_classes), class_weights)}

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
    if blending:
        print("Blending user data into active training pipeline (25% user / 75% base per batch)...")
    train_pipeline = create_blended_generator(
        base_train_gen,
        x_user_train,
        y_user_train,
        user_datagen=None,
        batch_size=args.batch_size,
        user_ratio=user_ratio,
        class_weight=class_weight_dict
    )

    n_trainable = set_train_scope(baseline_model, args.train_scope)
    print(f"Train scope: {args.train_scope} ({n_trainable} trainable layer(s); BatchNorm frozen)")

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
    print(f"Epochs: {args.epochs} | LR: {args.lr} | Batch Size: {args.batch_size} | Scope: {args.train_scope}")
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
    # Guardrail 4b: Evaluate Fine-Tuned Model
    # -------------------------------------------------------------
    print("\nEvaluating Fine-Tuned Model on Test Set (`dataset/test`)...")
    new_acc, y_true, y_pred = evaluate_model_on_test(baseline_model, class_labels)
    new_metrics = test_metrics(y_true, y_pred, len(class_labels))
    delta = new_acc - base_acc
    new_user_acc = evaluate_on_user_holdout(baseline_model, x_user_hold, y_user_hold)

    print("\n" + "=" * 60)
    print("FINE-TUNING EVALUATION SUMMARY & SAFETY VERIFICATION")
    print("=" * 60)
    print(f"FER2013 Test Accuracy:   {base_acc*100:.2f}% -> {new_acc*100:.2f}%  ({'+' if delta >= 0 else ''}{delta*100:.2f} pp)")
    f1_delta = new_metrics['macro_f1'] - base_metrics['macro_f1']
    print(f"FER2013 Macro-F1:        {base_metrics['macro_f1']*100:.2f}% -> {new_metrics['macro_f1']*100:.2f}%  ({'+' if f1_delta >= 0 else ''}{f1_delta*100:.2f} pp)")
    print("Per-class recall (FER2013):")
    for label, r0, r1 in zip(class_labels, base_metrics['recall'], new_metrics['recall']):
        print(f"  {label:<10} {r0*100:6.1f}% -> {r1*100:6.1f}%  ({'+' if r1 >= r0 else ''}{(r1 - r0)*100:.1f} pp)")
    if base_user_acc is not None:
        user_delta = new_user_acc - base_user_acc
        print(f"Your Held-Out Faces:     {base_user_acc*100:.2f}% -> {new_user_acc*100:.2f}%  ({'+' if user_delta >= 0 else ''}{user_delta*100:.2f} pp)")
    else:
        user_delta = None
        print("Your Held-Out Faces:     (not enough samples per class for a holdout; skipped)")

    promote, reasons = promotion_decision(base_metrics, new_metrics, base_user_acc, new_user_acc, class_labels,
                                          max_test_drop=args.max_test_drop, max_class_drop=args.max_class_drop)

    report_data = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "baseline_accuracy": base_acc,
        "finetuned_accuracy": new_acc,
        "accuracy_delta": delta,
        "baseline_macro_f1": base_metrics["macro_f1"],
        "finetuned_macro_f1": new_metrics["macro_f1"],
        "baseline_recall": dict(zip(class_labels, base_metrics["recall"])),
        "finetuned_recall": dict(zip(class_labels, new_metrics["recall"])),
        "rejection_reasons": reasons,
        "baseline_user_holdout_accuracy": base_user_acc,
        "finetuned_user_holdout_accuracy": new_user_acc,
        "user_holdout_size": int(len(x_user_hold)),
        "max_test_drop": args.max_test_drop,
        "max_class_drop": args.max_class_drop,
        "promoted": promote,
        "user_samples_count": counts,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "train_scope": args.train_scope
    }

    with open(REPORT_PATH, 'w') as f:
        json.dump(report_data, f, indent=2)

    # Safety Promotion & Rollback Guard
    baseline_model.save(FINETUNED_MODEL_PATH)
    print(f"Candidate model weights saved to: {FINETUNED_MODEL_PATH}")

    if promote:
        print("\n[SUCCESS] Candidate passed both guards. Promoting weights...")
        shutil.copyfile(MODEL_PATH, BACKUP_MODEL_PATH)
        baseline_model.save(MODEL_PATH)
        print(f"Previous production weights backed up to: {BACKUP_MODEL_PATH}")
        print(f"New production weights promoted to: {MODEL_PATH}")
    else:
        print(f"\n[ROLLBACK SAFETY] Not promoted: {'; '.join(reasons)}.")
        print(f"Production weights in {MODEL_PATH} remain UNTOUCHED.")
        print(f"Fine-tuned candidate is preserved in {FINETUNED_MODEL_PATH} for inspection.")
        print(f"Note: You can restore pristine baseline anytime via: python fine_tune.py --reset")

    print(f"Audit report written to {REPORT_PATH}.\n")


if __name__ == '__main__':
    main()
