"""
Automated Test Suite for EdgeVision: Photo Booth, Active Learning, Guardrails, Inference & Video
"""

import os
import json
import cv2
import numpy as np
import pytest
import tensorflow as tf
import app
import fine_tune
import explain
import monitor
from PIL import Image


# -------------------------------------------------------------
# Configuration & state
# -------------------------------------------------------------
def test_empirical_thresholds_match_artifact():
    """The typical-confidence table shown in the UI must come from outputs/empirical_thresholds.json."""
    expected_keys = {"Happy", "Surprise", "Neutral", "Angry", "Fear", "Sad", "Disgust"}
    assert set(app.EMPIRICAL_THRESHOLDS.keys()) == expected_keys
    with open(app.THRESHOLDS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    for emotion, value in app.EMPIRICAL_THRESHOLDS.items():
        assert 0.0 < value < 1.0
        assert value == round(float(data[emotion]["p25"]), 2)


def test_init_booth_state():
    """Verifies initial photo booth state has all 7 emotion slots empty."""
    state = app.init_booth_state()
    assert len(state) == 7
    for emo, data in state.items():
        assert data["score"] == 0.0
        assert data["crop"] is None
        assert data["train_crop"] is None
        assert data["is_sample"] is False
        assert data["saved"] is False


def test_use_sample_face_disgust_and_fear():
    """Verifies UX fail-safe loads sample faces without crashing and applies upscaling."""
    state = app.init_booth_state()

    updated_state, status, gallery = app.use_sample_face("Disgust", state)
    assert updated_state["Disgust"]["crop"] is not None
    assert updated_state["Disgust"]["is_sample"] is True
    assert updated_state["Disgust"]["score"] >= app.MIN_CAPTURE_FLOOR
    h, w, c = updated_state["Disgust"]["crop"].shape
    assert h >= 100 and w >= 100
    assert len(gallery) >= 1
    # Sample tiles carry no confidence number: the model does not predict that emotion for them
    assert gallery[0][1] == "Disgust (Sample)"

    updated_state, status, gallery = app.use_sample_face("Fear", updated_state)
    assert updated_state["Fear"]["crop"] is not None
    assert updated_state["Fear"]["is_sample"] is True
    assert len(gallery) >= 2


# -------------------------------------------------------------
# Local dataset builder
# -------------------------------------------------------------
def test_metadata_jsonl_append_and_parsing(tmp_path, monkeypatch):
    """Verifies append-only logging produces valid JSON lines and writes the image file."""
    fake_contrib_dir = str(tmp_path / "user_contributed")
    fake_jsonl = os.path.join(fake_contrib_dir, "metadata.jsonl")
    monkeypatch.setattr(app, "USER_CONTRIB_DIR", fake_contrib_dir)
    monkeypatch.setattr(app, "METADATA_JSONL", fake_jsonl)
    monkeypatch.setattr(app, "ALLOW_LOCAL_SAVE", True)

    state = app.init_booth_state()
    tight = np.full((60, 60), 77, dtype=np.uint8)
    state["Happy"] = {**app.empty_slot(), "score": 0.85, "crop": np.zeros((100, 100, 3), dtype=np.uint8),
                      "train_crop": tight, "label_source": "model_argmax"}

    feedback, banner = app.save_and_contribute(state, consent_given=True)
    assert "Saved **1**" in feedback
    assert os.path.exists(fake_jsonl)

    with open(fake_jsonl, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["emotion"] == "happy"
    assert record["score"] == 0.85
    assert record["consent_given"] is True
    assert record["label_source"] == "model_argmax"
    assert "verified_by_user" not in record
    assert record["timestamp"].endswith("+00:00")
    assert os.path.exists(record["image_path"])
    # The TIGHT grayscale crop (what the model classified) is what gets saved, not the padded RGB display crop
    saved = np.array(Image.open(record["image_path"]))
    assert saved.shape == (60, 60) and saved.max() == 77


def test_repeated_save_does_not_duplicate(tmp_path, monkeypatch):
    """Clicking Save twice must not write a second copy (would defeat the fine-tune volume gate)."""
    fake_contrib_dir = str(tmp_path / "user_contributed")
    monkeypatch.setattr(app, "USER_CONTRIB_DIR", fake_contrib_dir)
    monkeypatch.setattr(app, "METADATA_JSONL", os.path.join(fake_contrib_dir, "metadata.jsonl"))
    monkeypatch.setattr(app, "ALLOW_LOCAL_SAVE", True)

    state = app.init_booth_state()
    state["Happy"] = {**app.empty_slot(), "score": 0.85, "crop": np.zeros((40, 40, 3), dtype=np.uint8),
                      "train_crop": np.zeros((40, 40), dtype=np.uint8)}
    app.save_and_contribute(state, consent_given=True)
    feedback, _ = app.save_and_contribute(state, consent_given=True)
    assert "already saved" in feedback
    assert len(os.listdir(os.path.join(fake_contrib_dir, "happy"))) == 1
    assert state["Happy"]["saved"] is True


def test_samples_excluded_from_contribution(tmp_path, monkeypatch):
    """Verifies benchmark fallback samples are NEVER re-saved to the training pool."""
    fake_contrib_dir = str(tmp_path / "user_contributed")
    fake_jsonl = os.path.join(fake_contrib_dir, "metadata.jsonl")
    monkeypatch.setattr(app, "USER_CONTRIB_DIR", fake_contrib_dir)
    monkeypatch.setattr(app, "METADATA_JSONL", fake_jsonl)
    monkeypatch.setattr(app, "ALLOW_LOCAL_SAVE", True)

    state = app.init_booth_state()
    state["Disgust"] = {**app.empty_slot(), "score": 0.50, "crop": np.zeros((100, 100, 3), dtype=np.uint8), "is_sample": True}

    feedback, banner = app.save_and_contribute(state, consent_given=True)
    assert "benchmark samples are excluded" in feedback
    assert not os.path.exists(fake_jsonl)


def test_saving_disabled_on_shared_deployment(tmp_path, monkeypatch):
    """On a shared deployment (ALLOW_LOCAL_SAVE False) nothing is written, even with consent."""
    fake_contrib_dir = str(tmp_path / "user_contributed")
    monkeypatch.setattr(app, "USER_CONTRIB_DIR", fake_contrib_dir)
    monkeypatch.setattr(app, "METADATA_JSONL", os.path.join(fake_contrib_dir, "metadata.jsonl"))
    monkeypatch.setattr(app, "ALLOW_LOCAL_SAVE", False)

    state = app.init_booth_state()
    state["Happy"] = {**app.empty_slot(), "score": 0.9, "crop": np.zeros((80, 80, 3), dtype=np.uint8)}
    feedback, _ = app.save_and_contribute(state, consent_given=True)
    assert "disabled" in feedback
    assert not os.path.exists(fake_contrib_dir)
    assert "disabled" in app.get_local_dataset_stats()[2]


def test_local_dataset_stats(tmp_path, monkeypatch):
    """Verifies local active learning stats fall back to metadata.jsonl when no folders exist."""
    fake_contrib_dir = str(tmp_path / "user_contributed")
    fake_jsonl = os.path.join(fake_contrib_dir, "metadata.jsonl")
    os.makedirs(fake_contrib_dir, exist_ok=True)
    monkeypatch.setattr(app, "USER_CONTRIB_DIR", fake_contrib_dir)
    monkeypatch.setattr(app, "METADATA_JSONL", fake_jsonl)
    monkeypatch.setattr(app, "ALLOW_LOCAL_SAVE", True)

    with open(fake_jsonl, "w", encoding="utf-8") as f:
        for i in range(12):
            f.write(json.dumps({"emotion": "happy", "score": 0.9, "timestamp": "2026-09-18T10:00:00Z"}) + "\n")

    total, per_class, banner = app.get_local_dataset_stats()
    assert total == 12
    assert per_class["Happy"] == 12
    assert "12 Verified Faces Saved Locally" in banner
    assert "1/7 Classes Ready" in banner


def test_local_dataset_stats_counts_physical_files(tmp_path, monkeypatch):
    """Verifies get_local_dataset_stats prioritizes physical files for parity with fine_tune.py."""
    fake_contrib_dir = str(tmp_path / "user_contributed")
    os.makedirs(os.path.join(fake_contrib_dir, "happy"), exist_ok=True)
    os.makedirs(os.path.join(fake_contrib_dir, "surprise"), exist_ok=True)
    for i in range(3):
        Image.new("RGB", (48, 48)).save(os.path.join(fake_contrib_dir, "happy", f"h_{i}.png"))
    for i in range(2):
        Image.new("RGB", (48, 48)).save(os.path.join(fake_contrib_dir, "surprise", f"s_{i}.png"))
    monkeypatch.setattr(app, "USER_CONTRIB_DIR", fake_contrib_dir)
    monkeypatch.setattr(app, "ALLOW_LOCAL_SAVE", True)

    total, per_class, banner = app.get_local_dataset_stats()
    assert total == 5
    assert per_class["Happy"] == 3
    assert per_class["Surprise"] == 2
    assert "5 Verified Faces Saved Locally" in banner


# -------------------------------------------------------------
# fine_tune.py guardrails & data pipeline
# -------------------------------------------------------------
def test_fine_tune_guardrail_blocking(tmp_path, monkeypatch):
    """Verifies fine_tune.py blocks when volume requirements are not met."""
    monkeypatch.setattr(fine_tune, "USER_CONTRIB_DIR", str(tmp_path / "user_contributed"))
    passed, counts = fine_tune.check_guardrails(force=False)
    assert passed is False
    assert sum(counts.values()) == 0
    passed_forced, _ = fine_tune.check_guardrails(force=True)
    assert passed_forced is True


def test_fine_tune_requires_base_dataset(tmp_path, monkeypatch):
    """Without FER2013 on disk the script must exit with a clear message, not a traceback."""
    monkeypatch.setattr(fine_tune, "BASE_TRAIN_DIR", str(tmp_path / "nope" / "train"))
    monkeypatch.setattr(fine_tune, "TEST_DIR", str(tmp_path / "nope" / "test"))
    with pytest.raises(SystemExit):
        fine_tune.check_dataset_dirs()


def test_user_data_loaded_as_raw_pixels_like_inference(tmp_path, monkeypatch):
    """User faces load in the 0-255 range with the same grayscale + INTER_AREA path app.py uses."""
    contrib = tmp_path / "user_contributed"
    (contrib / "happy").mkdir(parents=True)
    rng = np.random.default_rng(3)
    face = rng.integers(0, 255, (96, 96), dtype=np.uint8)
    Image.fromarray(face).save(contrib / "happy" / "a.png")
    monkeypatch.setattr(fine_tune, "USER_CONTRIB_DIR", str(contrib))

    x, y = fine_tune.load_and_preprocess_user_data({"angry": 0, "happy": 1})
    assert x.shape == (1, 48, 48, 1) and y.shape == (1, 2)
    assert x.max() > 1.0
    assert y[0, 1] == 1.0
    expected = cv2.resize(face, (48, 48), interpolation=cv2.INTER_AREA).astype(np.float32)
    assert np.array_equal(x[0, :, :, 0], expected), "fine_tune resize differs from the inference path"


def test_duplicate_user_images_are_ignored(tmp_path, monkeypatch):
    """Byte-identical copies must not count toward the volume gate nor enter the dataset twice."""
    contrib = tmp_path / "user_contributed"
    (contrib / "happy").mkdir(parents=True)
    img = Image.fromarray(np.full((48, 48), 120, dtype=np.uint8))
    for i in range(12):
        img.save(contrib / "happy" / f"copy_{i}.png")       # 12 copies of ONE face
    Image.fromarray(np.full((48, 48), 50, dtype=np.uint8)).save(contrib / "happy" / "other.png")
    monkeypatch.setattr(fine_tune, "USER_CONTRIB_DIR", str(contrib))

    passed, counts = fine_tune.check_guardrails(force=False)
    assert counts["happy"] == 2
    assert passed is False
    x, _ = fine_tune.load_and_preprocess_user_data({"happy": 0})
    assert len(x) == 2


def test_blended_generator_with_real_augmentation_keeps_user_faces_visible():
    """
    REGRESSION TEST for the black-image bug: the DEFAULT user augmentation pipeline
    (brightness_range + rescale) must yield user samples in (0, 1] that still look like
    the input, not all-zero squares.
    """
    batch_size = 16
    rng = np.random.default_rng(0)
    user_x = rng.uniform(60, 230, size=(10, 48, 48, 1)).astype(np.float32)  # raw 0-255 faces
    user_y = np.zeros((10, 7), dtype=np.float32)
    user_y[:, 6] = 1.0

    def dummy_base_gen():
        while True:
            bx = np.full((batch_size, 48, 48, 1), 0.1, dtype=np.float32)
            by = np.zeros((batch_size, 7), dtype=np.float32)
            by[:, 0] = 1.0
            yield bx, by

    blended_gen = fine_tune.create_blended_generator(
        dummy_base_gen(), user_x, user_y, user_datagen=None, batch_size=batch_size, user_ratio=0.25
    )
    batch_x, batch_y = next(blended_gen)

    assert batch_x.shape == (batch_size, 48, 48, 1)
    assert batch_y.shape == (batch_size, 7)
    user_mask = batch_y[:, 6] == 1.0
    assert user_mask.sum() == 4, "Expected 4 user samples per 16-sample batch"
    user_pixels = batch_x[user_mask]
    assert user_pixels.max() <= 1.0, "User data was not rescaled to [0, 1]"
    assert user_pixels.mean() > 0.15, "User data collapsed to black (brightness_range on pre-normalized input)"
    # Base samples untouched
    assert np.allclose(batch_x[~user_mask], 0.1)


def test_blended_generator_applies_class_weights_as_sample_weights():
    """Keras 3 rejects class_weight for generators, so the balancing must ride along as sample weights."""
    user_x = np.full((4, 48, 48, 1), 128, dtype=np.float32)
    user_y = np.zeros((4, 7), dtype=np.float32); user_y[:, 1] = 1.0   # disgust

    def base():
        while True:
            by = np.zeros((6, 7), dtype=np.float32); by[:, 3] = 1.0    # happy
            yield np.full((6, 48, 48, 1), 0.5, dtype=np.float32), by

    cw = {1: 9.4, 3: 0.57}
    gen = fine_tune.create_blended_generator(base(), user_x, user_y, batch_size=8, user_ratio=0.25, class_weight=cw)
    x, y, w = next(gen)
    assert x.shape[0] == y.shape[0] == w.shape[0] == 8
    # FER2013 share is class-balanced; user samples (disgust here) are NOT up-weighted 9.4x
    assert np.allclose(w[y[:, 3] == 1], 0.57)
    assert np.allclose(w[y[:, 1] == 1], 1.0)

    # Without user data the base stream is still weighted
    x, y, w = next(fine_tune.create_blended_generator(base(), np.empty((0, 48, 48, 1)), np.empty((0, 7)), class_weight=cw))
    assert np.allclose(w, 0.57)


def test_train_scope_limits_trainable_layers():
    """Default fine-tune scope trains only the classifier head; BatchNorm is frozen in every scope."""
    m = tf.keras.models.load_model(fine_tune.MODEL_PATH)
    total = m.count_params()
    n_head = fine_tune.set_train_scope(m, "head")
    head_params = sum(int(tf.size(w)) for w in m.trainable_weights)
    assert n_head == 1 and head_params == 7 * 128 + 7
    n_all = fine_tune.set_train_scope(m, "all")
    all_params = sum(int(tf.size(w)) for w in m.trainable_weights)
    assert n_all > n_head and head_params < all_params < total     # BN gamma/beta stay frozen
    assert all(not l.trainable for l in m.layers if l.__class__.__name__ == "BatchNormalization")
    with pytest.raises(ValueError):
        fine_tune.set_train_scope(m, "everything")


def test_split_user_holdout_is_per_class_and_deterministic():
    """20% of each class with >= 5 samples is held out; tiny classes stay in training."""
    x = np.arange(25, dtype=np.float32).reshape(25, 1, 1, 1)
    y = np.zeros((25, 7), dtype=np.float32)
    y[:20, 3] = 1.0   # 20 happy
    y[20:, 1] = 1.0   # 5 disgust... but only 4 below threshold in next line
    y[24, 1] = 0.0; y[24, 2] = 1.0  # make disgust 4 (no holdout), fear 1 (no holdout)

    xt, yt, xh, yh = fine_tune.split_user_holdout(x, y)
    assert len(xh) == 4 and np.all(np.argmax(yh, axis=1) == 3)
    assert len(xt) == 21
    assert set(xt.ravel()) | set(xh.ravel()) == set(x.ravel())
    xt2, _, xh2, _ = fine_tune.split_user_holdout(x, y)
    assert np.array_equal(xh, xh2)


def test_promotion_decision_catches_per_class_collapse():
    """A candidate whose overall accuracy barely dips but whose Sad recall collapses is rejected."""
    labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    base = {"accuracy": 0.573, "macro_f1": 0.545, "recall": [0.54, 0.61, 0.30, 0.75, 0.61, 0.42, 0.75]}
    collapsed = {"accuracy": 0.565, "macro_f1": 0.540,
                 "recall": [0.54, 0.56, 0.26, 0.81, 0.61, 0.33, 0.75]}   # Sad -8.7pp, Disgust -5.4pp, Fear -4pp
    promote, reasons = fine_tune.promotion_decision(base, collapsed, 0.5, 0.5, labels)
    assert promote is False
    assert any("sad" in r for r in reasons) and any("disgust" in r for r in reasons)

    fine_ = {"accuracy": 0.571, "macro_f1": 0.544, "recall": [0.53, 0.60, 0.29, 0.76, 0.61, 0.41, 0.75]}
    promote, reasons = fine_tune.promotion_decision(base, fine_, 0.5, 0.6, labels)
    assert promote is True and reasons == []

    # Regression on the user's own held-out faces is a hard no
    promote, reasons = fine_tune.promotion_decision(base, fine_, 0.6, 0.5, labels)
    assert promote is False and "held-out" in reasons[0]


def test_original_model_preservation(tmp_path, monkeypatch):
    """ensure_original_baseline copies the production model once and never overwrites it."""
    model_path = tmp_path / "emotion_model.keras"
    original_path = tmp_path / "emotion_model_original.keras"
    model_path.write_bytes(b"v1")
    monkeypatch.setattr(fine_tune, "MODEL_PATH", str(model_path))
    monkeypatch.setattr(fine_tune, "ORIGINAL_MODEL_PATH", str(original_path))

    fine_tune.ensure_original_baseline()
    assert original_path.read_bytes() == b"v1"
    model_path.write_bytes(b"v2")
    fine_tune.ensure_original_baseline()
    assert original_path.read_bytes() == b"v1"


# -------------------------------------------------------------
# Inference & annotation
# -------------------------------------------------------------
def test_compiled_inference_matches_eager_model():
    """predict_probs (tf.function) must return the same softmax as the eager model."""
    roi = np.random.default_rng(1).uniform(0, 1, (1, 48, 48, 1)).astype(np.float32)
    fast = app.predict_probs(roi)
    eager = app.model(tf.constant(roi), training=False).numpy()[0]
    assert fast.shape == (7,)
    assert abs(float(fast.sum()) - 1.0) < 1e-4
    assert np.allclose(fast, eager, atol=1e-5)


def test_ci_backward_compatibility_alias():
    """predict_emotion alias returns (annotated, confidences) and rejects blank images."""
    assert hasattr(app, "predict_emotion")
    annotated, confs = app.predict_emotion(np.zeros((120, 120, 3), dtype=np.uint8))
    assert len(confs) == 0

    sample_path = os.path.join("assets", "samples", "happy.jpg")
    sample_img = cv2.imread(sample_path)
    _, sample_confs = app.predict_emotion(sample_img)
    assert len(sample_confs) == 7


def test_whole_image_fallback_only_for_tiny_crops():
    """Large images with no detectable face must NOT be classified (no 'not a face' class)."""
    rng = np.random.default_rng(0)
    noise_large = rng.uniform(0, 255, (200, 200, 3)).astype(np.uint8)
    _, confs, status = app.process_image(noise_large)
    assert confs == {}
    assert "No face detected" in status

    tiny = cv2.imread(os.path.join("assets", "samples", "sad.jpg"))
    _, confs, status = app.process_image(tiny)
    assert len(confs) == 7
    assert "fallback" in status


def test_annotate_frame_preserves_input_frame(monkeypatch):
    """Verifies annotate_frame does not mutate the input frame_bgr in-place."""
    fake_frame = np.full((120, 120, 3), 128, dtype=np.uint8)
    original_copy = fake_frame.copy()

    class DummyCascade:
        def detectMultiScale(self, *args, **kwargs):
            return [(20, 20, 60, 60)]

    monkeypatch.setattr(app, "face_cascade", DummyCascade())
    annotated, confs, _, face_infos = app.annotate_frame(fake_frame)
    assert np.array_equal(fake_frame, original_copy)
    assert not np.array_equal(annotated, original_copy)
    assert len(face_infos) == 1


def test_booth_crops_are_untainted(monkeypatch):
    """Verifies that facial crops captured in Photo Booth have no cyan overlay pixels."""
    test_frame_rgb = np.full((120, 120, 3), 180, dtype=np.uint8)

    class DummyCascade:
        def detectMultiScale(self, *args, **kwargs):
            return [(20, 20, 60, 60)]

    monkeypatch.setattr(app, "face_cascade", DummyCascade())
    dummy_preds = np.zeros(7, dtype=np.float32)
    dummy_preds[3] = 0.95  # 'happy'
    monkeypatch.setattr(app, "predict_probs", lambda roi: dummy_preds)

    annotated, status, gallery, new_state, ema = app.process_booth_frame(test_frame_rgb, app.init_booth_state())
    captured_crop = new_state["Happy"]["crop"]
    assert captured_crop is not None
    cyan_pixels = np.all(captured_crop == [255, 230, 0], axis=-1)
    assert not np.any(cyan_pixels)
    assert abs(float(app.ema_get(ema)[3]) - 0.95) < 1e-6
    assert new_state["Happy"]["model_label"] == "Happy"
    # The tight training crop is the exact Haar box (60x60), grayscale, unpadded
    assert new_state["Happy"]["train_crop"].shape == (60, 60)
    assert new_state["Happy"]["label_source"] == "model_argmax"
    # Padded display crop is bigger than the box
    assert captured_crop.shape[0] > 60


def test_sample_fill_does_not_block_live_capture(monkeypatch):
    """A 'Fill Fear' sample must be replaced by any live Fear capture above the floor."""
    state = app.init_booth_state()
    state, _, _ = app.use_sample_face("Fear", state)
    faces = [{"label": "Fear", "score": 0.34, "box": (10, 10, 50, 50), "train_crop": np.zeros((50, 50), np.uint8)}]
    monkeypatch.setattr(app, "annotate_frame", lambda f, smoothed_preds=None, alpha=0.7: (f, {}, smoothed_preds, faces))
    _, _, gallery, state, _ = app.process_booth_frame(np.full((120, 120, 3), 128, dtype=np.uint8), state)
    assert state["Fear"]["is_sample"] is False
    assert state["Fear"]["score"] == 0.34
    assert "Fear: 34%" in gallery[0][1]


def test_relabel_slot_moves_capture_and_marks_correction():
    state = app.init_booth_state()
    state["Surprise"] = {**app.empty_slot(), "score": 0.6, "crop": np.zeros((30, 30, 3), np.uint8),
                         "train_crop": np.zeros((30, 30), np.uint8), "label_source": "model_argmax",
                         "model_label": "Surprise"}
    state, status, gallery, msg = app.relabel_slot(state, "Surprise", "Fear")
    assert state["Surprise"]["crop"] is None
    assert state["Fear"]["crop"] is not None
    assert state["Fear"]["label_source"] == "user_corrected"
    assert state["Fear"]["model_label"] == "Surprise"
    # The 60% belonged to the model's Surprise call; the corrected tile must not claim it
    assert gallery[0][1] == "Fear (relabelled)"
    # Samples and empty slots cannot be relabelled
    state, _, _, msg = app.relabel_slot(state, "Happy", "Sad")
    assert "no live capture" in msg


def test_relabel_after_save_retracts_the_wrongly_labelled_file(tmp_path, monkeypatch):
    """Save -> relabel -> Save must leave ONE file, under the corrected label, with an audit trail."""
    fake_contrib_dir = str(tmp_path / "user_contributed")
    fake_jsonl = os.path.join(fake_contrib_dir, "metadata.jsonl")
    monkeypatch.setattr(app, "USER_CONTRIB_DIR", fake_contrib_dir)
    monkeypatch.setattr(app, "METADATA_JSONL", fake_jsonl)
    monkeypatch.setattr(app, "ALLOW_LOCAL_SAVE", True)

    state = app.init_booth_state()
    state["Surprise"] = {**app.empty_slot(), "score": 0.6, "crop": np.zeros((30, 30, 3), np.uint8),
                         "train_crop": np.full((30, 30), 9, np.uint8), "label_source": "model_argmax",
                         "model_label": "Surprise"}
    app.save_and_contribute(state, consent_given=True)
    assert len(os.listdir(os.path.join(fake_contrib_dir, "surprise"))) == 1

    state, _, _, msg = app.relabel_slot(state, "Surprise", "Fear")
    assert "deleted" in msg
    assert os.listdir(os.path.join(fake_contrib_dir, "surprise")) == []

    feedback, _ = app.save_and_contribute(state, consent_given=True)
    assert "Saved **1**" in feedback
    assert len(os.listdir(os.path.join(fake_contrib_dir, "fear"))) == 1

    records = [json.loads(l) for l in open(fake_jsonl, encoding="utf-8") if l.strip()]
    assert [r.get("event", r.get("emotion")) for r in records] == ["surprise", "relabel_retraction", "fear"]
    assert records[2]["label_source"] == "user_corrected"
    assert records[2]["score"] is None and records[2]["model_label"] == "surprise" and records[2]["model_score"] == 0.6
    # Stats count samples only, not the retraction event
    total, per_class, _ = app.get_local_dataset_stats()
    assert total == 1 and per_class["Fear"] == 1 and per_class["Surprise"] == 0


def test_multi_face_temporal_smoothing_isolation(monkeypatch):
    """Verifies temporal smoothing is applied only to the primary face without cross-talk."""
    fake_frame = np.full((200, 200, 3), 128, dtype=np.uint8)

    class MultiFaceCascade:
        def detectMultiScale(self, *args, **kwargs):
            return [(10, 10, 40, 40), (60, 60, 80, 80)]

    monkeypatch.setattr(app, "face_cascade", MultiFaceCascade())

    call_count = [0]
    def mock_predict(roi):
        call_count[0] += 1
        p = np.zeros(7, dtype=np.float32)
        if call_count[0] % 2 == 1:
            p[3] = 0.90  # happy for primary (80x80)
        else:
            p[0] = 0.85  # angry for secondary (40x40)
        return p

    monkeypatch.setattr(app, "predict_probs", mock_predict)
    prev_smoothed = np.zeros(7, dtype=np.float32)
    prev_smoothed[3] = 0.50
    _, confs, new_smoothed, face_infos = app.annotate_frame(fake_frame, smoothed_preds=prev_smoothed, alpha=0.5)
    assert abs(new_smoothed[3] - 0.70) < 1e-4
    assert new_smoothed[0] == 0.0
    assert face_infos[0]["box"][2] == 80  # largest face is primary


def test_booth_only_captures_primary_face(monkeypatch):
    """A bystander (secondary face) must never fill a booth slot."""
    frame = np.full((200, 200, 3), 128, dtype=np.uint8)
    faces = [
        {"label": "Happy", "score": 0.9, "box": (60, 60, 80, 80), "train_crop": np.zeros((80, 80), np.uint8)},
        {"label": "Angry", "score": 0.9, "box": (10, 10, 40, 40), "train_crop": np.zeros((40, 40), np.uint8)},
    ]
    monkeypatch.setattr(app, "annotate_frame", lambda f, smoothed_preds=None, alpha=0.7: (f, {}, smoothed_preds, faces))
    _, _, _, state, _ = app.process_booth_frame(frame, app.init_booth_state())
    assert state["Happy"]["crop"] is not None
    assert state["Angry"]["crop"] is None


def test_live_stream_carries_ema_state(monkeypatch):
    """The live webcam tab must smooth across calls via the returned EMA state."""
    frame = np.full((120, 120, 3), 128, dtype=np.uint8)

    class DummyCascade:
        def detectMultiScale(self, *args, **kwargs):
            return [(20, 20, 60, 60)]

    monkeypatch.setattr(app, "face_cascade", DummyCascade())
    p = np.zeros(7, dtype=np.float32); p[3] = 1.0
    monkeypatch.setattr(app, "predict_probs", lambda roi: p)

    _, confs1, ema1 = app.process_live_frame(frame, None)
    assert abs(confs1["Happy"] - 1.0) < 1e-6

    q = np.zeros(7, dtype=np.float32); q[0] = 1.0
    monkeypatch.setattr(app, "predict_probs", lambda roi: q)
    _, confs2, ema2 = app.process_live_frame(frame, ema1)
    # alpha=0.70: 0.7*angry + 0.3*previous happy
    assert abs(confs2["Angry"] - 0.70) < 1e-6
    assert abs(confs2["Happy"] - 0.30) < 1e-6

    # A brief detection dropout keeps the smoother (so the next frame is not a raw, unsmoothed spike)...
    monkeypatch.setattr(app, "face_cascade", type("NoFace", (), {"detectMultiScale": lambda self, *a, **k: []})())
    _, confs3, ema3 = app.process_live_frame(frame, ema2)
    assert confs3 == {} and np.array_equal(app.ema_get(ema3), app.ema_get(ema2)) and ema3["misses"] == 1
    # ...but once the face has really gone the smoother is dropped, so the next person starts clean
    state = ema3
    for _ in range(app.EMA_RESET_AFTER_MISSES - 1):
        _, _, state = app.process_live_frame(frame, state)
    assert app.ema_get(state) is None
    monkeypatch.setattr(app, "face_cascade", DummyCascade())
    _, confs4, state = app.process_live_frame(frame, state)
    assert abs(confs4["Angry"] - 1.0) < 1e-6 and state["misses"] == 0


def test_dynamic_peak_expression_tracking(monkeypatch):
    """Verifies that booth captures at the 25% floor and upgrades when beating a personal best."""
    state = app.init_booth_state()
    frame = np.full((120, 120, 3), 128, dtype=np.uint8)

    def mock(faces):
        for face in faces:
            face.setdefault("train_crop", np.zeros((50, 50), np.uint8))
        return lambda f, smoothed_preds=None, alpha=0.7: (f, {}, smoothed_preds, faces)

    monkeypatch.setattr(app, "annotate_frame", mock([{"label": "Surprise", "score": 0.20, "box": (10, 10, 50, 50)}]))
    _, banner, gallery, state_out, _ = app.process_booth_frame(frame, state)
    assert state_out["Surprise"]["score"] == 0.0
    assert state_out["Surprise"]["crop"] is None

    monkeypatch.setattr(app, "annotate_frame", mock([{"label": "Surprise", "score": 0.44, "box": (10, 10, 50, 50)}]))
    _, banner, gallery, state_out, _ = app.process_booth_frame(frame, state_out)
    assert state_out["Surprise"]["score"] == 0.44
    assert state_out["Surprise"]["crop"] is not None
    assert len(gallery) == 1

    monkeypatch.setattr(app, "annotate_frame", mock([{"label": "Surprise", "score": 0.62, "box": (10, 10, 50, 50)}]))
    _, banner, gallery, state_out, _ = app.process_booth_frame(frame, state_out)
    assert state_out["Surprise"]["score"] == 0.62
    assert "Surprise: 62%" in gallery[0][1]

    monkeypatch.setattr(app, "annotate_frame", mock([{"label": "Surprise", "score": 0.50, "box": (10, 10, 50, 50)}]))
    _, banner, gallery, state_out, _ = app.process_booth_frame(frame, state_out)
    assert state_out["Surprise"]["score"] == 0.62


def test_generate_photo_strip_returns_pil_image():
    """Verifies generate_photo_strip returns a PIL Image in-memory without disk writes."""
    state = app.init_booth_state()
    state["Happy"]["crop"] = np.full((100, 100, 3), 200, dtype=np.uint8)
    state["Happy"]["score"] = 0.95
    state, _, _ = app.use_sample_face("Disgust", state)
    strip = app.generate_photo_strip(state)
    assert isinstance(strip, Image.Image)
    assert strip.width > 0 and strip.height > 0
    assert app.slot_caption("Disgust", state["Disgust"]) == "Disgust (Sample)"
    assert app.slot_caption("Happy", state["Happy"]) == "Happy: 95%"


def test_gradio_cache_is_purged():
    """Gradio keeps its own copies of uploads/outputs; they must be on a purge schedule too."""
    assert app.demo.delete_cache == app.GRADIO_CACHE_TTL


def test_reset_booth_clears_ema():
    state, status, gallery, ema = app.reset_booth()
    assert gallery == [] and ema is None and "0 / 7" in status


# -------------------------------------------------------------
# Video upload
# -------------------------------------------------------------
def _write_synthetic_video(path, n_frames=30, size=(320, 240)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, size)
    for i in range(n_frames):
        frame = np.full((size[1], size[0], 3), (i * 8) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_process_video_file_respects_frame_cap(tmp_path, monkeypatch):
    """Uploaded videos are bounded: only MAX_VIDEO_FRAMES frames are written to the output."""
    src = tmp_path / "in.mp4"
    _write_synthetic_video(src, n_frames=30)
    monkeypatch.setattr(app, "MAX_VIDEO_FRAMES", 12)
    monkeypatch.setattr(app, "VIDEO_OUT_DIR", str(tmp_path / "out"))
    monkeypatch.setattr(app.gr, "Warning", lambda *a, **k: None)

    out_path = app.process_video_file(str(src), progress=lambda *a, **k: None)
    assert out_path and os.path.exists(out_path)
    assert os.path.dirname(out_path) == str(tmp_path / "out")
    cap = cv2.VideoCapture(out_path)
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 12
    cap.release()


def test_process_video_file_downscales_wide_videos(tmp_path, monkeypatch):
    src = tmp_path / "wide.mp4"
    _write_synthetic_video(src, n_frames=3, size=(1280, 720))
    monkeypatch.setattr(app, "VIDEO_OUT_DIR", str(tmp_path / "out"))
    out_path = app.process_video_file(str(src), progress=lambda *a, **k: None)
    cap = cv2.VideoCapture(out_path)
    assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 640
    cap.release()


def test_cleanup_old_videos(tmp_path):
    out_dir = tmp_path / "vids"
    out_dir.mkdir()
    old = out_dir / "old.mp4"; old.write_bytes(b"x")
    new = out_dir / "new.mp4"; new.write_bytes(b"x")
    os.utime(old, (0, 0))
    app._cleanup_old_videos(str(out_dir), ttl=60)
    assert not old.exists() and new.exists()


# -------------------------------------------------------------
# Explainability & drowsiness
# -------------------------------------------------------------
def test_explain_samples_fallback(tmp_path):
    """explain.py runs on a clean clone using assets/samples/ and writes to the given directory."""
    assert len(os.listdir("assets/samples")) >= 7
    out = explain.main(output_dir=str(tmp_path / "gradcam"))
    assert os.path.exists(out)
    assert os.path.dirname(out) == str(tmp_path / "gradcam")


class _LM:
    def __init__(self, x, y):
        self.x, self.y = x, y


def test_face_landmarker_constructs_and_measures_ear_on_real_face():
    """
    REGRESSION TEST: mediapipe>=1.0 removed mp.solutions; monitor.py must run on the pinned
    version. Builds the Tasks-API landmarker, runs it on a real (upscaled) face and a blank
    frame, and checks the EAR of the open-eyed sample is above the drowsiness threshold.
    """
    landmarker = monitor.create_landmarker()
    try:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        assert monitor.detect_landmarks(landmarker, blank, 0) == []

        face = cv2.cvtColor(cv2.resize(cv2.imread("assets/samples/neutral.jpg"), (256, 256),
                                       interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
        canvas = np.full((480, 640, 3), 128, dtype=np.uint8)
        canvas[112:368, 192:448] = face
        faces = monitor.detect_landmarks(landmarker, canvas, 33)
        assert len(faces) == 1
        assert len(faces[0]) >= 468
        ear = monitor.average_ear(faces[0], 640, 480)
        assert monitor.EAR_THRESHOLD < ear < 0.6
    finally:
        landmarker.close()


def test_eye_aspect_ratio_open_vs_closed():
    """EAR is ~0 for a closed eye and well above the drowsiness threshold for an open one."""
    # p1..p6 laid out as in calculate_ear: p1/p4 horizontal corners, p2,p3 upper, p6,p5 lower
    def eye(open_h):
        return [_LM(0.0, 0.5), _LM(0.3, 0.5 - open_h), _LM(0.7, 0.5 - open_h),
                _LM(1.0, 0.5), _LM(0.7, 0.5 + open_h), _LM(0.3, 0.5 + open_h)]
    idx = [0, 1, 2, 3, 4, 5]
    ear_open = monitor.calculate_ear(eye(0.15), idx, 100, 100)
    ear_closed = monitor.calculate_ear(eye(0.0), idx, 100, 100)
    assert ear_open > monitor.EAR_THRESHOLD
    assert ear_closed < monitor.EAR_THRESHOLD
    assert abs(ear_open - 0.30) < 1e-6
    # Degenerate (zero-width) eye returns the safe default instead of dividing by zero
    assert monitor.calculate_ear([_LM(0.5, 0.5)] * 6, idx, 100, 100) == 0.3
