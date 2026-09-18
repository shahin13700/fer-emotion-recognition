"""
Automated Test Suite for EdgeVision Active Learning, Photo Booth, & Guardrails
"""

import os
import json
import shutil
import tempfile
import cv2
import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import app
import fine_tune
import explain
from PIL import Image


def test_empirical_thresholds_defined():
    """Verifies that all 7 emotions have empirical thresholds properly defined."""
    assert len(app.EMPIRICAL_THRESHOLDS) == 7
    expected_keys = {"Happy", "Surprise", "Neutral", "Angry", "Fear", "Sad", "Disgust"}
    assert set(app.EMPIRICAL_THRESHOLDS.keys()) == expected_keys

    # Verify key empirical values
    assert app.EMPIRICAL_THRESHOLDS["Happy"] == 0.70
    assert app.EMPIRICAL_THRESHOLDS["Surprise"] == 0.65
    assert app.EMPIRICAL_THRESHOLDS["Disgust"] == 0.50
    assert app.EMPIRICAL_THRESHOLDS["Fear"] == 0.38
    assert app.EMPIRICAL_THRESHOLDS["Sad"] == 0.37


def test_init_booth_state():
    """Verifies initial photo booth state has all 7 emotion slots empty."""
    state = app.init_booth_state()
    assert len(state) == 7
    for emo, data in state.items():
        assert data["score"] == 0.0
        assert data["crop"] is None
        assert data["is_sample"] is False


def test_use_sample_face_disgust_and_fear():
    """Verifies UX fail-safe loads sample faces without crashing and applies upscaling."""
    state = app.init_booth_state()

    # Load Disgust sample
    updated_state, status, gallery = app.use_sample_face("Disgust", state)
    assert updated_state["Disgust"]["crop"] is not None
    assert updated_state["Disgust"]["is_sample"] is True
    assert updated_state["Disgust"]["score"] == app.EMPIRICAL_THRESHOLDS["Disgust"]
    # Check that upscale was applied for clean preview rendering
    h, w, c = updated_state["Disgust"]["crop"].shape
    assert h >= 100 and w >= 100
    assert len(gallery) >= 1

    # Load Fear sample
    updated_state, status, gallery = app.use_sample_face("Fear", updated_state)
    assert updated_state["Fear"]["crop"] is not None
    assert updated_state["Fear"]["is_sample"] is True
    assert updated_state["Fear"]["score"] == app.EMPIRICAL_THRESHOLDS["Fear"]
    assert len(gallery) >= 2


def test_metadata_jsonl_append_and_parsing(tmp_path, monkeypatch):
    """Verifies append-only atomic logging produces valid JSON lines."""
    fake_contrib_dir = str(tmp_path / "user_contributed")
    fake_jsonl = os.path.join(fake_contrib_dir, "metadata.jsonl")

    monkeypatch.setattr(app, "USER_CONTRIB_DIR", fake_contrib_dir)
    monkeypatch.setattr(app, "METADATA_JSONL", fake_jsonl)

    state = app.init_booth_state()
    dummy_crop = np.zeros((100, 100, 3), dtype=np.uint8)

    # Populate state with a genuine high-confidence Happy face
    state["Happy"] = {
        "score": 0.85,
        "crop": dummy_crop,
        "is_sample": False
    }

    feedback, banner = app.save_and_contribute(state, consent_given=True)
    assert "Successfully saved" in feedback
    assert os.path.exists(fake_jsonl)

    # Verify JSONL lines are independently valid JSON
    with open(fake_jsonl, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["emotion"] == "happy"
    assert record["score"] == 0.85
    assert record["verified_by_user"] is True
    assert os.path.exists(record["image_path"])


def test_samples_excluded_from_contribution(tmp_path, monkeypatch):
    """Verifies benchmark fallback samples are NEVER re-saved to training pool."""
    fake_contrib_dir = str(tmp_path / "user_contributed")
    fake_jsonl = os.path.join(fake_contrib_dir, "metadata.jsonl")

    monkeypatch.setattr(app, "USER_CONTRIB_DIR", fake_contrib_dir)
    monkeypatch.setattr(app, "METADATA_JSONL", fake_jsonl)

    state = app.init_booth_state()
    dummy_crop = np.zeros((100, 100, 3), dtype=np.uint8)

    # Benchmark sample face
    state["Disgust"] = {
        "score": 0.50,
        "crop": dummy_crop,
        "is_sample": True
    }

    feedback, banner = app.save_and_contribute(state, consent_given=True)
    assert "benchmark samples are excluded" in feedback
    if os.path.exists(fake_jsonl):
        with open(fake_jsonl, "r", encoding="utf-8") as f:
            assert len(f.readlines()) == 0


def test_local_dataset_stats(tmp_path, monkeypatch):
    """Verifies local active learning stats calculate correctly."""
    fake_contrib_dir = str(tmp_path / "user_contributed")
    fake_jsonl = os.path.join(fake_contrib_dir, "metadata.jsonl")
    os.makedirs(fake_contrib_dir, exist_ok=True)

    monkeypatch.setattr(app, "USER_CONTRIB_DIR", fake_contrib_dir)
    monkeypatch.setattr(app, "METADATA_JSONL", fake_jsonl)

    # Write 12 dummy records for happy
    with open(fake_jsonl, "w", encoding="utf-8") as f:
        for i in range(12):
            entry = {"emotion": "happy", "score": 0.9, "timestamp": "2026-09-18T10:00:00Z"}
            f.write(json.dumps(entry) + "\n")

    total, per_class, banner = app.get_local_dataset_stats()
    assert total == 12
    assert per_class["Happy"] == 12
    assert "12 Verified Faces Saved Locally" in banner
    assert "1/7 Classes Ready" in banner


def test_fine_tune_guardrail_blocking(tmp_path, monkeypatch):
    """Verifies fine_tune.py blocks when volume requirements are not met."""
    fake_contrib_dir = str(tmp_path / "user_contributed")
    monkeypatch.setattr(fine_tune, "USER_CONTRIB_DIR", fake_contrib_dir)

    passed, counts = fine_tune.check_guardrails(force=False)
    assert passed is False
    assert sum(counts.values()) == 0

    # With force=True, it should allow bypass
    passed_forced, _ = fine_tune.check_guardrails(force=True)
    assert passed_forced is True


def test_ci_backward_compatibility_alias():
    """Verifies predict_emotion alias exists and returns exact expected confidences."""
    assert hasattr(app, "predict_emotion")
    # 1. Blank image returns 0 confidences
    dummy_img = np.zeros((120, 120, 3), dtype=np.uint8)
    annotated, confs = app.predict_emotion(dummy_img)
    assert len(confs) == 0, f"Expected 0 confidences on blank image, got {len(confs)}"

    # 2. Benchmark sample crop returns full 7 class probabilities via headshot fallback
    sample_path = os.path.join("assets", "samples", "happy.jpg")
    if os.path.exists(sample_path):
        sample_img = cv2.imread(sample_path)
        _, sample_confs = app.predict_emotion(sample_img)
        assert len(sample_confs) == 7, f"Expected 7 confidences, got {len(sample_confs)}"


def test_blended_generator_actually_mixes_user_data():
    """
    CRITICAL TEST: Verifies create_blended_generator actively combines
    user data with base data into the output training batches.
    """
    batch_size = 16
    user_ratio = 0.25

    # Mock base generator that yields distinct recognizable labels (value 1.0)
    def dummy_base_gen():
        while True:
            bx = np.full((batch_size, 48, 48, 1), 0.1, dtype=np.float32)
            by = np.zeros((batch_size, 7), dtype=np.float32)
            by[:, 0] = 1.0  # class 0
            yield bx, by

    # Mock user data with distinct recognizable signature (value 0.9, class 6)
    num_user = 10
    user_x = np.full((num_user, 48, 48, 1), 0.9, dtype=np.float32)
    user_y = np.zeros((num_user, 7), dtype=np.float32)
    user_y[:, 6] = 1.0  # class 6 (Disgust)

    datagen = ImageDataGenerator()

    blended_gen = fine_tune.create_blended_generator(
        dummy_base_gen(),
        user_x,
        user_y,
        datagen,
        batch_size=batch_size,
        user_ratio=user_ratio
    )

    # Fetch a batch from the blended generator
    batch_x, batch_y = next(blended_gen)

    # Verify batch shape
    assert batch_x.shape == (batch_size, 48, 48, 1)
    assert batch_y.shape == (batch_size, 7)

    # Verify user samples are actually in the batch!
    # Expected user count = max(1, int(16 * 0.25)) = 4
    # User samples have class 6 activated
    user_samples_in_batch = np.sum(batch_y[:, 6] == 1.0)
    assert user_samples_in_batch == 4, f"Expected 4 user samples in batch, found {user_samples_in_batch}"

    # Verify user samples are NOT double-rescaled
    user_mask = (batch_y[:, 6] == 1.0)
    assert np.all(batch_x[user_mask] > 0.5), "User data was double-rescaled (values dropped near zero)!"



def test_original_model_preservation():
    """Verifies that an immutable original model baseline exists on disk."""
    fine_tune.ensure_original_baseline()
    assert os.path.exists(fine_tune.ORIGINAL_MODEL_PATH)
    assert os.path.getsize(fine_tune.ORIGINAL_MODEL_PATH) > 800000  # ~817 KB


def test_annotate_frame_preserves_input_frame(monkeypatch):
    """Verifies annotate_frame does not mutate the input frame_bgr in-place."""
    fake_frame = np.full((120, 120, 3), 128, dtype=np.uint8)
    original_copy = fake_frame.copy()

    # Mock face_cascade to detect a face so annotations are drawn
    class DummyCascade:
        def detectMultiScale(self, *args, **kwargs):
            return [(20, 20, 60, 60)]

    monkeypatch.setattr(app, "face_cascade", DummyCascade())
    annotated, confs, _, face_infos = app.annotate_frame(fake_frame)

    # Input frame must remain unmodified
    assert np.array_equal(fake_frame, original_copy), "annotate_frame modified input frame in-place!"
    # Annotated frame must have differences (drawn bounding box and badge)
    assert not np.array_equal(annotated, original_copy), "annotated frame should contain drawings!"


def test_booth_crops_are_untainted(monkeypatch):
    """Verifies that facial crops captured in Photo Booth have no cyan overlay pixels."""
    test_frame_rgb = np.full((120, 120, 3), 180, dtype=np.uint8)

    class DummyCascade:
        def detectMultiScale(self, *args, **kwargs):
            return [(20, 20, 60, 60)]

    monkeypatch.setattr(app, "face_cascade", DummyCascade())
    booth_state = app.init_booth_state()

    # Mock model to return high confidence for 'happy' so it triggers a crop capture
    dummy_preds = np.zeros(7, dtype=np.float32)
    dummy_preds[3] = 0.95  # 'happy'
    monkeypatch.setattr(app, "model", lambda x, training=False: tf.constant([dummy_preds]))

    annotated, status, gallery, new_state = app.process_booth_frame(test_frame_rgb, booth_state)
    captured_crop = new_state["Happy"]["crop"]
    assert captured_crop is not None

    # Cyan color in RGB is (255, 230, 0)
    cyan_pixels = np.all(captured_crop == [255, 230, 0], axis=-1)
    assert not np.any(cyan_pixels), "Captured crop contains drawn cyan bounding box pixels!"


def test_multi_face_temporal_smoothing_isolation(monkeypatch):
    """Verifies temporal smoothing is applied only to primary face without cross-talk."""
    fake_frame = np.full((200, 200, 3), 128, dtype=np.uint8)

    # Two faces: Face 1 (smaller: 40x40), Face 2 (larger: 80x80)
    class MultiFaceCascade:
        def detectMultiScale(self, *args, **kwargs):
            return [(10, 10, 40, 40), (60, 60, 80, 80)]

    monkeypatch.setattr(app, "face_cascade", MultiFaceCascade())

    call_count = [0]
    def mock_model(tensor_input, training=False):
        call_count[0] += 1
        p = np.zeros(7, dtype=np.float32)
        if call_count[0] % 2 == 1:
            p[3] = 0.90  # happy for primary (80x80)
        else:
            p[0] = 0.85  # angry for secondary (40x40)
        return tf.constant([p])

    monkeypatch.setattr(app, "model", mock_model)

    prev_smoothed = np.zeros(7, dtype=np.float32)
    prev_smoothed[3] = 0.50  # initial happy smoothing
    annotated, confs, new_smoothed, face_infos = app.annotate_frame(fake_frame, smoothed_preds=prev_smoothed, alpha=0.5)

    # Primary face was 80x80, smoothed with alpha=0.5: 0.5 * 0.90 + 0.5 * 0.50 = 0.70
    assert abs(new_smoothed[3] - 0.70) < 1e-4
    # Secondary face (angry) did NOT cross-contaminate new_smoothed
    assert new_smoothed[0] == 0.0


def test_generate_photo_strip_returns_pil_image():
    """Verifies generate_photo_strip returns a PIL Image in-memory without disk leaks."""
    state = app.init_booth_state()
    state["Happy"]["crop"] = np.full((100, 100, 3), 200, dtype=np.uint8)
    state["Happy"]["score"] = 0.95
    strip = app.generate_photo_strip(state)
    assert isinstance(strip, Image.Image)
    assert strip.width > 0 and strip.height > 0


def test_local_dataset_stats_counts_physical_files(tmp_path, monkeypatch):
    """Verifies get_local_dataset_stats prioritizes physical files for parity with fine_tune.py."""
    fake_contrib_dir = str(tmp_path / "user_contributed")
    os.makedirs(os.path.join(fake_contrib_dir, "happy"), exist_ok=True)
    os.makedirs(os.path.join(fake_contrib_dir, "surprise"), exist_ok=True)

    # Save 3 dummy images in happy, 2 in surprise
    for i in range(3):
        Image.new("RGB", (48, 48)).save(os.path.join(fake_contrib_dir, "happy", f"h_{i}.png"))
    for i in range(2):
        Image.new("RGB", (48, 48)).save(os.path.join(fake_contrib_dir, "surprise", f"s_{i}.png"))

    monkeypatch.setattr(app, "USER_CONTRIB_DIR", fake_contrib_dir)
    total, per_class, banner = app.get_local_dataset_stats()

    assert total == 5
    assert per_class["Happy"] == 3
    assert per_class["Surprise"] == 2
    assert "5 Verified Faces Saved Locally" in banner


def test_explain_samples_fallback():
    """Verifies explain.py executes successfully using assets/samples/."""
    # Ensure assets/samples contains sample images
    assert os.path.exists("assets/samples")
    assert len(os.listdir("assets/samples")) >= 7
    # Verify main runs without error
    explain.main()
    assert os.path.exists(os.path.join("outputs", "gradcam_samples", "gradcam_gallery.png"))
