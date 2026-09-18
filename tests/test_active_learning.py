"""
Automated Test Suite for EdgeVision Active Learning, Photo Booth, & Guardrails
"""

import os
import json
import shutil
import tempfile
import numpy as np
import pytest
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import app
import fine_tune


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
    """Verifies predict_emotion alias exists for existing CI smoke tests."""
    assert hasattr(app, "predict_emotion")
    dummy_img = np.zeros((120, 120, 3), dtype=np.uint8)
    annotated, confs = app.predict_emotion(dummy_img)
    assert len(confs) == 7 or len(confs) == 0


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

    # Base samples have class 0 activated
    base_samples_in_batch = np.sum(batch_y[:, 0] == 1.0)
    assert base_samples_in_batch == 12, f"Expected 12 base samples in batch, found {base_samples_in_batch}"


def test_original_model_preservation():
    """Verifies that an immutable original model baseline exists on disk."""
    fine_tune.ensure_original_baseline()
    assert os.path.exists(fine_tune.ORIGINAL_MODEL_PATH)
    assert os.path.getsize(fine_tune.ORIGINAL_MODEL_PATH) > 800000  # ~817 KB
