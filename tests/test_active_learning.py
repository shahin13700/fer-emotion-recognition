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
        assert data["is_sample"] is False


def test_use_sample_face_disgust_and_fear():
    """Verifies UX fail-safe loads sample faces without crashing and applies upscaling."""
    state = app.init_booth_state()

    updated_state, status, gallery = app.use_sample_face("Disgust", state)
    assert updated_state["Disgust"]["crop"] is not None
    assert updated_state["Disgust"]["is_sample"] is True
    assert updated_state["Disgust"]["score"] == app.EMPIRICAL_THRESHOLDS["Disgust"]
    h, w, c = updated_state["Disgust"]["crop"].shape
    assert h >= 100 and w >= 100
    assert len(gallery) >= 1
    assert "(Sample)" in gallery[0][1]

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
    state["Happy"] = {"score": 0.85, "crop": np.zeros((100, 100, 3), dtype=np.uint8), "is_sample": False}

    feedback, banner = app.save_and_contribute(state, consent_given=True)
    assert "Successfully saved" in feedback
    assert os.path.exists(fake_jsonl)

    with open(fake_jsonl, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["emotion"] == "happy"
    assert record["score"] == 0.85
    assert record["verified_by_user"] is True
    assert record["timestamp"].endswith("+00:00")
    assert os.path.exists(record["image_path"])


def test_samples_excluded_from_contribution(tmp_path, monkeypatch):
    """Verifies benchmark fallback samples are NEVER re-saved to the training pool."""
    fake_contrib_dir = str(tmp_path / "user_contributed")
    fake_jsonl = os.path.join(fake_contrib_dir, "metadata.jsonl")
    monkeypatch.setattr(app, "USER_CONTRIB_DIR", fake_contrib_dir)
    monkeypatch.setattr(app, "METADATA_JSONL", fake_jsonl)
    monkeypatch.setattr(app, "ALLOW_LOCAL_SAVE", True)

    state = app.init_booth_state()
    state["Disgust"] = {"score": 0.50, "crop": np.zeros((100, 100, 3), dtype=np.uint8), "is_sample": True}

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
    state["Happy"] = {"score": 0.9, "crop": np.zeros((80, 80, 3), dtype=np.uint8), "is_sample": False}
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


def test_user_data_loaded_as_raw_pixels(tmp_path, monkeypatch):
    """User faces must be loaded in the 0-255 range; the generator does the rescaling."""
    contrib = tmp_path / "user_contributed"
    (contrib / "happy").mkdir(parents=True)
    Image.fromarray(np.full((60, 60), 200, dtype=np.uint8)).save(contrib / "happy" / "a.png")
    monkeypatch.setattr(fine_tune, "USER_CONTRIB_DIR", str(contrib))

    x, y = fine_tune.load_and_preprocess_user_data({"angry": 0, "happy": 1})
    assert x.shape == (1, 48, 48, 1) and y.shape == (1, 2)
    assert x.max() > 1.0 and abs(float(x.mean()) - 200.0) < 1.0
    assert y[0, 1] == 1.0


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
    assert ema is not None and abs(float(ema[3]) - 0.95) < 1e-6


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
        {"label": "Happy", "score": 0.9, "box": (60, 60, 80, 80)},
        {"label": "Angry", "score": 0.9, "box": (10, 10, 40, 40)},
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

    # No face -> smoother resets
    monkeypatch.setattr(app, "face_cascade", type("NoFace", (), {"detectMultiScale": lambda self, *a, **k: []})())
    _, confs3, ema3 = app.process_live_frame(frame, ema2)
    assert confs3 == {} and ema3 is None


def test_dynamic_peak_expression_tracking(monkeypatch):
    """Verifies that booth captures at the 25% floor and upgrades when beating a personal best."""
    state = app.init_booth_state()
    frame = np.full((120, 120, 3), 128, dtype=np.uint8)

    def mock(faces):
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
    strip = app.generate_photo_strip(state)
    assert isinstance(strip, Image.Image)
    assert strip.width > 0 and strip.height > 0


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
