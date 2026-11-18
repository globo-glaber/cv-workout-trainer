import io
import urllib.request

import numpy as np
import pytest

from config import EXERCISES
from counter import (
    ExerciseCounter,
    calculate_angle,
    calculate_pixel_angle,
    evaluate_form,
)
from detector import download_model_if_missing
from ui import _format_angle

config_flexion = EXERCISES["bicep"]
config_extension = EXERCISES["shoulder_press"]


def test_calculate_angle_straight():
    assert np.isclose(calculate_angle((0.0, 100.0), (0.0, 0.0), (0.0, -100.0)), 180.0)


def test_calculate_angle_right():
    assert np.isclose(calculate_angle((100.0, 0.0), (0.0, 0.0), (0.0, 100.0)), 90.0)


def test_calculate_angle_tilted_orientation():
    """Angle remains accurate when limb vectors are arbitrarily rotated."""
    a = (10.0, 10.0)
    b = (0.0, 0.0)
    c = (-10.0, 10.0)
    assert np.isclose(calculate_angle(a, b, c), 90.0)


def test_calculate_angle_degenerate_overlap():
    assert calculate_angle((5.0, 5.0), (5.0, 5.0), (10.0, 5.0)) == 0.0


@pytest.mark.parametrize("width,height", [(1920, 1080), (1080, 1920), (1080, 1080)])
@pytest.mark.parametrize("true_angle_deg", [50.0, 90.0, 160.0])
def test_pixel_angle_invariant_to_aspect_ratio(width: int, height: int, true_angle_deg: float):
    """Vertical (9:16), horizontal (16:9), and square frames yield identical true angles."""
    arm_len_px = 200.0
    elbow_px = (width / 2.0, height / 2.0)
    shoulder_px = (elbow_px[0], elbow_px[1] - arm_len_px)

    rad = np.radians(true_angle_deg)
    wrist_px = (
        elbow_px[0] + arm_len_px * np.sin(rad),
        elbow_px[1] - arm_len_px * np.cos(rad),
    )

    p1 = (shoulder_px[0] / width, shoulder_px[1] / height)
    p2 = (elbow_px[0] / width, elbow_px[1] / height)
    p3 = (wrist_px[0] / width, wrist_px[1] / height)

    computed = calculate_pixel_angle(p1, p2, p3, width, height)
    assert np.isclose(computed, true_angle_deg, atol=1e-6)


def test_flexion_state_machine():
    counter = ExerciseCounter(config_flexion, cooldown_ms=0.0, smoothing_tau_ms=0.0)
    counter.update(160.0, timestamp_ms=0.0)
    assert counter.stage == "extended"

    counter.update(55.0, timestamp_ms=500.0)
    assert counter.stage == "flexed"
    assert counter.counter == 1


def test_extension_state_machine_valid_and_invalid_spatial():
    counter = ExerciseCounter(config_extension, cooldown_ms=0.0, smoothing_tau_ms=0.0)
    counter.update(80.0, timestamp_ms=0.0, spatial_valid=True)
    assert counter.stage == "flexed"

    counter.update(165.0, timestamp_ms=400.0, spatial_valid=False)
    assert counter.counter == 0
    assert counter.get_progress_percent() == 0

    counter.update(165.0, timestamp_ms=800.0, spatial_valid=True)
    assert counter.stage == "extended"
    assert counter.counter == 1


def test_time_based_ema_is_fps_independent():
    """Step response over identical elapsed time matches across 30 FPS and 60 FPS."""
    c30 = ExerciseCounter(config_flexion, cooldown_ms=0.0, smoothing_tau_ms=35.0)
    c60 = ExerciseCounter(config_flexion, cooldown_ms=0.0, smoothing_tau_ms=35.0)

    c30.update(160.0, timestamp_ms=0.0)
    c60.update(160.0, timestamp_ms=0.0)

    for step in range(1, 4):
        c30.update(60.0, timestamp_ms=step * (100.0 / 3.0))
    for step in range(1, 7):
        c60.update(60.0, timestamp_ms=step * (100.0 / 6.0))

    assert np.isclose(c30.smoothed_angle, c60.smoothed_angle, atol=1e-6)


@pytest.mark.parametrize("fps", [30, 60])
@pytest.mark.parametrize("rep_duration_s", [1.0, 1.5, 3.0])
def test_fast_reps_not_eaten_by_ema(fps: int, rep_duration_s: float):
    """Simulates 4 bicep reps (165 deg -> 40 deg) across speeds and frame rates."""
    counter = ExerciseCounter(config_flexion)
    dt_ms = 1000.0 / fps
    frames_per_half = max(1, int(round((rep_duration_s / 2.0) * fps)))

    t = 0.0
    counter.update(165.0, timestamp_ms=t)
    for _ in range(4):
        for i in range(1, frames_per_half + 1):
            t += dt_ms
            angle = 165.0 - (125.0 * i / frames_per_half)
            counter.update(angle, timestamp_ms=t)
        for i in range(1, frames_per_half + 1):
            t += dt_ms
            angle = 40.0 + (125.0 * i / frames_per_half)
            counter.update(angle, timestamp_ms=t)

    assert counter.counter == 4


def test_cooldown_allows_first_rep_and_blocks_rapid_duplicate():
    counter = ExerciseCounter(config_flexion, cooldown_ms=400.0, smoothing_tau_ms=0.0)

    counter.update(160.0, timestamp_ms=100.0)
    counter.update(55.0, timestamp_ms=200.0)
    assert counter.counter == 1

    counter.update(160.0, timestamp_ms=300.0)
    counter.update(55.0, timestamp_ms=450.0)
    assert counter.counter == 1

    counter.update(55.0, timestamp_ms=650.0)
    assert counter.counter == 2


def test_reset_smoothing_prevents_cross_limb_mixing():
    counter = ExerciseCounter(EXERCISES["squat"], cooldown_ms=0.0, smoothing_tau_ms=35.0)
    counter.update(170.0, timestamp_ms=0.0)
    counter.reset_smoothing()
    counter.update(90.0, timestamp_ms=33.0)
    assert counter.smoothed_angle == 90.0


def test_form_horizontal_body_blocks_standing_curl_as_pushup():
    standing_valid = evaluate_form(
        "horizontal_body",
        shoulder=(0.5, 0.3, 0.9),
        elbow=(0.5, 0.45, 0.9),
        wrist=(0.5, 0.35, 0.9),
        hip=(0.5, 0.6, 0.9),
        frame_width=1920,
        frame_height=1080,
    )
    assert standing_valid is False

    plank_valid = evaluate_form(
        "horizontal_body",
        shoulder=(0.7, 0.55, 0.9),
        elbow=(0.65, 0.7, 0.9),
        wrist=(0.65, 0.85, 0.9),
        hip=(0.35, 0.52, 0.9),
        frame_width=1920,
        frame_height=1080,
    )
    assert plank_valid is True


def test_form_upright_body_for_squat():
    upright_valid = evaluate_form(
        "upright_body",
        shoulder=(0.5, 0.25, 0.9),
        elbow=(0.5, 0.4, 0.9),
        wrist=(0.5, 0.5, 0.9),
        hip=(0.52, 0.55, 0.9),
        frame_width=1920,
        frame_height=1080,
    )
    assert upright_valid is True

    lying_invalid = evaluate_form(
        "upright_body",
        shoulder=(0.7, 0.5, 0.9),
        elbow=(0.65, 0.6, 0.9),
        wrist=(0.65, 0.7, 0.9),
        hip=(0.3, 0.52, 0.9),
        frame_width=1920,
        frame_height=1080,
    )
    assert lying_invalid is False


def test_form_wrist_above_shoulder_for_press():
    overhead_valid = evaluate_form(
        "wrist_above_shoulder",
        shoulder=(0.5, 0.4, 0.9),
        elbow=(0.5, 0.3, 0.9),
        wrist=(0.5, 0.2, 0.9),
        hip=(0.5, 0.7, 0.9),
        frame_width=1920,
        frame_height=1080,
    )
    assert overhead_valid is True

    lowered_invalid = evaluate_form(
        "wrist_above_shoulder",
        shoulder=(0.5, 0.4, 0.9),
        elbow=(0.5, 0.55, 0.9),
        wrist=(0.5, 0.7, 0.9),
        hip=(0.5, 0.7, 0.9),
        frame_width=1920,
        frame_height=1080,
    )
    assert lowered_invalid is False


def test_form_bicep_posture_blocks_overhead_arms():
    valid = evaluate_form(
        "bicep_curl_posture",
        shoulder=(0.5, 0.3, 0.9),
        elbow=(0.5, 0.45, 0.9),
        wrist=(0.52, 0.35, 0.9),
        hip=(0.5, 0.6, 0.9),
        frame_width=1080,
        frame_height=1920,
    )
    assert valid is True

    invalid = evaluate_form(
        "bicep_curl_posture",
        shoulder=(0.5, 0.4, 0.9),
        elbow=(0.5, 0.25, 0.9),
        wrist=(0.5, 0.15, 0.9),
        hip=(0.5, 0.7, 0.9),
        frame_width=1080,
        frame_height=1920,
    )
    assert invalid is False


def test_format_angle_handles_zero_and_none():
    """Regression test: 0.0 degrees must display as '0 deg', not '-- deg'."""
    assert _format_angle(0.0) == "0 deg"
    assert _format_angle(None) == "-- deg"
    assert _format_angle(89.6) == "90 deg"


def test_download_model_cleans_up_on_interrupted_stream(tmp_path, monkeypatch):
    """Interrupted downloads must never leave corrupt .task or .tmp files behind."""

    class _FailingStream:
        headers = {"Content-Length": "100"}

        def __init__(self) -> None:
            self._sent = False

        def read(self, _size: int = -1) -> bytes:
            if not self._sent:
                self._sent = True
                return b"partial-bytes"
            raise ConnectionError("Network dropped mid-download")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb) -> None:
            return None

    monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: _FailingStream())

    target = tmp_path / "pose_landmarker_full.task"
    with pytest.raises(ConnectionError):
        download_model_if_missing(str(target))

    assert not target.exists()
    assert not (tmp_path / "pose_landmarker_full.task.tmp").exists()


def test_download_model_rejects_truncated_content_length(tmp_path, monkeypatch):
    """Truncated response shorter than Content-Length must fail and clean up."""

    class _TruncatedStream(io.BytesIO):
        headers = {"Content-Length": "500"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb) -> None:
            self.close()

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *args, **kwargs: _TruncatedStream(b"too-short"),
    )

    target = tmp_path / "pose_landmarker_full.task"
    with pytest.raises(RuntimeError, match="Incomplete model download"):
        download_model_if_missing(str(target))

    assert not target.exists()
    assert not (tmp_path / "pose_landmarker_full.task.tmp").exists()


def test_exercise_config_is_immutable():
    with pytest.raises(Exception):
        EXERCISES["bicep"].angle_flexed = 99.0  # type: ignore[misc]