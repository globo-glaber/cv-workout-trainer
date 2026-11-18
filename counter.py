from typing import Tuple

import numpy as np

from config import ExerciseConfig, FormRule

# Normalized landmark tuple: (x, y, visibility)
Landmark2D = Tuple[float, float, float]


def calculate_angle(
    a: Tuple[float, float],
    b: Tuple[float, float],
    c: Tuple[float, float],
) -> float:
    """Calculates the inner angle (in degrees) between three 2D pixel-space coordinates."""
    a_np = np.array(a, dtype=float)
    b_np = np.array(b, dtype=float)
    c_np = np.array(c, dtype=float)

    ba = a_np - b_np
    bc = c_np - b_np

    norm_ba = float(np.linalg.norm(ba))
    norm_bc = float(np.linalg.norm(bc))
    if norm_ba == 0.0 or norm_bc == 0.0:
        return 0.0

    cosine_angle = float(np.clip(np.dot(ba, bc) / (norm_ba * norm_bc), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine_angle)))


def calculate_pixel_angle(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    frame_width: int,
    frame_height: int,
) -> float:
    """Converts normalized (0..1) landmarks to true pixel space before computing the angle."""
    return calculate_angle(
        (p1[0] * frame_width, p1[1] * frame_height),
        (p2[0] * frame_width, p2[1] * frame_height),
        (p3[0] * frame_width, p3[1] * frame_height),
    )


def evaluate_form(
    form_rule: FormRule,
    shoulder: Landmark2D,
    elbow: Landmark2D,
    wrist: Landmark2D,
    hip: Landmark2D,
    frame_width: int,
    frame_height: int,
) -> bool:
    """Enforces aspect-ratio-safe posture constraints for each exercise."""
    if form_rule == "none":
        return True

    if form_rule == "wrist_above_shoulder":
        return wrist[1] < shoulder[1]

    dx_torso = abs(shoulder[0] - hip[0]) * frame_width
    dy_torso = abs(shoulder[1] - hip[1]) * frame_height
    torso_visible = shoulder[2] >= 0.4 and hip[2] >= 0.4

    if form_rule == "horizontal_body":
        if not torso_visible:
            return False
        angle_from_horizontal = float(np.degrees(np.arctan2(dy_torso, dx_torso)))
        return angle_from_horizontal < 45.0

    if form_rule == "upright_body":
        if not torso_visible:
            return True
        angle_from_vertical = float(np.degrees(np.arctan2(dx_torso, dy_torso)))
        return angle_from_vertical < 65.0

    if form_rule == "bicep_curl_posture":
        elbow_below_shoulder = elbow[1] > shoulder[1]
        if not elbow_below_shoulder:
            return False
        if torso_visible:
            angle_from_vertical = float(np.degrees(np.arctan2(dx_torso, dy_torso)))
            return angle_from_vertical < 45.0
        return True

    return True


class ExerciseCounter:
    """Tracks joint angles with FPS-independent time-based EMA and FSM rep counting."""

    def __init__(
        self,
        config: ExerciseConfig,
        cooldown_ms: float = 400.0,
        smoothing_tau_ms: float = 35.0,
    ) -> None:
        self.config = config
        self.counter: int = 0
        self.stage: str | None = None
        self.cooldown_ms: float = cooldown_ms
        self.smoothing_tau_ms: float = smoothing_tau_ms

        self.last_count_time_ms: float = -float("inf")
        self.last_timestamp_ms: float | None = None
        self.smoothed_angle: float | None = None
        self.is_spatially_valid: bool = True

    def reset_smoothing(self) -> None:
        """Clears EMA history when switching tracked limbs to prevent cross-limb mixing."""
        self.smoothed_angle = None
        self.last_timestamp_ms = None

    def update(
        self,
        raw_angle: float,
        timestamp_ms: float,
        spatial_valid: bool = True,
    ) -> None:
        """Applies time-constant EMA smoothing and updates the rep counter state machine."""
        self.is_spatially_valid = spatial_valid

        if (
            self.smoothed_angle is None
            or self.last_timestamp_ms is None
            or self.smoothing_tau_ms <= 0.0
        ):
            self.smoothed_angle = raw_angle
        else:
            dt_ms = max(0.0, timestamp_ms - self.last_timestamp_ms)
            alpha = 1.0 - float(np.exp(-dt_ms / self.smoothing_tau_ms))
            self.smoothed_angle = (alpha * raw_angle) + ((1.0 - alpha) * self.smoothed_angle)

        self.last_timestamp_ms = timestamp_ms

        if self.config.movement_type == "flexion":
            if self.smoothed_angle > self.config.angle_extended and spatial_valid:
                self.stage = "extended"

            if (
                self.smoothed_angle < self.config.angle_flexed
                and self.stage == "extended"
                and spatial_valid
            ):
                if timestamp_ms - self.last_count_time_ms >= self.cooldown_ms:
                    self.stage = "flexed"
                    self.counter += 1
                    self.last_count_time_ms = timestamp_ms

        elif self.config.movement_type == "extension":
            if self.smoothed_angle < self.config.angle_flexed:
                self.stage = "flexed"

            if (
                self.smoothed_angle > self.config.angle_extended
                and self.stage == "flexed"
                and spatial_valid
            ):
                if timestamp_ms - self.last_count_time_ms >= self.cooldown_ms:
                    self.stage = "extended"
                    self.counter += 1
                    self.last_count_time_ms = timestamp_ms

    def get_progress_percent(self) -> int:
        """Returns normalized rep progress (0-100%), zeroed when form is invalid."""
        if self.smoothed_angle is None or not self.is_spatially_valid:
            return 0

        full_range = self.config.angle_extended - self.config.angle_flexed
        if full_range == 0:
            return 0

        if self.config.movement_type == "flexion":
            progress = (self.config.angle_extended - self.smoothed_angle) / full_range
        else:
            progress = (self.smoothed_angle - self.config.angle_flexed) / full_range

        return int(max(0.0, min(1.0, progress)) * 100)