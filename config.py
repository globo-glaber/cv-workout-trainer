from dataclasses import dataclass
from typing import Dict, Literal, Tuple

MovementType = Literal["flexion", "extension"]
FormRule = Literal[
    "none",
    "wrist_above_shoulder",
    "horizontal_body",
    "upright_body",
    "bicep_curl_posture",
]


@dataclass(frozen=True)
class ExerciseConfig:
    """Immutable configuration for joint indices, thresholds, and form rules."""

    joints_left: Tuple[int, int, int]
    joints_right: Tuple[int, int, int]
    angle_extended: float
    angle_flexed: float
    name: str
    is_bilateral: bool
    movement_type: MovementType
    form_rule: FormRule = "none"
    form_hint: str = ""


EXERCISES: Dict[str, ExerciseConfig] = {
    "bicep": ExerciseConfig(
        joints_left=(11, 13, 15),
        joints_right=(12, 14, 16),
        angle_extended=150.0,
        angle_flexed=65.0,
        name="BICEP CURL",
        is_bilateral=False,
        movement_type="flexion",
        form_rule="bicep_curl_posture",
        form_hint="Stand upright, elbows down",
    ),
    "squat": ExerciseConfig(
        joints_left=(23, 25, 27),
        joints_right=(24, 26, 28),
        angle_extended=155.0,
        angle_flexed=100.0,
        name="BARBELL SQUAT",
        is_bilateral=True,
        movement_type="flexion",
        form_rule="upright_body",
        form_hint="Keep torso upright",
    ),
    "pushup": ExerciseConfig(
        joints_left=(11, 13, 15),
        joints_right=(12, 14, 16),
        angle_extended=150.0,
        angle_flexed=90.0,
        name="PUSH-UP",
        is_bilateral=True,
        movement_type="flexion",
        form_rule="horizontal_body",
        form_hint="Keep body horizontal",
    ),
    "shoulder_press": ExerciseConfig(
        joints_left=(11, 13, 15),
        joints_right=(12, 14, 16),
        angle_extended=150.0,
        angle_flexed=90.0,
        name="SHOULDER PRESS",
        is_bilateral=False,
        movement_type="extension",
        form_rule="wrist_above_shoulder",
        form_hint="Press wrists above shoulders",
    ),
}