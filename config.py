from typing import Tuple, Dict

class ExerciseConfig:
    """Stores joint indices and angle thresholds for dynamic exercise tracking."""
    def __init__(self, joints_left: Tuple[int, int, int], joints_right: Tuple[int, int, int], 
                 angle_extended: float, angle_flexed: float, name: str, is_bilateral: bool, 
                 movement_type: str, require_wrist_above_shoulder: bool = False):
        self.joints_left = joints_left
        self.joints_right = joints_right
        self.angle_extended = angle_extended
        self.angle_flexed = angle_flexed
        self.name = name
        self.is_bilateral = is_bilateral
        self.movement_type = movement_type
        self.require_wrist_above_shoulder = require_wrist_above_shoulder

# Registry of supported exercises
EXERCISES: Dict[str, ExerciseConfig] = {
    "bicep": ExerciseConfig(
        joints_left=(11, 13, 15),   # Shoulder, Elbow, Wrist
        joints_right=(12, 14, 16), 
        angle_extended=160.0, 
        angle_flexed=50.0, 
        name="BICEP CURL", 
        is_bilateral=False,
        movement_type="flexion"
    ),
    "squat": ExerciseConfig(
        joints_left=(23, 25, 27),   # Hip, Knee, Ankle
        joints_right=(24, 26, 28), 
        angle_extended=160.0, 
        angle_flexed=90.0, 
        name="BARBELL SQUAT", 
        is_bilateral=True,
        movement_type="flexion"
    ),
    "pushup": ExerciseConfig(
        joints_left=(11, 13, 15),   # Shoulder, Elbow, Wrist
        joints_right=(12, 14, 16), 
        angle_extended=160.0, 
        angle_flexed=80.0,          # 80 degrees for a deep push-up
        name="PUSH-UP", 
        is_bilateral=True,          # Tracked as one unified movement
        movement_type="flexion"
    ),
    "shoulder_press": ExerciseConfig(
        joints_left=(11, 13, 15),   # Shoulder, Elbow, Wrist
        joints_right=(12, 14, 16), 
        angle_extended=160.0, 
        angle_flexed=70.0,          # Dumbbells lowered to ear/shoulder level
        name="SHOULDER PRESS", 
        is_bilateral=False,         # Left and right arms move independently
        movement_type="extension",  # Effort phase is pushing up
        require_wrist_above_shoulder=True  # Enforce spatial constraints
    )
}