import numpy as np
import time
from typing import List
from config import ExerciseConfig

class ExerciseCounter:
    """Tracks joint angles and implements finite state machine logic for counting reps."""
    def __init__(self, config: ExerciseConfig, cooldown: float = 1.0):
        self.config = config
        self.counter: int = 0
        self.stage: str | None = None
        self.cooldown: float = cooldown
        self.last_count_time: float = 0.0
        
        # Exponential Moving Average (EMA) filter variables
        self.smoothed_angle: float | None = None
        self.smoothing_factor: float = 0.2
        self.is_spatially_valid: bool = True

    def calculate_angle(self, a: List[float], b: List[float], c: List[float]) -> float:
        """Calculates the inner angle between three points in degrees."""
        a_np, b_np, c_np = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c_np[1] - b_np[1], c_np[0] - b_np[0]) - np.arctan2(a_np[1] - b_np[1], a_np[0] - b_np[0])
        angle = float(np.abs(radians * 180.0 / np.pi))
        return 360.0 - angle if angle > 180.0 else angle

    def update(self, raw_angle: float, spatial_valid: bool = True) -> None:
        """Applies EMA smoothing and updates the rep counter state machine."""
        self.is_spatially_valid = spatial_valid
        
        if self.smoothed_angle is None:
            self.smoothed_angle = raw_angle
        else:
            self.smoothed_angle = (self.smoothing_factor * raw_angle) + ((1 - self.smoothing_factor) * self.smoothed_angle)

        current_time = time.time()
        
        # Logic for pulling/flexing movements (e.g., Bicep Curl, Squat)
        if self.config.movement_type == "flexion":
            if self.smoothed_angle > self.config.angle_extended and spatial_valid:
                self.stage = "extended"
                
            if self.smoothed_angle < self.config.angle_flexed and self.stage == "extended":
                if current_time - self.last_count_time >= self.cooldown:
                    self.stage = "flexed"
                    self.counter += 1
                    self.last_count_time = current_time

        # Logic for pushing/extending movements (e.g., Shoulder Press)
        elif self.config.movement_type == "extension":
            if self.smoothed_angle < self.config.angle_flexed:
                self.stage = "flexed"
                
            if self.smoothed_angle > self.config.angle_extended and self.stage == "flexed":
                if spatial_valid and (current_time - self.last_count_time >= self.cooldown):
                    self.stage = "extended"
                    self.counter += 1
                    self.last_count_time = current_time

    def get_progress_percent(self) -> int:
        """Calculates the exercise execution progress as a percentage (0-100%)."""
        if self.smoothed_angle is None: 
            return 0
            
        # Reset progress bar if spatial requirements (like wrist height) are violated
        if not self.is_spatially_valid:
            return 0
        
        full_range = self.config.angle_extended - self.config.angle_flexed
        if full_range == 0: 
            return 0
        
        if self.config.movement_type == "flexion":
            progress = (self.config.angle_extended - self.smoothed_angle) / full_range
        else:
            progress = (self.smoothed_angle - self.config.angle_flexed) / full_range
            
        return int(max(0.0, min(1.0, progress)) * 100)