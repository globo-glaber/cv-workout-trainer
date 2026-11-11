import pytest
import numpy as np
from counter import ExerciseCounter
from config import ExerciseConfig

# Mock configuration for flexion testing
config_flexion = ExerciseConfig(
    joints_left=(11, 13, 15), joints_right=(12, 14, 16),
    angle_extended=160.0, angle_flexed=50.0,
    name="TEST FLEXION", is_bilateral=False, movement_type="flexion"
)

# Mock configuration for extension testing
config_extension = ExerciseConfig(
    joints_left=(11, 13, 15), joints_right=(12, 14, 16),
    angle_extended=160.0, angle_flexed=70.0,
    name="TEST EXTENSION", is_bilateral=False, movement_type="extension",
    require_wrist_above_shoulder=True
)

def test_calculate_angle_straight():
    """Ensures straight joint calculations return ~180 degrees."""
    counter = ExerciseCounter(config_flexion)
    a, b, c = [0.5, 0.2], [0.5, 0.5], [0.5, 0.8]
    angle = counter.calculate_angle(a, b, c)
    assert np.isclose(angle, 180.0)

def test_flexion_state_machine():
    """Validates finite state machine for flexion-based movements."""
    counter = ExerciseCounter(config_flexion, cooldown=0.0)
    counter.smoothing_factor = 1.0  # Disable EMA for instant step testing
    
    counter.update(170.0)  # Extended
    assert counter.stage == "extended"
    assert counter.counter == 0
    
    counter.update(40.0)   # Flexed
    assert counter.stage == "flexed"
    assert counter.counter == 1
    
    counter.update(170.0)  # Back to extended
    counter.update(40.0)   # Second rep
    assert counter.counter == 2

def test_extension_state_machine_valid_spatial():
    """Validates extension logic with valid spatial positioning."""
    counter = ExerciseCounter(config_extension, cooldown=0.0)
    counter.smoothing_factor = 1.0  # Disable EMA for instant step testing
    
    counter.update(60.0, spatial_valid=True)   # Flexed (Starting position)
    assert counter.stage == "flexed"
    assert counter.counter == 0
    
    counter.update(170.0, spatial_valid=True)  # Extended (Pressing up)
    assert counter.stage == "extended"
    assert counter.counter == 1

def test_extension_state_machine_invalid_spatial():
    """Ensures reps are rejected if spatial validation fails (e.g., arms down)."""
    counter = ExerciseCounter(config_extension, cooldown=0.0)
    counter.smoothing_factor = 1.0  # Disable EMA for instant step testing
    
    counter.update(60.0, spatial_valid=True)   # Starting position
    
    # Extended but spatial constraint failed (arms are straight but pointing down)
    counter.update(170.0, spatial_valid=False)  
    assert counter.stage == "flexed"           # Stage should not update
    assert counter.counter == 0                # Rep should not be counted