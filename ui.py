import cv2
from config import ExerciseConfig

def resize_maintaining_aspect_ratio(image, target_height=720):
    """Rescales the image preventing pixel distortion by maintaining aspect ratio."""
    h, w = image.shape[:2]
    aspect_ratio = w / h
    target_width = int(target_height * aspect_ratio)
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)

def draw_modern_ui(image, counter_left, counter_right, config: ExerciseConfig):
    """Renders a dynamic, compact HUD that adjusts to video width and exercise type."""
    h, w, _ = image.shape
    
    # Dynamic panel width: 35% of video width, but max 250px so it doesn't cover the screen
    panel_w = min(250, int(w * 0.35))
    
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

    # Header
    cv2.putText(image, config.name, (15, 30), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(image, "SPACE: Pause", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    if config.is_bilateral:
        # UNIFIED VIEW (For Squats) - Shows only one main counter
        reps = counter_left.counter 
        angle_text = f"{int(counter_left.smoothed_angle)} deg" if counter_left.smoothed_angle else "-- deg"
        
        cv2.putText(image, "TOTAL REPS", (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(image, str(reps), (15, 150), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 2)
        cv2.putText(image, f"Angle: {angle_text}", (15, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        prog = counter_left.get_progress_percent()
        cv2.rectangle(image, (15, 200), (panel_w - 15, 215), (100, 100, 100), 1)
        cv2.rectangle(image, (15, 200), (15 + int((panel_w - 30) * (prog / 100)), 215), (0, 255, 0), -1)
    
    else:
        # SPLIT VIEW (For Bicep Curls) - Shows Left and Right independently
        # Left Side
        cv2.putText(image, "LEFT REPS", (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(image, str(counter_left.counter), (15, 145), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2)
        
        prog_l = counter_left.get_progress_percent()
        cv2.rectangle(image, (15, 160), (panel_w - 15, 170), (100, 100, 100), 1)
        cv2.rectangle(image, (15, 160), (15 + int((panel_w - 30) * (prog_l / 100)), 170), (0, 255, 0), -1)

        # Right Side
        cv2.putText(image, "RIGHT REPS", (15, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(image, str(counter_right.counter), (15, 255), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2)
        
        prog_r = counter_right.get_progress_percent()
        cv2.rectangle(image, (15, 270), (panel_w - 15, 280), (100, 100, 100), 1)
        cv2.rectangle(image, (15, 270), (15 + int((panel_w - 30) * (prog_r / 100)), 280), (0, 255, 0), -1)