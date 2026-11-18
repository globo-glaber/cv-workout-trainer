import cv2
from config import ExerciseConfig


def resize_maintaining_aspect_ratio(image, target_height: int = 720):
    """Rescales the image without pixel distortion by preserving aspect ratio."""
    h, w = image.shape[:2]
    aspect_ratio = w / h
    target_width = int(target_height * aspect_ratio)
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)


def _format_angle(angle: float | None) -> str:
    """Formats an angle value safely, ensuring 0.0 is never treated as missing."""
    return f"{int(round(angle))} deg" if angle is not None else "-- deg"


def _draw_counter_block(
    image,
    title: str,
    counter,
    y_start: int,
    panel_w: int,
    form_hint: str,
    large: bool = False,
) -> None:
    angle_text = _format_angle(counter.smoothed_angle)
    stage_text = counter.stage or "ready"

    cv2.putText(image, title, (15, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(
        image,
        str(counter.counter),
        (15, y_start + (40 if large else 35)),
        cv2.FONT_HERSHEY_DUPLEX,
        1.2 if large else 1.0,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        image,
        f"Angle: {angle_text}",
        (15, y_start + (68 if large else 58)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        image,
        f"Stage: {stage_text}",
        (15, y_start + (88 if large else 76)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 220, 255),
        1,
    )

    bar_y = y_start + (102 if large else 88)
    bar_h = 14 if large else 10
    prog = counter.get_progress_percent()
    cv2.rectangle(image, (15, bar_y), (panel_w - 15, bar_y + bar_h), (100, 100, 100), 1)
    cv2.rectangle(
        image,
        (15, bar_y),
        (15 + int((panel_w - 30) * (prog / 100)), bar_y + bar_h),
        (0, 255, 0),
        -1,
    )

    if not counter.is_spatially_valid and form_hint:
        cv2.putText(
            image,
            f"FORM: {form_hint}",
            (15, bar_y + bar_h + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 140, 255),
            1,
        )


def draw_modern_ui(image, counter_left, counter_right, config: ExerciseConfig) -> None:
    """Renders a compact HUD showing reps, pixel-space joint angles, stages, and form alerts."""
    h, w = image.shape[:2]
    panel_w = min(260, int(w * 0.36))

    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

    cv2.putText(image, config.name, (15, 30), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(image, "SPACE: Pause | Q: Quit", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)

    if config.is_bilateral:
        _draw_counter_block(
            image, "TOTAL REPS", counter_left, 105, panel_w, config.form_hint, large=True
        )
    else:
        _draw_counter_block(
            image, "LEFT REPS", counter_left, 98, panel_w, config.form_hint, large=False
        )
        _draw_counter_block(
            image, "RIGHT REPS", counter_right, 245, panel_w, config.form_hint, large=False
        )