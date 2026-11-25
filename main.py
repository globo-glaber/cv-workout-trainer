import argparse
import time

import cv2

from config import EXERCISES
from counter import ExerciseCounter, calculate_pixel_angle, evaluate_form
from detector import PoseDetector
from ui import draw_modern_ui, resize_maintaining_aspect_ratio

WINDOW_NAME = "CV Workout Tracker"
QUIT_KEYS = {ord("q"), ord("Q"), 27, 233, 201}
SIDE_SWITCH_HYSTERESIS = 0.15
BILATERAL_MIN_VISIBILITY = 0.60
UNILATERAL_MIN_VISIBILITY = 0.55


def _to_tuple(landmark) -> tuple[float, float, float]:
    return (landmark.x, landmark.y, landmark.visibility)


def main() -> None:
    parser = argparse.ArgumentParser(description="CV Workout Tracker")
    parser.add_argument("--source", type=str, default="0", help="0 for webcam, or path to .mp4")
    parser.add_argument(
        "--exercise",
        type=str,
        choices=list(EXERCISES.keys()),
        default="bicep",
    )
    args = parser.parse_args()

    config = EXERCISES[args.exercise]
    counter_left = ExerciseCounter(config)
    counter_right = ExerciseCounter(config)

    video_source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(video_source)
    is_live_camera = isinstance(video_source, int)

    if not cap.isOpened():
        print(f"Error: Cannot open video source {video_source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_duration_ms = 1000.0 / fps if fps and fps > 0 else 33.33

    paused = False
    frame = None
    active_side: str | None = None
    base_time_ms = time.monotonic() * 1000.0
    last_ts_ms = -1
    frame_idx = 0

    try:
        with PoseDetector() as detector:
            while cap.isOpened():
                if not paused:
                    ret, current_frame = cap.read()
                    if not ret:
                        print("Video stream ended.")
                        break

                    frame = current_frame.copy()

                if not paused and frame is not None:
                    if is_live_camera:
                        ts_ms = int((time.monotonic() * 1000.0) - base_time_ms)
                    else:
                        pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                        ts_ms = int(pos_ms) if pos_ms > 0 else int(frame_idx * frame_duration_ms)
                        frame_idx += 1

                    if ts_ms <= last_ts_ms:
                        ts_ms = last_ts_ms + 1
                    last_ts_ms = ts_ms

                    processed_frame, results = detector.process_frame(frame, ts_ms)

                    if results.pose_landmarks:
                        landmarks = results.pose_landmarks[0]
                        h, w = processed_frame.shape[:2]

                        jl1, jl2, jl3 = config.joints_left
                        jr1, jr2, jr3 = config.joints_right

                        l1, l2, l3 = landmarks[jl1], landmarks[jl2], landmarks[jl3]
                        r1, r2, r3 = landmarks[jr1], landmarks[jr2], landmarks[jr3]

                        vis_l = (l1.visibility + l2.visibility + l3.visibility) / 3.0
                        vis_r = (r1.visibility + r2.visibility + r3.visibility) / 3.0

                        form_l = evaluate_form(
                            config.form_rule,
                            _to_tuple(landmarks[11]),
                            _to_tuple(landmarks[13]),
                            _to_tuple(landmarks[15]),
                            _to_tuple(landmarks[23]),
                            w,
                            h,
                        )
                        form_r = evaluate_form(
                            config.form_rule,
                            _to_tuple(landmarks[12]),
                            _to_tuple(landmarks[14]),
                            _to_tuple(landmarks[16]),
                            _to_tuple(landmarks[24]),
                            w,
                            h,
                        )

                        if config.is_bilateral:
                            if active_side is None:
                                active_side = "left" if vis_l >= vis_r else "right"
                            else:
                                new_side = active_side
                                if active_side == "left" and vis_r > vis_l + SIDE_SWITCH_HYSTERESIS:
                                    new_side = "right"
                                elif active_side == "right" and vis_l > vis_r + SIDE_SWITCH_HYSTERESIS:
                                    new_side = "left"

                                if new_side != active_side:
                                    active_side = new_side
                                    counter_left.reset_smoothing()

                            active_vis = vis_l if active_side == "left" else vis_r
                            if active_vis >= BILATERAL_MIN_VISIBILITY:
                                if active_side == "left":
                                    angle = calculate_pixel_angle((l1.x, l1.y), (l2.x, l2.y), (l3.x, l3.y), w, h)
                                    valid = form_l
                                else:
                                    angle = calculate_pixel_angle((r1.x, r1.y), (r2.x, r2.y), (r3.x, r3.y), w, h)
                                    valid = form_r
                                counter_left.update(angle, float(ts_ms), valid)
                        else:
                            if vis_l >= UNILATERAL_MIN_VISIBILITY:
                                angle_l = calculate_pixel_angle((l1.x, l1.y), (l2.x, l2.y), (l3.x, l3.y), w, h)
                                counter_left.update(angle_l, float(ts_ms), form_l)

                            if vis_r >= UNILATERAL_MIN_VISIBILITY:
                                angle_r = calculate_pixel_angle((r1.x, r1.y), (r2.x, r2.y), (r3.x, r3.y), w, h)
                                counter_right.update(angle_r, float(ts_ms), form_r)

                        detector.draw_landmarks(processed_frame, results)

                    if is_live_camera:
                        processed_frame = cv2.flip(processed_frame, 1)

                    draw_modern_ui(processed_frame, counter_left, counter_right, config)
                    display_frame = resize_maintaining_aspect_ratio(processed_frame, target_height=720)
                    cv2.imshow(WINDOW_NAME, display_frame)

                raw_key = cv2.waitKey(1)
                if raw_key != -1:
                    key = raw_key & 0xFF
                    if key in QUIT_KEYS or raw_key in (0x0439, 0x0419):
                        break
                    elif key == ord(" "):
                        paused = not paused

                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()