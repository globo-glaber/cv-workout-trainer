import cv2
import argparse
from detector import PoseDetector
from config import EXERCISES
from counter import ExerciseCounter
from ui import draw_modern_ui, resize_maintaining_aspect_ratio

def main() -> None:
    parser = argparse.ArgumentParser(description="CV Workout Tracker")
    parser.add_argument('--source', type=str, default='0', help="0 for webcam, or path to .mp4")
    parser.add_argument('--exercise', type=str, choices=['bicep', 'squat', 'pushup', 'shoulder_press'], default='bicep')
    args = parser.parse_args()

    detector = PoseDetector()
    config = EXERCISES[args.exercise]
    
    # Initialize trackers
    counter_left = ExerciseCounter(config)
    counter_right = ExerciseCounter(config)
    
    video_source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"Error: Cannot open video source {video_source}")
        return

    paused = False
    frame = None

    while cap.isOpened():
        if not paused:
            ret, current_frame = cap.read()
            if not ret:
                print("Video stream ended.")
                break
            frame = current_frame.copy()

        if not paused and frame is not None:
            processed_frame, results = detector.process_frame(frame)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks[0]
                
                # Fetch points for both sides
                jl1, jl2, jl3 = config.joints_left
                l_p1, l_p2, l_p3 = landmarks[jl1], landmarks[jl2], landmarks[jl3]
                
                jr1, jr2, jr3 = config.joints_right
                r_p1, r_p2, r_p3 = landmarks[jr1], landmarks[jr2], landmarks[jr3]
                
                vis_l = (l_p1.visibility + l_p2.visibility + l_p3.visibility) / 3
                vis_r = (r_p1.visibility + r_p2.visibility + r_p3.visibility) / 3

                if config.is_bilateral:
                    # Logic for unified movements (e.g., Squats, Push-ups)
                    spatial_valid = True
                    if vis_l > vis_r:
                        if config.require_wrist_above_shoulder:
                            spatial_valid = l_p3.y < l_p1.y
                        angle = counter_left.calculate_angle([l_p1.x, l_p1.y], [l_p2.x, l_p2.y], [l_p3.x, l_p3.y])
                    else:
                        if config.require_wrist_above_shoulder:
                            spatial_valid = r_p3.y < r_p1.y
                        angle = counter_left.calculate_angle([r_p1.x, r_p1.y], [r_p2.x, r_p2.y], [r_p3.x, r_p3.y])
                        
                    counter_left.update(angle, spatial_valid)
                    
                else:
                    # Logic for independent arm tracking (e.g., Bicep Curls, Shoulder Press)
                    spatial_valid_l = (l_p3.y < l_p1.y) if config.require_wrist_above_shoulder else True
                    if vis_l > 0.5:
                        angle_l = counter_left.calculate_angle([l_p1.x, l_p1.y], [l_p2.x, l_p2.y], [l_p3.x, l_p3.y])
                        counter_left.update(angle_l, spatial_valid_l)
                    
                    spatial_valid_r = (r_p3.y < r_p1.y) if config.require_wrist_above_shoulder else True
                    if vis_r > 0.5:
                        angle_r = counter_right.calculate_angle([r_p1.x, r_p1.y], [r_p2.x, r_p2.y], [r_p3.x, r_p3.y])
                        counter_right.update(angle_r, spatial_valid_r)

                detector.draw_landmarks(processed_frame, results)

            draw_modern_ui(processed_frame, counter_left, counter_right, config)
            display_frame = resize_maintaining_aspect_ratio(processed_frame, target_height=720)
            cv2.imshow('CV Workout Tracker', display_frame)

        # Keyboard controls (30ms delay)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):  
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()