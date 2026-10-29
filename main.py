import cv2
import mediapipe as mp
import numpy as np
import time

class PoseDetector:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def process_frame(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results

    def draw_landmarks(self, image, results):
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image, 
                results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )
        return image

class ExerciseCounter:
    def __init__(self, cooldown=1.0):
        self.counter_l = 0
        self.stage_l = None
        self.counter_r = 0
        self.stage_r = None
        self.cooldown = cooldown
        self.last_count_time_l = 0
        self.last_count_time_r = 0

    def calculate_angle(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360.0 - angle
            
        return angle

    def update_left(self, angle):
        current_time = time.time()
        
        if angle > 160:
            self.stage_l = "down"
        if angle < 50 and self.stage_l == "down":
            if current_time - self.last_count_time_l > self.cooldown:
                self.stage_l = "up"
                self.counter_l += 1
                self.last_count_time_l = current_time

    def update_right(self, angle):
        current_time = time.time()
        
        if angle > 160:
            self.stage_r = "down"
        if angle < 50 and self.stage_r == "down":
            if current_time - self.last_count_time_r > self.cooldown:
                self.stage_r = "up"
                self.counter_r += 1
                self.last_count_time_r = current_time

def main():
    detector = PoseDetector()
    counter = ExerciseCounter(cooldown=1.0)
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image, results = detector.process_frame(frame)

        try:
            landmarks = results.pose_landmarks.landmark
            mp_pose = detector.mp_pose
            
            shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow_l = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist_l = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            angle_l = counter.calculate_angle(shoulder_l, elbow_l, wrist_l)
            counter.update_left(angle_l)
            
            cv2.putText(image, str(int(angle_l)), tuple(np.multiply(elbow_l, [640, 480]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist_r = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            
            angle_r = counter.calculate_angle(shoulder_r, elbow_r, wrist_r)
            counter.update_right(angle_r)
            
            cv2.putText(image, str(int(angle_r)), tuple(np.multiply(elbow_r, [640, 480]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        except:
            pass

        cv2.rectangle(image, (0,0), (250,73), (245,117,16), -1)
        
        cv2.putText(image, 'L REPS', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter.counter_l), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        cv2.putText(image, 'R REPS', (130,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter.counter_r), (125,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

        image = detector.draw_landmarks(image, results)
        
        cv2.imshow('CV Tracker', image)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
