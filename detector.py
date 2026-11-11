import cv2
import mediapipe as mp
import numpy as np
import os
import urllib.request
from typing import Tuple, Any
from mediapipe.framework.formats import landmark_pb2

class PoseDetector:
    """Handles MediaPipe PoseLandmarker initialization and frame processing."""
    def __init__(self, model_asset_path: str = 'pose_landmarker_lite.task'):
        self.model_path = model_asset_path
        self._ensure_model_exists()
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self.model_path),
            running_mode=mp.tasks.vision.RunningMode.IMAGE
        )
        self.detector = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def _ensure_model_exists(self) -> None:
        """Downloads the required AI model if it is missing."""
        if not os.path.exists(self.model_path):
            print(f"Downloading model {self.model_path}...")
            url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
            urllib.request.urlretrieve(url, self.model_path)
            print("Download complete.")

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Any]:
        """Converts frame to RGB and extracts pose landmarks."""
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        results = self.detector.detect(mp_image)
        return frame, results

    def draw_landmarks(self, image: np.ndarray, results: Any) -> np.ndarray:
        """Renders the skeletal connections on the provided image."""
        if results.pose_landmarks:
            for pose_landmarks in results.pose_landmarks:
                pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                pose_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=l.x, y=l.y, z=l.z) for l in pose_landmarks
                ])
                self.mp_drawing.draw_landmarks(
                    image,
                    pose_landmarks_proto,
                    self.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )
        return image