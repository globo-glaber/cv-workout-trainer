import os
import shutil
import urllib.request
from typing import Any, Tuple

import cv2
import mediapipe as mp
import numpy as np

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/1/pose_landmarker_full.task"
)

# MediaPipe 33-point BlazePose skeletal topology (Tasks API only, no legacy mp.solutions)
POSE_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
)


def download_model_if_missing(
    model_path: str,
    url: str = MODEL_URL,
    timeout: float = 30.0,
) -> None:
    """Downloads the model atomically and verifies byte completeness before replacing the target."""
    if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
        return

    temp_path = model_path + ".tmp"
    print(f"Downloading model {model_path}...")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            expected_len_header = response.headers.get("Content-Length")
            expected_len = int(expected_len_header) if expected_len_header is not None else None

            with open(temp_path, "wb") as out_file:
                shutil.copyfileobj(response, out_file)

        actual_len = os.path.getsize(temp_path)
        if actual_len == 0:
            raise RuntimeError("Downloaded model file is empty.")
        if expected_len is not None and actual_len != expected_len:
            raise RuntimeError(
                f"Incomplete model download: expected {expected_len} bytes, got {actual_len}."
            )

        os.replace(temp_path, model_path)
        print("Download complete.")
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise


class PoseDetector:
    """Handles MediaPipe Tasks PoseLandmarker in VIDEO mode with pure OpenCV rendering."""

    def __init__(self, model_asset_path: str = "pose_landmarker_full.task") -> None:
        self.model_path = model_asset_path
        download_model_if_missing(self.model_path)

        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self.model_path),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.6,
            min_pose_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.detector = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def process_frame(self, frame: np.ndarray, timestamp_ms: int) -> Tuple[np.ndarray, Any]:
        """Converts frame to RGB and runs temporal pose tracking."""
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        results = self.detector.detect_for_video(mp_image, timestamp_ms)
        return frame, results

    def draw_landmarks(
        self,
        image: np.ndarray,
        results: Any,
        min_visibility: float = 0.5,
    ) -> np.ndarray:
        """Draws visible skeletal connections and joints using pure OpenCV."""
        if not results.pose_landmarks:
            return image

        h, w = image.shape[:2]
        for pose_landmarks in results.pose_landmarks:
            points = [
                (int(lm.x * w), int(lm.y * h), lm.visibility)
                for lm in pose_landmarks
            ]

            for start_idx, end_idx in POSE_CONNECTIONS:
                x1, y1, v1 = points[start_idx]
                x2, y2, v2 = points[end_idx]
                if v1 >= min_visibility and v2 >= min_visibility:
                    cv2.line(image, (x1, y1), (x2, y2), (245, 66, 230), 2, cv2.LINE_AA)

            for x, y, vis in points:
                if vis >= min_visibility:
                    cv2.circle(image, (x, y), 4, (245, 117, 66), -1, cv2.LINE_AA)
                    cv2.circle(image, (x, y), 4, (255, 255, 255), 1, cv2.LINE_AA)

        return image

    def close(self) -> None:
        self.detector.close()

    def __enter__(self) -> "PoseDetector":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()