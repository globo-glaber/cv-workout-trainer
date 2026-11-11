# AI Workout Rep Controller

A modular Computer Vision application that tracks workout repetitions, joint angles, and enforces exercise form using MediaPipe and OpenCV. 

## Features
* **Modular Architecture**: Clean separation of concerns (Config, Detector, Counter, UI).
* **Exponential Moving Average (EMA) Filter**: Smooths joint angle calculations to eliminate neural network "jitter".
* **Spatial Constraints (Form Checking)**: Enforces rules like "wrists must be above shoulders" for shoulder presses to prevent false positives.
* **Bilateral & Unilateral Tracking**: Automatically selects the most visible side for unified movements (e.g., Squats) or tracks limbs independently (e.g., Bicep Curls).
* **Dynamic HUD**: Real-time progress bar, rep counter, and angle tracking scaled perfectly for horizontal or vertical (Shorts/TikTok) videos.
* **Pause/Analyze Mode**: Press `SPACE` to pause the video and inspect form/angles on a specific frame.

## Architecture
* `config.py`: Registry of exercises, joint indices, angles, and spatial rules.
* `detector.py`: MediaPipe PoseLandmarker initialization and rendering.
* `counter.py`: Trigonometry, EMA filtering, and Finite State Machine logic.
* `ui.py`: OpenCV rendering for the HUD and aspect-ratio-preserving resizing.
* `main.py`: Application entry point and video processing loop.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/globo-glaber/cv-workout-trainer.git
    cd rep-controller
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv .venv
    
    # On Windows:
    .\.venv\Scripts\activate
    
    # On macOS/Linux:
    source .venv/bin/activate
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: The `pose_landmarker_lite.task` model will be downloaded automatically on the first run.*

## Usage

Run the program via CLI. You can use a webcam (`0`) or provide a path to an `.mp4` file.

```bash
python main.py --source <path_or_0> --exercise <exercise_name>
```

**Supported Exercises:**
* `bicep` (Bicep Curls)
* `squat` (Barbell Squats)
* `pushup` (Push-ups)
* `shoulder_press` (Dumbbell Shoulder Press)

**Example (Local Video):**
```bash
python main.py --source test_squat.mp4 --exercise squat
```

## Testing
The project includes a robust test suite for the Finite State Machine (including flexion, extension, and spatial constraints).
```bash
pytest test_counter.py -v
```