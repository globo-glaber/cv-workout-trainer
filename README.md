# AI Workout Rep Controller

A modular Computer Vision application that tracks workout repetitions, computes aspect-ratio-invariant joint angles, and enforces posture rules in real time using the MediaPipe Tasks API and OpenCV.

## Features
- **Modular Architecture**: Clean separation of concerns across configuration, pose detection, signal processing/FSM counting, and UI rendering.
- **Aspect-Ratio-Invariant Pixel-Space Angles**: Converts normalized landmarks into true pixel coordinates `(x * w, y * h)` before computing joint angles, ensuring consistent thresholds across horizontal (`16:9`), vertical (`9:16` Shorts/TikTok), and square videos.
- **Temporal Tracking (`RunningMode.VIDEO`)**: Uses MediaPipe's video tracking mode with monotonically increasing frame timestamps for lower jitter and higher stability than per-frame image detection.
- **FPS-Independent Time-Based EMA**: Applies exponential smoothing using elapsed frame time ($\alpha = 1 - e^{-\Delta t / \tau}$, $\tau = 35\text{ ms}$) and video-timestamp cooldowns, preventing fast reps from being clipped across 30/60 FPS inputs.
- **Exercise Form Enforcement**: Validates posture for every supported movement: upright torso and elbows down (`bicep`), upright torso (`squat`), horizontal plank alignment (`pushup`), and wrists above shoulders (`shoulder_press`).
- **Hysteresis-Protected Bilateral Tracking**: Automatically tracks the more visible side for bilateral movements (`squat`, `pushup`) with a `+0.15` visibility hysteresis margin, minimum visibility gating, and EMA reset on side switches to prevent cross-limb signal mixing.
- **Pure OpenCV Rendering**: Renders the skeleton and dynamic HUD directly with OpenCV without mixing legacy `mp.solutions` APIs.
- **Pause/Analyze Mode**: Press `SPACE` to freeze any frame and inspect exact angles, FSM stage, and form alerts (`Q` to quit).

## Architecture
- `config.py`: Immutable `@dataclass(frozen=True)` exercise registry (`EXERCISES`) with `Literal`-typed movement mechanics and form rules.
- `detector.py`: MediaPipe Tasks `PoseLandmarker` wrapper in `VIDEO` mode, atomic model downloading (`.tmp` -> `.task`), and visibility-filtered OpenCV skeleton rendering.
- `counter.py`: Pure vector-angle math, posture validation (`evaluate_form`), time-based EMA filtering, and Finite State Machine (`ExerciseCounter`).
- `ui.py`: Aspect-ratio-preserving frame resizing and real-time HUD rendering (reps, angles, FSM stage, progress bars, form hints).
- `main.py`: CLI entry point, monotonic timestamp generation, bilateral hysteresis logic, and main video loop.

## Design Note: Pixel-Scaled 2D vs. Monocular 3D
During development, MediaPipe's `pose_world_landmarks` (3D metric coordinates) was evaluated alongside pixel-scaled 2D coordinates. On monocular RGB video (webcam and smartphone clips), estimated Z-axis depth noise systematically compressed extreme joint angles (for example, shifting a ~60 deg bicep curl to ~85 deg and a straight overhead press to ~150 deg). Pixel-scaled 2D coordinates eliminate aspect-ratio distortion while preserving full anatomical angle range without depth hallucination.

## Requirements
- **Python 3.10 - 3.12** (required for prebuilt `mediapipe==0.10.14` and `numpy==1.26.4` wheels)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/globo-glaber/cv-workout-trainer.git
    cd cv-workout-trainer
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv .venv

    # On Windows:
    .\.venv\Scripts\activate

    # On macOS/Linux:
    source .venv/bin/activate
    ```

3. Install runtime dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: The `pose_landmarker_full.task` model is downloaded atomically on the first run.*

4. *(Optional)* Install development/testing dependencies:
    ```bash
    pip install -r requirements-dev.txt
    ```

## Usage

Run via CLI using a webcam (`0`) or a path to a local video file:

```bash
python main.py --source <path_or_0> --exercise <exercise_name>
```

**Supported Exercises:**
- `bicep` — Bicep Curls (unilateral, flexion)
- `squat` — Barbell Squats (bilateral, flexion)
- `pushup` — Push-ups (bilateral, flexion)
- `shoulder_press` — Dumbbell Shoulder Press (unilateral, extension)

**Example (Local Video):**
```bash
python main.py --source test_squat.mp4 --exercise squat
```

## Testing

The test suite (`test_counter.py`) covers angle geometry (including tilted limbs and degenerate overlaps), aspect-ratio invariance (`16:9`, `9:16`, `1:1`), flexion/extension FSM transitions, FPS-independent EMA response, fast 1s-3s rep simulations at 30/60 FPS, video-timestamp cooldowns, cross-limb EMA resets, form rules, and config immutability.

```bash
pytest test_counter.py -v
```