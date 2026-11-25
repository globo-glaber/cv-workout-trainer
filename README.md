# AI Workout Rep Controller

![Python](https://img.shields.io/badge/Python-3.10%20--%203.12-blue)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Tasks%20API%200.10.14-0097A7)
![OpenCV](https://img.shields.io/badge/OpenCV-4.10-5C3EE8)
![Tests](https://img.shields.io/badge/pytest-32%20passing-brightgreen)

Real-time Computer Vision rep counter and form checker built with **MediaPipe BlazePose**, **OpenCV**, and **NumPy**. Works with a standard webcam or local `.mp4` videos (both `16:9` horizontal and `9:16` vertical Shorts/TikTok clips) without needing a GPU or depth camera.

---

## Highlights

- **4 Supported Exercises** — Bicep Curls, Barbell Squats, Push-Ups, and Dumbbell Shoulder Press.
- **Works on Any Aspect Ratio** — Calculates joint angles in true pixel space instead of raw normalized coordinates, so vertical and widescreen videos behave identically.
- **Real-Time Form Validation** — Checks body posture on every frame (e.g., rejects standing arm curls in push-up mode) and shows live HUD warnings.
- **Smooth & FPS-Independent** — Uses MediaPipe `VIDEO` tracking mode with a time-based EMA filter so fast reps are not clipped across 30/60 FPS streams.
- **Anatomically Correct Webcam Mirror** — Detects pose on the original unflipped frame so `LEFT REPS` and `RIGHT REPS` match your real arms, then mirrors the preview naturally.
- **Freeze-Frame Inspection** — Pause any frame with `SPACE` to check exact angles, FSM stage, and form hints.

---

## Quick Start

### Requirements
- **Python 3.10 – 3.12** (required for prebuilt `mediapipe==0.10.14` and `numpy==1.26.4` wheels)
- A webcam or local `.mp4` workout video

### Installation

```bash
git clone https://github.com/globo-glaber/cv-workout-trainer.git
cd cv-workout-trainer

python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

> On the first run, `pose_landmarker_full.task` is downloaded automatically (atomically via `.tmp` with `Content-Length` verification).

### Usage

Run from the terminal with `--source` (`0` for webcam or path to a video file) and `--exercise`:

```bash
# Live webcam (default: bicep curls)
python main.py --source 0 --exercise bicep

# Local video files
python main.py --source test_squat.mp4 --exercise squat
python main.py --source test_pushup.mp4 --exercise pushup
python main.py --source test_shoulderpress.mp4 --exercise shoulder_press
```

### Controls

| Key | Action |
| :--- | :--- |
| `SPACE` | Pause / resume playback to inspect angles and form on a single frame |
| `Q` / `ESC` / Window `X` | Quit cleanly (works regardless of English or Cyrillic keyboard layout) |

---

## Supported Exercises

| Flag (`--exercise`) | Exercise | Mode | Thresholds (Ext / Flex) | Form Check | Best Camera Angle |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `bicep` | Bicep Curl | Left & Right | $> 150^\circ$ / $< 65^\circ$ | Upright torso ($< 45^\circ$), elbow below shoulder | Front or $30^\circ\text{--}45^\circ$ |
| `squat` | Barbell Squat | Bilateral (Total) | $> 155^\circ$ / $< 100^\circ$ | Upright torso ($< 65^\circ$) | Side profile |
| `pushup` | Push-Up | Bilateral (Total) | $> 150^\circ$ / $< 90^\circ$ | Horizontal plank ($< 45^\circ$ from horizontal) | Side profile |
| `shoulder_press` | Shoulder Press | Left & Right | $> 150^\circ$ / $< 90^\circ$ | Wrist positioned above shoulder | Front view |

---

## How It Works

Each frame flows through a lightweight, pure Tasks API pipeline (no legacy `mp.solutions`):

1. **Pose Detection (`detector.py`)** — Runs MediaPipe `PoseLandmarker` in `RunningMode.VIDEO` with strictly monotonic timestamps (`CAP_PROP_POS_MSEC` for videos, `time.monotonic()` for webcams).
2. **Side Selection & Occlusion Gating (`main.py`)** — Bilateral exercises lock onto the clearer side using a $+0.15$ visibility hysteresis margin and reset the filter on side switches. Low-visibility limbs ($< 0.60$ bilateral / $< 0.55$ unilateral) are skipped to avoid false counts.
3. **Geometry & Form (`counter.py`)** — Converts landmarks to pixel space `(x * w, y * h)`, checks posture rules via `evaluate_form()`, applies time-based EMA smoothing, and updates the rep Finite State Machine with a `400 ms` video-time cooldown.
4. **HUD & Display (`ui.py`)** — Draws the visibility-filtered skeleton directly in OpenCV, mirrors live webcam frames for natural viewing, overlays reps/angles/progress bars, and resizes to 720p height while keeping aspect ratio intact.

<details>
<summary><strong>Technical Details: Why Pixel-Scaled 2D Instead of 3D, and Time-Based EMA</strong></summary>

#### Pixel-Scaled 2D vs. Monocular 3D (`pose_world_landmarks`)
Computing angles directly on MediaPipe's normalized `(0..1)` coordinates distorts non-square frames (on `9:16` video, a true $50^\circ$ curl stretches to $\approx 65^\circ$ and a $160^\circ$ extension shrinks to $\approx 147^\circ$).

Both metric 3D (`pose_world_landmarks`) and pixel-scaled 2D `(x * w, y * h)` were benchmarked on real clips:

| Exercise & Position | Pixel-Scaled 2D | Monocular 3D | Result |
| :--- | :--- | :--- | :--- |
| **Shoulder Press** (bottom / top) | $32^\circ$ / $158^\circ$ | $109^\circ$ / $148^\circ$ | Z-axis depth noise prevented 3D from entering `flexed` |
| **Bicep Curl** (`9:16`, top) | $58^\circ$ / $61^\circ$ | $86^\circ$ / $90^\circ$ | 3D added $\approx +28^\circ$ depth bias and missed reps |
| **Push-Up** (bottom) | $71^\circ$ | $88^\circ$ | 2D captured true depth clearly below the $90^\circ$ cutoff |

Pixel-scaled 2D eliminates aspect-ratio distortion without the depth estimation bias of a single RGB camera.

#### FPS-Independent EMA
Instead of a fixed per-frame factor (which clips fast 1-second reps and changes behavior between 30 and 60 FPS), smoothing is parameterized by elapsed frame time $\Delta t$ with $\tau = 35\text{ ms}$:

$$\alpha = 1 - e^{-\Delta t / \tau}$$

</details>

---

## Project Structure

```text
cv-workout-trainer/
├── main.py              # CLI entry point, video loop, hysteresis & webcam mirroring
├── detector.py          # MediaPipe Tasks PoseLandmarker (VIDEO mode), OpenCV skeleton & atomic model download
├── counter.py           # Pixel-space angle math, posture rules, time-based EMA & FSM
├── config.py            # Immutable ExerciseConfig dataclasses and exercise registry
├── ui.py                # Aspect-ratio-safe resizing, HUD layout & form alerts
├── test_counter.py      # 32 unit tests (geometry, EMA, FSM, form rules, UI, download recovery)
├── requirements.txt     # Pinned runtime dependencies
└── requirements-dev.txt # Development dependencies (pytest)
```

---

## Running Tests

Install dev dependencies and run the test suite (32 deterministic unit tests, runs in `< 1s` without a camera or internet):

```bash
pip install -r requirements-dev.txt
pytest test_counter.py -v
```

Tests verify angle geometry, `16:9` / `9:16` / `1:1` invariance, FPS-independent EMA equivalence, fast `1s–3s` rep simulations at 30/60 FPS, cooldown behavior, all four posture rules, `0 deg` HUD formatting, config immutability, and interrupted model download cleanup.

---

## Camera Tips & Limitations

- **Occluded Limbs:** On a single RGB camera, strictly side-profile bicep curls intentionally track only the camera-facing arm rather than guessing the hidden arm behind the torso. Use a front or $30^\circ\text{--}45^\circ$ angle to track both arms.
- **Recommended Angles:** Record `squat` and `pushup` from the side so knee/elbow flexion stays parallel to the camera; record `shoulder_press` from the front.
- **Lighting & Equipment:** Heavy dumbbells crossing directly in front of the wrist or poor lighting can briefly drop landmark visibility.