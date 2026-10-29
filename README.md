# CV Workout Tracker

A Computer Vision application that tracks body posture and automatically counts exercise repetitions in real-time. This project was developed as a university lab assignment to demonstrate practical usage of computer vision and stream processing.

## Tech Stack
* Python
* OpenCV (Video capture & HUD rendering)
* MediaPipe (Pose Landmark Detection)
* NumPy (Trigonometry & angle calculation)

## Features
* **Real-time skeletal tracking:** Extracts exact (X, Y) coordinates for shoulders, elbows, and wrists.
* **Dynamic angle calculation:** Computes joint angles on each frame.
* **State Machine logic:** Registers "up" and "down" movement phases to accurately increment the counter.
* **Anti-flickering mechanism:** Implements a time-based cooldown to prevent false multiple counts caused by frame noise.
