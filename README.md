# Hand Controlled 3D Cube

A 3D cube controlled by hand gestures in real time using MediaPipe and Pygame.

## Demo
- **Left hand** controls X rotation (tilt up/down)
- **Right hand** controls Y rotation (tilt left/right)
- **Open hand** to rotate, **close fist** to freeze that axis

## Setup

### Install dependencies
pip install opencv-python mediapipe pygame numpy

### Download the MediaPipe hand tracking model
Download `hand_landmarker.task` from [Google's hand landmarking guide](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) and place it in a `models/` folder.

### Run
python handTracking.py