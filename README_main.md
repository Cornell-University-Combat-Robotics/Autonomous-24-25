# main.py

The `main.py` file is the main code that calls every other function from the subsystems.

## Table of Contents
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
- [Configuration](#configuration)

## Features
- Captures video from a camera or pre-recorded file
- Warps the captured image using homography transformation
- Detects objects using YOLO or Roboflow models
- Identifies robot corners and orientation
- Computes movement decisions based on detected positions
- Sends movement commands to motors via serial communication
- Displays visual debugging information

## Prerequisites
Ensure you have the following installed:
- Python 3.10+
- OpenCV
- NumPy
- `line_profiler` (for profiling)
- `torch` (for machine learning models)
- `cv2`, `os`, `time`
- Additional dependencies for `RoboflowModel` and `YoloModel`

## Usage
Run the main script using:
```bash
python main.py
```
or 
```bash
python3 main.py
```
For debugging and profiling:
```bash
kernprof -l -v --unit 1e-3 main.py
```

## Configuration
Modify the following global variables in `main.py` to change behavior:

- `COMP_SETTINGS`: Optimize for competition (disable visuals)
- `WARP_AND_COLOR_PICKING`: Enable redoing warp and color picking
- `IS_TRANSMITTING`: Set to True when using a live robot
- `SHOW_FRAME`: Toggle displaying frames
- `DISPLAY_ANGLES`: Show current and future orientation angles
- `IS_ORIGINAL_FPS`: Process every frame
- `camera_number`: Path to video file or camera input