# Realtime Pose Estimation

A collection of real-time computer vision projects focused on human pose estimation, hand tracking, and exercise detection using MediaPipe and OpenCV.

## Projects Included

| File | Description |
|------|-------------|
| `hand.py` | Real-time hand landmark detection and tracking |
| `full_skeleton.py` | Full-body pose estimation with skeleton visualization |
| `pushups_detection.py` | Automatic push-up counting using pose landmarks |

## Features

- Real-time hand tracking with 21 landmarks
- Full-body skeleton visualization (33 landmarks)
- Push-up detection & counting based on elbow/shoulder angles
- Webcam-based inference
- Clean OpenCV visualization

## Tech Stack

- Python 3.8+
- OpenCV
- MediaPipe
- NumPy

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/realtime-pose-estimation.git
cd realtime-pose-estimation
pip install opencv-python mediapipe numpy
```

## Usage

### Hand Tracking
```bash
python hand.py
```

### Full Body Skeleton
```bash
python full_skeleton.py
```

### Push-up Detection
```bash
python pushups_detection.py
```

## How It Works

- **Hand Tracking**: Uses MediaPipe Hands to detect and draw 21 hand landmarks in real time.
- **Full Skeleton**: Uses MediaPipe Pose to detect 33 body landmarks and draw the complete skeleton.
- **Push-up Counter**: Calculates the angle between shoulder, elbow, and wrist. Counts a repetition when the arm goes below a threshold and returns up.

## Project Structure

```
computer_vision/
├── hand.py
├── full_skeleton.py
├── pushups_detection.py
└── README.md
```

## Future Improvements

- [ ] Add squat / pull-up counters
- [ ] GUI with Streamlit or Gradio
- [ ] Save workout statistics
- [ ] Multi-person support

## Author

Your Name  
Computer Vision Enthusiast

## License

MIT
```
