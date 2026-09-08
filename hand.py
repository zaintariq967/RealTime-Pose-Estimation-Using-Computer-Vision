"""
Hand Skeleton Tracker
Uses MediaPipe Tasks (Hand Landmarker) + OpenCV + Tkinter
to show a live webcam feed with the hand skeleton drawn on it.

Press 'q' or close the window to exit.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import tkinter as tk
from PIL import Image, ImageTk
import urllib.request
import os

# -------------------------------------------------
# Model download (only once)
# -------------------------------------------------
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = "hand_landmarker.task"


def download_model_if_needed():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand_landmarker.task model (one-time)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded successfully.")


# -------------------------------------------------
# Hand connections (21 landmarks)
# -------------------------------------------------
HAND_CONNECTIONS = [
    # Palm
    (0, 1), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17),
    # Thumb
    (1, 2), (2, 3), (3, 4),
    # Index
    (5, 6), (6, 7), (7, 8),
    # Middle
    (9, 10), (10, 11), (11, 12),
    # Ring
    (13, 14), (14, 15), (15, 16),
    # Pinky
    (17, 18), (18, 19), (19, 20),
]


def draw_hand_skeleton(frame, hand_landmarks):
    """Draw landmarks and connections on the BGR frame."""
    h, w, _ = frame.shape

    # Convert normalized landmarks → pixel coordinates
    points = []
    for lm in hand_landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        points.append((x, y))

    # Draw connections (skeleton bones)
    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(frame, points[start_idx], points[end_idx], (0, 255, 0), 2)

    # Draw landmark points
    for pt in points:
        cv2.circle(frame, pt, 5, (0, 0, 255), -1)


class HandSkeletonApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hand Skeleton Tracker - MediaPipe + Tkinter")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Video display
        self.video_label = tk.Label(root)
        self.video_label.pack(padx=10, pady=10)

        # Status label
        self.status_label = tk.Label(
            root,
            text="Show your hand to the camera. Press 'q' or close window to quit.",
            font=("Arial", 11),
            fg="#333"
        )
        self.status_label.pack(pady=(0, 10))

        # Download model if needed
        download_model_if_needed()

        # Create HandLandmarker (VIDEO mode)
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

        # Webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_label.config(text="Error: Could not open webcam.", fg="red")
            return

        self.running = True
        self.frame_timestamp_ms = 0
        self.update_frame()

    def update_frame(self):
        if not self.running:
            return

        success, frame = self.cap.read()
        if not success:
            self.status_label.config(text="Failed to grab frame.", fg="red")
            self.root.after(30, self.update_frame)
            return

        # Mirror for selfie view
        frame = cv2.flip(frame, 1)

        # Convert BGR → RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect hands
        self.frame_timestamp_ms += 33  # ~30 FPS
        result = self.landmarker.detect_for_video(mp_image, self.frame_timestamp_ms)

        # Draw skeleton
        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                draw_hand_skeleton(frame, hand_landmarks)

            num_hands = len(result.hand_landmarks)
            self.status_label.config(
                text=f"Detected {num_hands} hand(s). Press 'q' or close window to quit.",
                fg="green"
            )
        else:
            self.status_label.config(
                text="No hand detected. Show your hand to the camera.",
                fg="#333"
            )

        # Show frame in Tkinter
        rgb_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_display)
        imgtk = ImageTk.PhotoImage(image=img)

        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.root.after(15, self.update_frame)

    def on_closing(self):
        self.running = False
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.release()
        if hasattr(self, "landmarker"):
            self.landmarker.close()
        self.root.destroy()


def main():
    root = tk.Tk()

    def on_key(event):
        if event.char.lower() == "q":
            app.on_closing()

    root.bind("<Key>", on_key)
    app = HandSkeletonApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()