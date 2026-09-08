"""
═══════════════════════════════════════════════════════════════
  PRO BODY SKELETON  |  Real-time Pose Tracking
  MediaPipe Pose Landmarker + OpenCV + Tkinter
═══════════════════════════════════════════════════════════════
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw, ImageFont
import urllib.request
import os
import time
import math

# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
)
MODEL_PATH = "pose_landmarker_heavy.task"

# Professional color palette (BGR)
COLOR_BONE       = (180, 255, 100)   # Soft neon green
COLOR_JOINT      = (80, 80, 255)     # Soft red
COLOR_JOINT_CORE = (255, 255, 255)   # White core
COLOR_ACCENT     = (255, 180, 50)    # Cyan accent
COLOR_BG_PANEL   = (25, 25, 30)

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]


def download_model():
    if not os.path.exists(MODEL_PATH):
        print("↓ Downloading professional pose model (one-time)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("✓ Model ready.")


def draw_glow_line(img, pt1, pt2, color, thickness=3):
    """Draw a soft glow line for premium look."""
    # Outer glow
    cv2.line(img, pt1, pt2, tuple(c // 3 for c in color), thickness + 4, cv2.LINE_AA)
    # Main bone
    cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_AA)


def draw_joint(img, center, radius=7):
    """Draw a professional joint with core + ring."""
    cv2.circle(img, center, radius + 2, COLOR_JOINT, 2, cv2.LINE_AA)
    cv2.circle(img, center, radius - 1, COLOR_JOINT_CORE, -1, cv2.LINE_AA)


def draw_body_skeleton(frame, landmarks):
    h, w = frame.shape[:2]
    pts = []

    for lm in landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        vis = getattr(lm, "visibility", 1.0)
        pts.append((x, y, vis))

    # Bones
    for a, b in POSE_CONNECTIONS:
        if a < len(pts) and b < len(pts):
            x1, y1, v1 = pts[a]
            x2, y2, v2 = pts[b]
            if v1 > 0.55 and v2 > 0.55:
                draw_glow_line(frame, (x1, y1), (x2, y2), COLOR_BONE, 3)

    # Joints
    for x, y, v in pts:
        if v > 0.55:
            draw_joint(frame, (x, y))


class ProBodySkeletonApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PRO BODY SKELETON  •  Real-time Pose Tracking")
        self.root.configure(bg="#0f0f12")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # ── Header ──────────────────────────────────────────
        header = tk.Frame(root, bg="#0f0f12")
        header.pack(fill="x", padx=20, pady=(15, 5))

        tk.Label(
            header,
            text="PRO BODY SKELETON",
            font=("Segoe UI", 18, "bold"),
            fg="#e8e8e8",
            bg="#0f0f12"
        ).pack(side="left")

        self.fps_label = tk.Label(
            header,
            text="FPS: --",
            font=("Consolas", 11),
            fg="#7cffb2",
            bg="#0f0f12"
        )
        self.fps_label.pack(side="right")

        # ── Video Canvas ────────────────────────────────────
        self.video_label = tk.Label(root, bg="#1a1a1f", bd=0)
        self.video_label.pack(padx=20, pady=10)

        # ── Status Bar ──────────────────────────────────────
        status_frame = tk.Frame(root, bg="#0f0f12")
        status_frame.pack(fill="x", padx=20, pady=(0, 15))

        self.status_dot = tk.Label(
            status_frame, text="●", font=("Segoe UI", 12),
            fg="#555", bg="#0f0f12"
        )
        self.status_dot.pack(side="left")

        self.status_text = tk.Label(
            status_frame,
            text="  Initializing camera...",
            font=("Segoe UI", 10),
            fg="#aaa",
            bg="#0f0f12"
        )
        self.status_text.pack(side="left")

        tk.Label(
            status_frame,
            text="Press  Q  to quit",
            font=("Segoe UI", 9),
            fg="#555",
            bg="#0f0f12"
        ).pack(side="right")

        # ── Setup MediaPipe ─────────────────────────────────
        download_model()

        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.55,
            min_pose_presence_confidence=0.55,
            min_tracking_confidence=0.55
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

        if not self.cap.isOpened():
            self.status_text.config(text="  Camera not found", fg="#ff5555")
            self.status_dot.config(fg="#ff5555")
            return

        self.running = True
        self.frame_ts = 0
        self.prev_time = time.time()
        self.fps = 0
        self.fps_smooth = 0

        self.root.bind("<Key>", self.on_key)
        self.update()

    def on_key(self, event):
        if event.char.lower() == "q":
            self.on_close()

    def update(self):
        if not self.running:
            return

        ok, frame = self.cap.read()
        if not ok:
            self.root.after(20, self.update)
            return

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # MediaPipe detection
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self.frame_ts += 33
        result = self.landmarker.detect_for_video(mp_img, self.frame_ts)

        detected = False
        if result.pose_landmarks:
            for landmarks in result.pose_landmarks:
                draw_body_skeleton(frame, landmarks)
            detected = True

        # FPS calculation (smoothed)
        now = time.time()
        instant_fps = 1.0 / max(now - self.prev_time, 1e-6)
        self.prev_time = now
        self.fps_smooth = self.fps_smooth * 0.9 + instant_fps * 0.1
        self.fps_label.config(text=f"FPS: {self.fps_smooth:.0f}")

        # Overlay subtle vignette for cinematic feel
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 40), (15, 15, 20), -1)
        cv2.rectangle(overlay, (0, h - 35), (w, h), (15, 15, 20), -1)
        frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

        # Status update
        if detected:
            self.status_dot.config(fg="#7cffb2")
            self.status_text.config(text="  Body locked", fg="#7cffb2")
        else:
            self.status_dot.config(fg="#ffaa44")
            self.status_text.config(text="  Searching for body...", fg="#ffaa44")

        # Push to Tkinter
        rgb_show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_show)
        photo = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = photo
        self.video_label.configure(image=photo)

        self.root.after(12, self.update)

    def on_close(self):
        self.running = False
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.release()
        if hasattr(self, "landmarker"):
            self.landmarker.close()
        self.root.destroy()


def main():
    root = tk.Tk()
    root.resizable(False, False)
    app = ProBodySkeletonApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()