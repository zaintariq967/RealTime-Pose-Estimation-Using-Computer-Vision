"""
═══════════════════════════════════════════════════════════════
  PUSH-UP DETECTOR PRO
  Core: Choose Camera  OR  Upload Video / Image
═══════════════════════════════════════════════════════════════
"""

import streamlit as st
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import tempfile
import os
import urllib.request
from PIL import Image

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Push-Up Detector Pro",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .block-container {
        padding-top: 1.2rem;
        max-width: 1100px;
    }

    .hero {
        background: linear-gradient(145deg, #0d0d12 0%, #16161f 100%);
        border: 1px solid #2a2a35;
        border-radius: 18px;
        padding: 2rem 2.2rem;
        text-align: center;
        margin-bottom: 1.8rem;
    }

    .hero h1 {
        font-size: 2.2rem;
        font-weight: 700;
        color: #f2f2f2;
        margin: 0 0 0.35rem 0;
        letter-spacing: -0.6px;
    }

    .hero p {
        color: #8a8a9a;
        font-size: 1rem;
        margin: 0;
    }

    .choice-card {
        background: #14141a;
        border: 1px solid #2c2c38;
        border-radius: 14px;
        padding: 1.6rem 1.4rem;
        text-align: center;
        transition: all 0.2s;
        height: 100%;
    }

    .choice-card:hover {
        border-color: #00c853;
        background: #18181f;
    }

    .choice-card h3 {
        color: #e8e8e8;
        font-size: 1.15rem;
        margin: 0.6rem 0 0.3rem 0;
    }

    .choice-card p {
        color: #777;
        font-size: 0.88rem;
        margin: 0;
    }

    .stButton > button {
        background: linear-gradient(90deg, #00c853, #69f0ae);
        color: #000;
        font-weight: 600;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 1.2rem;
        width: 100%;
        font-size: 0.95rem;
    }

    .stButton > button:hover {
        box-shadow: 0 4px 18px rgba(0, 200, 83, 0.35);
        transform: translateY(-1px);
    }

    .metric-card {
        background: #14141a;
        border: 1px solid #2c2c38;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# MODEL + HELPERS
# ─────────────────────────────────────────────────────────────
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
)
MODEL_PATH = "pose_landmarker_heavy.task"

L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24

GREEN  = (80, 220, 100)
RED    = (50, 50, 255)
YELLOW = (0, 200, 255)
WHITE  = (245, 245, 245)
CYAN   = (255, 200, 50)
DARK   = (16, 16, 20)


@st.cache_resource
def get_model_path():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading AI model (first run only)..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def angle_at(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    ang = np.abs(rad * 180.0 / np.pi)
    return 360 - ang if ang > 180 else ang


def pt(landmarks, idx, w, h):
    lm = landmarks[idx]
    return int(lm.x * w), int(lm.y * h)


def analyze(frame, landmarker, state, image_mode=False):
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    if image_mode:
        result = landmarker.detect(mp_image)
    else:
        state["ts"] += 33
        result = landmarker.detect_for_video(mp_image, state["ts"])

    if not result.pose_landmarks:
        state["feedback"] = "No person detected"
        state["form_ok"] = False
        return frame, state

    lm = result.pose_landmarks[0]

    ls = pt(lm, L_SHOULDER, w, h)
    rs = pt(lm, R_SHOULDER, w, h)
    le = pt(lm, L_ELBOW, w, h)
    re = pt(lm, R_ELBOW, w, h)
    lw = pt(lm, L_WRIST, w, h)
    rw = pt(lm, R_WRIST, w, h)
    lh = pt(lm, L_HIP, w, h)
    rh = pt(lm, R_HIP, w, h)

    shoulder = ((ls[0]+rs[0])//2, (ls[1]+rs[1])//2)
    elbow    = ((le[0]+re[0])//2, (le[1]+re[1])//2)
    wrist    = ((lw[0]+rw[0])//2, (lw[1]+rw[1])//2)
    hip      = ((lh[0]+rh[0])//2, (lh[1]+rh[1])//2)

    elbow_ang = angle_at(shoulder, elbow, wrist)
    state["elbow"] = elbow_ang
    form_ok = abs(shoulder[1] - hip[1]) < h * 0.18
    state["form_ok"] = form_ok

    if image_mode:
        if elbow_ang < 100:
            state["stage"] = "DOWN"
            state["feedback"] = "PUSH-UP  •  DOWN position"
        elif elbow_ang > 150:
            state["stage"] = "UP"
            state["feedback"] = "PUSH-UP  •  UP position"
        else:
            state["stage"] = "MID"
            state["feedback"] = "MID / Transition"
        if not form_ok:
            state["feedback"] += "  •  Keep body straight"
    else:
        if elbow_ang < 95:
            if state["stage"] == "UP":
                state["stage"] = "DOWN"
                state["feedback"] = "Going down..."
        elif elbow_ang > 155:
            if state["stage"] == "DOWN":
                state["stage"] = "UP"
                state["reps"] += 1
                state["feedback"] = "Good rep!"
            else:
                state["feedback"] = "Ready"
        if not form_ok:
            state["feedback"] = "Keep body straight"

    # Draw skeleton
    def bone(p1, p2, col=GREEN, t=3):
        cv2.line(frame, p1, p2, col, t, cv2.LINE_AA)

    bone(ls, rs, CYAN, 2)
    bone(ls, le)
    bone(le, lw)
    bone(rs, re)
    bone(re, rw)
    bone(ls, lh)
    bone(rs, rh)
    bone(lh, rh, CYAN, 2)

    for p in [ls, rs, le, re, lw, rw, lh, rh]:
        cv2.circle(frame, p, 6, RED, -1, cv2.LINE_AA)
        cv2.circle(frame, p, 3, WHITE, -1, cv2.LINE_AA)

    # HUD
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 76), DARK, -1)
    cv2.rectangle(overlay, (0, h-46), (w, h), DARK, -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    cv2.putText(frame, f"REPS  {state['reps']}", (20, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, GREEN, 3, cv2.LINE_AA)

    stage_col = YELLOW if state["stage"] == "DOWN" else GREEN
    cv2.putText(frame, f"STAGE  {state['stage']}", (w-240, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, stage_col, 2, cv2.LINE_AA)

    cv2.putText(frame, f"Elbow  {int(elbow_ang)} deg", (20, h-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, WHITE, 2, cv2.LINE_AA)

    fb_col = GREEN if form_ok else RED
    cv2.putText(frame, state["feedback"], (w//2-150, h-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, fb_col, 2, cv2.LINE_AA)

    return frame, state


def make_landmarker(image_mode=False):
    path = get_model_path()
    mode = vision.RunningMode.IMAGE if image_mode else vision.RunningMode.VIDEO
    opts = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=path),
        running_mode=mode,
        num_poses=1,
        min_pose_detection_confidence=0.6,
        min_pose_presence_confidence=0.6,
        min_tracking_confidence=0.6
    )
    return vision.PoseLandmarker.create_from_options(opts)


def init_state():
    return {
        "stage": "UP",
        "reps": 0,
        "feedback": "Ready",
        "form_ok": True,
        "elbow": 0,
        "ts": 0
    }


# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "home"


def go_home():
    st.session_state.page = "home"


def go_camera():
    st.session_state.page = "camera"


def go_video():
    st.session_state.page = "video"


def go_image():
    st.session_state.page = "image"


# ─────────────────────────────────────────────────────────────
# HOME – FIRST CHOICE
# ─────────────────────────────────────────────────────────────
if st.session_state.page == "home":
    st.markdown("""
    <div class="hero">
        <h1>💪 Push-Up Detector Pro</h1>
        <p>Choose how you want to analyze push-ups</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Select Input Source")
    st.write("")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="choice-card">
            <div style="font-size:2.4rem;">📹</div>
            <h3>Live Camera</h3>
            <p>Real-time detection &amp; rep counting using your webcam</p>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        st.button("Use Camera", key="btn_cam", on_click=go_camera, use_container_width=True)

    with col2:
        st.markdown("""
        <div class="choice-card">
            <div style="font-size:2.4rem;">🎬</div>
            <h3>Upload Video</h3>
            <p>Upload a recorded video and get full analysis + download</p>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        st.button("Upload Video", key="btn_vid", on_click=go_video, use_container_width=True)

    with col3:
        st.markdown("""
        <div class="choice-card">
            <div style="font-size:2.4rem;">🖼️</div>
            <h3>Upload Image</h3>
            <p>Check a single photo for UP / DOWN / MID position</p>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        st.button("Upload Image", key="btn_img", on_click=go_image, use_container_width=True)

    st.write("")
    st.markdown("---")
    st.caption("Core capabilities: Elbow angle tracking • Rep counting • Body alignment check • Real-time feedback")

# ─────────────────────────────────────────────────────────────
# CAMERA PAGE
# ─────────────────────────────────────────────────────────────
elif st.session_state.page == "camera":
    st.button("← Back to Home", on_click=go_home)
    st.subheader("Live Camera Mode")
    st.caption("Perform push-ups in front of the camera. Click Start when ready.")

    c1, c2 = st.columns([3.2, 1])
    start = c2.button("▶  Start Camera", use_container_width=True)
    stop  = c2.button("■  Stop", use_container_width=True)

    frame_slot = c1.empty()
    metric_slot = c2.empty()

    if start:
        landmarker = make_landmarker(False)
        state = init_state()
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while cap.isOpened() and not stop:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            frame, state = analyze(frame, landmarker, state)
            frame_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                             channels="RGB", use_container_width=True)

            with metric_slot.container():
                st.metric("Reps", state["reps"])
                st.metric("Stage", state["stage"])
                st.metric("Elbow", f"{int(state['elbow'])}°")
                st.write(state["feedback"])

        cap.release()
        landmarker.close()
        st.success(f"Session finished  •  Total reps: **{state['reps']}**")

# ─────────────────────────────────────────────────────────────
# VIDEO PAGE
# ─────────────────────────────────────────────────────────────
elif st.session_state.page == "video":
    st.button("← Back to Home", on_click=go_home)
    st.subheader("Upload Video")
    st.caption("Upload a video of push-ups. The app will count reps and let you download the analyzed video.")

    uploaded = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov", "mkv"],
        help="MP4, AVI, MOV or MKV"
    )

    if uploaded is not None:
        st.video(uploaded)

        if st.button("🔍 Analyze Video", use_container_width=True):
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded.read())
            path = tfile.name

            landmarker = make_landmarker(False)
            state = init_state()

            cap = cv2.VideoCapture(path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            progress = st.progress(0)
            status = st.empty()
            preview = st.empty()
            frames = []
            idx = 0

            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame, state = analyze(frame, landmarker, state)
                frames.append(frame)
                idx += 1
                progress.progress(min(idx / total, 1.0))
                status.caption(f"Processing frame {idx}/{total}  •  Reps so far: {state['reps']}")
                if idx % 5 == 0:
                    preview.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                  channels="RGB", use_container_width=True)

            cap.release()
            landmarker.close()

            if frames:
                h, w = frames[0].shape[:2]
                out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
                for f in frames:
                    writer.write(f)
                writer.release()

                st.success("Analysis complete")
                m1, m2, m3 = st.columns(3)
                m1.metric("Total Reps", state["reps"])
                m2.metric("Final Stage", state["stage"])
                m3.metric("Last Elbow Angle", f"{int(state['elbow'])}°")

                with open(out_path, "rb") as f:
                    st.download_button(
                        "⬇ Download Analyzed Video",
                        f,
                        file_name="pushup_analyzed.mp4",
                        mime="video/mp4",
                        use_container_width=True
                    )

# ─────────────────────────────────────────────────────────────
# IMAGE PAGE
# ─────────────────────────────────────────────────────────────
elif st.session_state.page == "image":
    st.button("← Back to Home", on_click=go_home)
    st.subheader("Upload Image")
    st.caption("Upload a photo to check if the person is in UP, DOWN or MID push-up position.")

    uploaded = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded is not None:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Original Image", use_container_width=True)

        if st.button("🔍 Analyze Image", use_container_width=True):
            arr = np.array(img)
            frame = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

            landmarker = make_landmarker(True)
            state = init_state()
            result, state = analyze(frame, landmarker, state, image_mode=True)
            landmarker.close()

            st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
                     caption="Analyzed Result", use_container_width=True)

            m1, m2, m3 = st.columns(3)
            m1.metric("Stage", state["stage"])
            m2.metric("Elbow Angle", f"{int(state['elbow'])}°")
            m3.metric("Form", "Good ✅" if state["form_ok"] else "Needs work ⚠️")

            st.info(state["feedback"])