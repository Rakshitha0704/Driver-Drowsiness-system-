# =============================================================================
# live_inference.py
#
# HOW TO USE:
#   1. Make sure drowsiness_model.pt is in the same folder
#   2. Run this script (press F5 or python live_inference.py)
#   3. Press 'q' to quit
#
# ALERT LOGIC:
#   - Alert fires after 20 consecutive drowsy frames
#   - Sound plays in background thread for max 15 seconds
#   - Sound STOPS immediately when you become alert again
#   - New alert can fire again if drowsiness is detected again
# =============================================================================

import time
import platform
import threading
import cv2
import torch
import torch.nn as nn
import numpy as np
import mediapipe as mp

from config import (
    MODEL_SAVE_PATH, WINDOW_NAME,
    CONSEC_FRAMES,
    COLOR_GREEN, COLOR_RED, COLOR_YELLOW, COLOR_WHITE,
)
from detector import build_face_mesh, extract_landmarks, compute_features


# ── Model ─────────────────────────────────────────────────────────────────────

class DrowsinessNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.network(x)


# ── Load model ────────────────────────────────────────────────────────────────

print("Loading model...")
checkpoint  = torch.load(MODEL_SAVE_PATH, map_location="cpu")
model       = DrowsinessNet(
    input_size  = checkpoint["input_size"],
    hidden_size = checkpoint["hidden_size"],
    output_size = checkpoint["output_size"],
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

scaler_mean = np.array(checkpoint["scaler_mean"], dtype=np.float32)
scaler_std  = np.array(checkpoint["scaler_std"],  dtype=np.float32)

print(f"  Accuracy : {checkpoint['accuracy']*100:.2f}%")
print(f"  F1 score : {checkpoint['f1_score']:.4f}")
print("Model loaded.\n")


# ── Alert manager ─────────────────────────────────────────────────────────────

class AlertManager:
    """
    Manages the alert sound in a background thread.
    - Starts beeping when triggered
    - Stops immediately when stop() is called
    - Automatically stops after max_seconds
    - Webcam loop is never blocked
    """

    def __init__(self, max_seconds=15):
        self.max_seconds  = max_seconds
        self._stop_event  = threading.Event()
        self._thread      = None
        self.is_alerting  = False

    def start(self):
        """Fire the alert. Does nothing if already alerting."""
        if self.is_alerting:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._beep, daemon=True)
        self._thread.start()
        self.is_alerting = True

    def stop(self):
        """Stop the alert immediately."""
        self._stop_event.set()
        self.is_alerting = False

    def _beep(self):
        end_time = time.perf_counter() + self.max_seconds
        try:
            if platform.system() == "Windows":
                import winsound
                while not self._stop_event.is_set() and time.perf_counter() < end_time:
                    winsound.Beep(2500, 500)
            elif platform.system() == "Darwin":
                import subprocess
                while not self._stop_event.is_set() and time.perf_counter() < end_time:
                    subprocess.run(["afplay", "/System/Library/Sounds/Ping.aiff"],
                                   check=False)
            else:
                while not self._stop_event.is_set() and time.perf_counter() < end_time:
                    print("\a", end="", flush=True)
                    time.sleep(0.5)
        except Exception:
            pass
        # Auto-reset after max_seconds so alert can fire again
        self.is_alerting = False


# ── Helpers ───────────────────────────────────────────────────────────────────

def normalize(ear, mar):
    x = np.array([[ear, mar]], dtype=np.float32)
    return (x - scaler_mean) / scaler_std


class FPSCounter:
    def __init__(self, window=30):
        self._times  = []
        self._window = window

    def tick(self):
        self._times.append(time.perf_counter())
        if len(self._times) > self._window:
            self._times.pop(0)

    @property
    def fps(self):
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0


# ── Setup ─────────────────────────────────────────────────────────────────────

face_mesh   = build_face_mesh()
mp_drawing  = mp.solutions.drawing_utils
mp_styles   = mp.solutions.drawing_styles
alert_mgr   = AlertManager(max_seconds=15)
fps_counter = FPSCounter()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

frame_counter = 0
total_frames  = 0
drowsy_frames = 0

print("=" * 50)
print("Live inference started. Press 'q' to quit.")
print("=" * 50)

# ── Main loop ─────────────────────────────────────────────────────────────────

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    total_frames += 1
    fps_counter.tick()

    frame     = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results   = face_mesh.process(rgb_frame)

    ear          = None
    mar          = None
    prediction   = None
    confidence   = None
    inference_ms = None

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]

        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style(),
        )

        landmarks = extract_landmarks(face_landmarks, frame.shape)

        try:
            ear, mar = compute_features(landmarks)

            x_tensor = torch.tensor(normalize(ear, mar))

            inf_start = time.perf_counter()
            with torch.no_grad():
                logits     = model(x_tensor)
                probs      = torch.softmax(logits, dim=1)
                prediction = probs.argmax(dim=1).item()
                confidence = probs[0][prediction].item()
            inference_ms = (time.perf_counter() - inf_start) * 1000

            if prediction == 1:         # drowsy
                frame_counter += 1
                drowsy_frames += 1
                if frame_counter >= CONSEC_FRAMES:
                    alert_mgr.start()   # fires only if not already alerting
            else:                       # alert
                frame_counter = 0
                alert_mgr.stop()        # stop sound immediately when alert

        except ValueError:
            pass

    # ── HUD ───────────────────────────────────────────────────────────────────

    h, w = frame.shape[:2]

    if alert_mgr.is_alerting:
        cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 180), -1)
        cv2.putText(frame, "DROWSY — WAKE UP!",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.6, COLOR_WHITE, 3)
    elif prediction == 1:
        cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 120), -1)
        cv2.putText(frame, "DROWSY DETECTED",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.4, COLOR_YELLOW, 3)
    else:
        cv2.rectangle(frame, (0, 0), (w, 80), (0, 100, 0), -1)
        cv2.putText(frame, "ALERT",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.6, COLOR_WHITE, 3)

    if ear is not None:
        cv2.putText(frame, f"EAR: {ear:.3f}",
                    (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_YELLOW, 1)
        cv2.putText(frame, f"MAR: {mar:.3f}",
                    (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_YELLOW, 1)
        if confidence is not None:
            cv2.putText(frame, f"Conf: {confidence*100:.1f}%",
                        (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_YELLOW, 1)
        if inference_ms is not None:
            cv2.putText(frame, f"Inf: {inference_ms:.2f}ms",
                        (10, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_YELLOW, 1)

    cv2.putText(frame, f"FPS: {fps_counter.fps:.1f}",
                (w - 110, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_GREEN, 1)

    bar_width = min(int((frame_counter / CONSEC_FRAMES) * 200), 200)
    cv2.rectangle(frame, (10, h - 30), (210, h - 15), (60, 60, 60), -1)
    cv2.rectangle(frame, (10, h - 30), (10 + bar_width, h - 15), COLOR_RED, -1)
    cv2.putText(frame, "Drowsy meter",
                (10, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1)

    cv2.imshow(WINDOW_NAME, frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ── Cleanup ───────────────────────────────────────────────────────────────────

alert_mgr.stop()       # stop sound immediately on quit
cap.release()
cv2.destroyAllWindows()
face_mesh.close()

drowsy_pct = (drowsy_frames / total_frames * 100) if total_frames > 0 else 0

print("\n" + "=" * 50)
print("Session summary")
print("=" * 50)
print(f"  Total frames   : {total_frames}")
print(f"  Drowsy frames  : {drowsy_frames}  ({drowsy_pct:.1f}%)")
print(f"  Avg FPS        : {fps_counter.fps:.1f}")
print("=" * 50)
