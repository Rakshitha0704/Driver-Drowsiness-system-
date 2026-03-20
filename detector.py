import numpy as np
import mediapipe as mp

from config import (
    LEFT_EYE_IDX, RIGHT_EYE_IDX, MOUTH_IDX,
    MAX_NUM_FACES, DETECTION_CONFIDENCE, TRACKING_CONFIDENCE,
)


# ── MediaPipe setup ───────────────────────────────────────────────────────────

def build_face_mesh():
    """
    Create and return a MediaPipe FaceMesh object.
    Call this once at startup and reuse the same object.
    """
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=MAX_NUM_FACES,
        refine_landmarks=True,
        min_detection_confidence=DETECTION_CONFIDENCE,
        min_tracking_confidence=TRACKING_CONFIDENCE,
    )


# ── Landmark extraction ───────────────────────────────────────────────────────

def extract_landmarks(face_landmarks, frame_shape):
    """
    Convert normalised MediaPipe landmarks to pixel (x, y) coordinates.
    Returns np.ndarray of shape (468, 2).
    """
    h, w = frame_shape[:2]
    return np.array(
        [(lm.x * w, lm.y * h) for lm in face_landmarks.landmark],
        dtype=np.float32,
    )


# ── EAR ───────────────────────────────────────────────────────────────────────

def eye_aspect_ratio(eye):
    """
    Eye Aspect Ratio (Soukupova & Cech, 2016).
    eye: np.ndarray of shape (6, 2)
    Returns float — higher = open, lower = closing.
    """
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return float((A + B) / (2.0 * C))


# ── MAR ───────────────────────────────────────────────────────────────────────

def mouth_aspect_ratio(mouth):
    """
    Mouth Aspect Ratio — detects yawning.
    mouth: np.ndarray of shape (8, 2)
    Returns float — higher = mouth more open.
    """
    A = np.linalg.norm(mouth[3] - mouth[7])
    B = np.linalg.norm(mouth[2] - mouth[6])
    C = np.linalg.norm(mouth[0] - mouth[4])
    return float((A + B) / (2.0 * C))


# ── Combined feature extraction ───────────────────────────────────────────────

def compute_features(landmarks):
    """
    Given a full (468, 2) landmark array, return (ear, mar).
    Raises ValueError on bad input instead of silently passing.
    """
    try:
        left_eye  = landmarks[LEFT_EYE_IDX]
        right_eye = landmarks[RIGHT_EYE_IDX]
        mouth     = landmarks[MOUTH_IDX]
    except IndexError as e:
        raise ValueError(f"Landmark array too small: {e}") from e

    ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
    mar = mouth_aspect_ratio(mouth)
    return ear, mar
