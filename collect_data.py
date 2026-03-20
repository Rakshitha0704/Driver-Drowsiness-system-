# =============================================================================
# collect_data.py
#
# HOW TO USE:
#   1. Run this script in Spyder (press F5)
#   2. A webcam window will open
#   3. Press 'a' to START recording ALERT frames (press again to stop)
#      Press 'd' to START recording DROWSY frames (press again to stop)
#      Press 'q' to quit and save
#
# TIP: Press 'a' once → it auto-logs every frame until you press 'a' again.
#      Same for 'd'. You just toggle recording on/off.
#
# AIM: 300+ alert + 300+ drowsy = 600+ rows total
# =============================================================================

import csv
import os
import cv2
import mediapipe as mp

from config import DATA_CSV_PATH, LABEL_ALERT, LABEL_DROWSY, WINDOW_NAME
from detector import build_face_mesh, extract_landmarks, compute_features

# ── Setup ─────────────────────────────────────────────────────────────────────

face_mesh  = build_face_mesh()
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Check camera index.")

# ── CSV setup ─────────────────────────────────────────────────────────────────

file_exists = os.path.isfile(DATA_CSV_PATH)
csv_file    = open(DATA_CSV_PATH, "a", newline="")
writer      = csv.writer(csv_file)

if not file_exists:
    writer.writerow(["ear", "mar", "label"])

# ── State ─────────────────────────────────────────────────────────────────────

count_alert  = 0
count_drowsy = 0
recording    = None   # None = idle, 'alert' = recording alert, 'drowsy' = recording drowsy

print("=" * 50)
print("Data collection started.")
print("  Press 'a' -> toggle ALERT recording on/off")
print("  Press 'd' -> toggle DROWSY recording on/off")
print("  Press 'q' -> quit and save")
print("=" * 50)

# ── Main loop ─────────────────────────────────────────────────────────────────

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame     = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results   = face_mesh.process(rgb_frame)

    ear, mar = None, None

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
        except ValueError:
            pass

    # ── Auto-log every frame if recording is active ───────────────────────────

    if ear is not None and recording == 'alert':
        writer.writerow([f"{ear:.6f}", f"{mar:.6f}", LABEL_ALERT])
        count_alert += 1

    elif ear is not None and recording == 'drowsy':
        writer.writerow([f"{ear:.6f}", f"{mar:.6f}", LABEL_DROWSY])
        count_drowsy += 1

    # ── HUD ───────────────────────────────────────────────────────────────────

    if ear is not None:
        cv2.putText(frame, f"EAR: {ear:.3f}  MAR: {mar:.3f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Recording status indicator
    if recording == 'alert':
        status_text  = "RECORDING: ALERT"
        status_color = (0, 255, 0)       # green
    elif recording == 'drowsy':
        status_text  = "RECORDING: DROWSY"
        status_color = (0, 0, 255)       # red
    else:
        status_text  = "IDLE - press 'a' or 'd'"
        status_color = (200, 200, 200)   # gray

    cv2.putText(frame, status_text,
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    cv2.putText(frame, f"Alert: {count_alert}  Drowsy: {count_drowsy}",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.putText(frame, "'a'=alert  'd'=drowsy  'q'=quit",
                (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (200, 200, 200), 1)

    cv2.imshow(WINDOW_NAME, frame)

    # ── Key handling ──────────────────────────────────────────────────────────

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    elif key == ord("a"):
        if recording == 'alert':
            recording = None
            print(f"  [ALERT recording STOPPED] Total alert={count_alert}")
        else:
            recording = 'alert'
            print(f"  [ALERT recording STARTED] Sit normally, eyes open...")

    elif key == ord("d"):
        if recording == 'drowsy':
            recording = None
            print(f"  [DROWSY recording STOPPED] Total drowsy={count_drowsy}")
        else:
            recording = 'drowsy'
            print(f"  [DROWSY recording STARTED] Close eyes halfway, yawn...")

# ── Cleanup ───────────────────────────────────────────────────────────────────

cap.release()
cv2.destroyAllWindows()
csv_file.close()
face_mesh.close()

print("\n" + "=" * 50)
print("Collection done.")
print(f"  Alert samples  : {count_alert}")
print(f"  Drowsy samples : {count_drowsy}")
print(f"  Total          : {count_alert + count_drowsy}")
print(f"  Saved to       : {DATA_CSV_PATH}")
print("=" * 50)
