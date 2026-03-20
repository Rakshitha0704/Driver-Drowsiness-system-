# =============================================================================
# config.py
# Central configuration for the drowsiness detection project.
# Change values here — nowhere else.
# =============================================================================

# ── MediaPipe landmark indices ────────────────────────────────────────────────
# These are fixed by the MediaPipe 468-point face mesh spec.
# Do not change unless MediaPipe releases a new model.

LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
MOUTH_IDX     = [78, 308, 14, 13, 312, 82, 87, 317]


# ── Detection thresholds ──────────────────────────────────────────────────────
# EAR: Eye Aspect Ratio — below this = eyes closing
# MAR: Mouth Aspect Ratio — above this = yawning
# CONSEC_FRAMES: how many consecutive frames before alert fires

EAR_THRESH    = 0.25
MAR_THRESH    = 0.70
CONSEC_FRAMES = 20


# ── MediaPipe FaceMesh settings ───────────────────────────────────────────────
MAX_NUM_FACES          = 1
DETECTION_CONFIDENCE   = 0.5
TRACKING_CONFIDENCE    = 0.5


# ── Data collection ───────────────────────────────────────────────────────────
# Where the CSV will be saved when you run collect_data.py
DATA_CSV_PATH = "drowsiness_data.csv"

# Labels used in the CSV
LABEL_ALERT  = 0
LABEL_DROWSY = 1


# ── PyTorch model ─────────────────────────────────────────────────────────────
# Input features: [EAR, MAR]  →  2 input neurons
INPUT_SIZE   = 2
HIDDEN_SIZE  = 16       # small — our feature space is tiny
OUTPUT_SIZE  = 2        # binary: alert vs drowsy

LEARNING_RATE = 0.001
BATCH_SIZE    = 32
NUM_EPOCHS    = 100

# Train / val / test split ratios (must sum to 1.0)
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# Where the trained model weights will be saved
MODEL_SAVE_PATH = "drowsiness_model.pt"


# ── Display ───────────────────────────────────────────────────────────────────
WINDOW_NAME = "Driver Drowsiness Detection"
FONT        = 0   # cv2.FONT_HERSHEY_SIMPLEX

# Colours (BGR)
COLOR_GREEN  = (0, 255, 0)
COLOR_RED    = (0, 0, 255)
COLOR_WHITE  = (255, 255, 255)
COLOR_YELLOW = (0, 255, 255)
