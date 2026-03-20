import time
import torch
import torch.nn as nn
import numpy as np
import csv

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix, classification_report
)
from config import (
    DATA_CSV_PATH, MODEL_SAVE_PATH,
    INPUT_SIZE, OUTPUT_SIZE,
    LEARNING_RATE, BATCH_SIZE,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
)

HIDDEN_SIZE = 32
NUM_EPOCHS  = 300

# ── 1. Load CSV ───────────────────────────────────────────────────────────────

print("=" * 55)
print("DROWSINESS DETECTION — MODEL TRAINING v2")
print("=" * 55)

print("\n[1/5] Loading dataset...")

ears, mars, labels = [], [], []

with open(DATA_CSV_PATH, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        ears.append(float(row["ear"]))
        mars.append(float(row["mar"]))
        labels.append(int(row["label"]))

X = np.column_stack([ears, mars]).astype(np.float32)
y = np.array(labels, dtype=np.int64)

print(f"  Total samples  : {len(y)}")
print(f"  Alert  (0)     : {(y == 0).sum()}")
print(f"  Drowsy (1)     : {(y == 1).sum()}")
print(f"  EAR range      : {X[:,0].min():.3f} – {X[:,0].max():.3f}")
print(f"  MAR range      : {X[:,1].min():.3f} – {X[:,1].max():.3f}")


# ── 2. Split ──────────────────────────────────────────────────────────────────

print("\n[2/5] Splitting dataset...")

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=TEST_RATIO, random_state=42, stratify=y
)
val_size = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=val_size, random_state=42, stratify=y_temp
)

print(f"  Train : {len(y_train)} samples")
print(f"  Val   : {len(y_val)} samples")
print(f"  Test  : {len(y_test)} samples")

# ── Normalize ─────────────────────────────────────────────────────────────────
# Fit ONLY on training data, apply to val and test

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

print(f"\n  Scaler mean : EAR={scaler.mean_[0]:.4f}  MAR={scaler.mean_[1]:.4f}")
print(f"  Scaler std  : EAR={scaler.scale_[0]:.4f}  MAR={scaler.scale_[1]:.4f}")

# Convert to tensors
X_train_t = torch.tensor(X_train.astype(np.float32))
y_train_t = torch.tensor(y_train)
X_val_t   = torch.tensor(X_val.astype(np.float32))
y_val_t   = torch.tensor(y_val)
X_test_t  = torch.tensor(X_test.astype(np.float32))
y_test_t  = torch.tensor(y_test)


# ── 3. Define model ───────────────────────────────────────────────────────────

print("\n[3/5] Building model...")

class DrowsinessNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE),
        )

    def forward(self, x):
        return self.network(x)


model     = DrowsinessNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

total_params = sum(p.numel() for p in model.parameters())
print(f"  Architecture   : {INPUT_SIZE} -> {HIDDEN_SIZE} -> {HIDDEN_SIZE} -> {OUTPUT_SIZE}")
print(f"  Total params   : {total_params}")
print(f"  Epochs         : {NUM_EPOCHS}  Batch size: {BATCH_SIZE}")


# ── 4. Training loop ──────────────────────────────────────────────────────────

print("\n[4/5] Training...")
print(f"  {'Epoch':>6}  {'Train Loss':>10}  {'Val Loss':>10}  {'Val Acc':>8}")
print("  " + "-" * 42)

train_start      = time.perf_counter()
best_val_loss    = float("inf")
best_model_state = None

for epoch in range(1, NUM_EPOCHS + 1):

    model.train()
    indices     = torch.randperm(len(X_train_t))
    epoch_loss  = 0.0
    num_batches = 0

    for i in range(0, len(X_train_t), BATCH_SIZE):
        batch_idx = indices[i : i + BATCH_SIZE]
        xb = X_train_t[batch_idx]
        yb = y_train_t[batch_idx]

        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()

        epoch_loss  += loss.item()
        num_batches += 1

    scheduler.step()
    avg_train_loss = epoch_loss / num_batches

    model.eval()
    with torch.no_grad():
        val_out   = model(X_val_t)
        val_preds = val_out.argmax(dim=1)
        val_loss  = criterion(val_out, y_val_t).item()
        val_acc   = accuracy_score(y_val_t.numpy(), val_preds.numpy())

    if val_loss < best_val_loss:
        best_val_loss    = val_loss
        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

    if epoch % 50 == 0 or epoch == 1:
        print(f"  {epoch:>6}  {avg_train_loss:>10.4f}  {val_loss:>10.4f}  {val_acc:>7.1%}")

train_time = time.perf_counter() - train_start
print(f"\n  Training complete in {train_time:.2f} seconds")


# ── 5. Evaluation ─────────────────────────────────────────────────────────────

print("\n[5/5] Evaluating on held-out test set...")

model.load_state_dict(best_model_state)
model.eval()

# Inference latency
dummy_input  = torch.tensor([[0.3, 0.5]], dtype=torch.float32)
latency_runs = 1000

with torch.no_grad():
    for _ in range(10):
        model(dummy_input)
    lat_start = time.perf_counter()
    for _ in range(latency_runs):
        model(dummy_input)
    lat_end = time.perf_counter()

avg_latency_ms = (lat_end - lat_start) / latency_runs * 1000

with torch.no_grad():
    test_preds = model(X_test_t).argmax(dim=1).numpy()

y_test_np = y_test_t.numpy()
accuracy  = accuracy_score(y_test_np, test_preds)
f1        = f1_score(y_test_np, test_preds, average="weighted")
precision = precision_score(y_test_np, test_preds, average="weighted")
recall    = recall_score(y_test_np, test_preds, average="weighted")
cm        = confusion_matrix(y_test_np, test_preds)
report    = classification_report(y_test_np, test_preds,
                                   target_names=["Alert", "Drowsy"])

print("\n" + "=" * 55)
print("RESULTS — COPY THESE INTO YOUR RESUME")
print("=" * 55)
print(f"  Dataset size       : {len(y)} samples")
print(f"  Train/Val/Test     : {len(y_train)}/{len(y_val)}/{len(y_test)}")
print(f"  Training time      : {train_time:.2f} seconds")
print(f"  Inference latency  : {avg_latency_ms:.3f} ms per frame")
print(f"  Test accuracy      : {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"  Weighted F1 score  : {f1:.4f}")
print(f"  Weighted Precision : {precision:.4f}")
print(f"  Weighted Recall    : {recall:.4f}")
print("\n  Confusion Matrix:")
print(f"              Predicted Alert  Predicted Drowsy")
print(f"  Actual Alert      {cm[0][0]:>5}              {cm[0][1]:>5}")
print(f"  Actual Drowsy     {cm[1][0]:>5}              {cm[1][1]:>5}")
print("\n  Classification Report:")
print(report)
print("=" * 55)

# ── Save model + scaler ───────────────────────────────────────────────────────

torch.save({
    "model_state_dict" : best_model_state,
    "input_size"       : INPUT_SIZE,
    "hidden_size"      : HIDDEN_SIZE,
    "output_size"      : OUTPUT_SIZE,
    "accuracy"         : accuracy,
    "f1_score"         : f1,
    "train_time_sec"   : train_time,
    "latency_ms"       : avg_latency_ms,
    "scaler_mean"      : scaler.mean_.tolist(),
    "scaler_std"       : scaler.scale_.tolist(),
}, MODEL_SAVE_PATH)

print(f"\n  Model saved to : {MODEL_SAVE_PATH}")

with open("metrics.txt", "w") as mf:
    mf.write("DROWSINESS DETECTION — METRICS\n")
    mf.write("=" * 40 + "\n")
    mf.write(f"Dataset size      : {len(y)} samples\n")
    mf.write(f"Training time     : {train_time:.2f} seconds\n")
    mf.write(f"Inference latency : {avg_latency_ms:.3f} ms per frame\n")
    mf.write(f"Test accuracy     : {accuracy*100:.2f}%\n")
    mf.write(f"Weighted F1       : {f1:.4f}\n")
    mf.write(f"Weighted Precision: {precision:.4f}\n")
    mf.write(f"Weighted Recall   : {recall:.4f}\n")
    mf.write("\nConfusion Matrix:\n")
    mf.write(f"  True Alert  -> Alert  : {cm[0][0]}\n")
    mf.write(f"  True Alert  -> Drowsy : {cm[0][1]}\n")
    mf.write(f"  True Drowsy -> Alert  : {cm[1][0]}\n")
    mf.write(f"  True Drowsy -> Drowsy : {cm[1][1]}\n")
    mf.write("\nClassification Report:\n")
    mf.write(report)

print(f"  Metrics saved  : metrics.txt")
print("\nDone! Run live_inference.py next.")
