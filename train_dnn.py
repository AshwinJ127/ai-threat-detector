# train_dnn.py
import os
import time
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset, WeightedRandomSampler
from torch.optim import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight

DATA_DIR = "data/processed"
RESULTS_DIR = "results"
MODELS_DIR = "models"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

BATCH_SIZE = 1024
EPOCHS = 50
LR = 1e-4
WEIGHT_DECAY = 1e-5
HIDDEN_SIZES = [512, 256, 128]
DROPOUT = 0.3
PRINT_EVERY_BATCHES = 200
PATIENCE = 7
USE_FOCAL_LOSS = False
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "dnn_best.pth")

def now_ts():
    return time.strftime("%Y%m%d_%H%M%S")

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, ignore_index=-100):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    def forward(self, input, target):
        logpt = -self.ce(input, target)
        pt = torch.exp(logpt)
        loss = ((1 - pt) ** self.gamma) * (-logpt)
        return loss.mean()

class TabularDNN(nn.Module):
    def __init__(self, in_features, hidden_sizes=[512,256,128], dropout=0.3, num_classes=2):
        super().__init__()
        layers = []
        prev = in_features
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

print("Loading processed data from:", DATA_DIR)
X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train_scaled.csv"))
X_test  = pd.read_csv(os.path.join(DATA_DIR, "X_test_scaled.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv"))["Label"]
y_test  = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv"))["Label"]

VALID_FRAC = 0.1
np.random.seed(42)
train_idx = np.arange(len(X_train))
np.random.shuffle(train_idx)
val_size = int(len(train_idx) * VALID_FRAC)
val_idx = train_idx[:val_size]
train_idx2 = train_idx[val_size:]

X_val = X_train.iloc[val_idx].reset_index(drop=True)
y_val = y_train.iloc[val_idx].reset_index(drop=True)
X_train_final = X_train.iloc[train_idx2].reset_index(drop=True)
y_train_final = y_train.iloc[train_idx2].reset_index(drop=True)

# Combine all labels as strings to avoid unseen label issues
all_labels = pd.concat([y_train_final, y_val, y_test]).astype(str)

# Always fit LabelEncoder on all labels
label_encoder = LabelEncoder()
label_encoder.fit(all_labels)
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# Transform labels consistently as strings
y_train_encoded = label_encoder.transform(y_train_final.astype(str))
y_val_encoded   = label_encoder.transform(y_val.astype(str))
y_test_encoded  = label_encoder.transform(y_test.astype(str))
NUM_CLASSES = len(label_encoder.classes_)

X_tr_np = X_train_final.values.astype(np.float32)
X_val_np = X_val.values.astype(np.float32)
X_test_np = X_test.values.astype(np.float32)

device = "cuda" if torch.cuda.is_available() else "cpu"
class_counts = Counter(y_train_encoded)
class_sample_weight = compute_class_weight(class_weight="balanced",
                                           classes=np.arange(NUM_CLASSES),
                                           y=y_train_encoded)
sample_weights = np.array([class_sample_weight[c] for c in y_train_encoded])
weighted_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

# Use np.int64 instead of deprecated np.long
train_dataset = TensorDataset(torch.from_numpy(X_tr_np), torch.from_numpy(y_train_encoded.astype(np.int64)))
val_dataset   = TensorDataset(torch.from_numpy(X_val_np), torch.from_numpy(y_val_encoded.astype(np.int64)))
test_dataset  = TensorDataset(torch.from_numpy(X_test_np), torch.from_numpy(y_test_encoded.astype(np.int64)))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=weighted_sampler, drop_last=False, pin_memory=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

model = TabularDNN(in_features=X_tr_np.shape[1],
                   hidden_sizes=HIDDEN_SIZES,
                   dropout=DROPOUT,
                   num_classes=NUM_CLASSES).to(device)

class_weights_tensor = torch.tensor(class_sample_weight, dtype=torch.float32).to(device)
criterion = FocalLoss(gamma=2.0, weight=class_weights_tensor) if USE_FOCAL_LOSS else nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

best_val_recall = 0.0
epochs_no_improve = 0
history = {"train_loss": [], "val_loss": [], "val_recall": [], "val_f1": []}
start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    n_batches = 0
    t0 = time.time()

    for batch_idx, (bx, by) in enumerate(train_loader, 1):
        bx = bx.to(device)
        by = by.to(device)
        optimizer.zero_grad()
        logits = model(bx)
        loss = criterion(logits, by)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        n_batches += 1
        if batch_idx % PRINT_EVERY_BATCHES == 0:
            avg_so_far = running_loss / n_batches
            print(f"Epoch {epoch}/{EPOCHS} | Batch {batch_idx}/{len(train_loader)} | Avg Loss: {avg_so_far:.6f}")

    train_loss = running_loss / max(1, n_batches)
    model.eval()
    all_preds, all_labels = [], []
    val_loss = 0.0
    with torch.no_grad():
        for bx, by in val_loader:
            bx = bx.to(device)
            by = by.to(device)
            logits = model(bx)
            loss = criterion(logits, by)
            val_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(by.cpu().numpy())

    val_loss /= max(1, len(val_loader))
    val_recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_recall"].append(val_recall)
    history["val_f1"].append(val_f1)

    scheduler.step(val_recall)
    epoch_time = time.time() - t0
    print(f"--- Epoch {epoch} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val Recall: {val_recall:.4f} | Val F1: {val_f1:.4f} | Time: {epoch_time:.1f}s ---")

    if val_recall > best_val_recall + 1e-6:
        best_val_recall = val_recall
        epochs_no_improve = 0
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_recall": val_recall,
            "label_encoder": label_encoder
        }, BEST_MODEL_PATH)
        print(f"*** New best model saved (val_recall={val_recall:.4f}) -> {BEST_MODEL_PATH}")
    else:
        epochs_no_improve += 1
        print(f"No improvement for {epochs_no_improve} epoch(s). Best val_recall: {best_val_recall:.4f}")

    if epochs_no_improve >= PATIENCE:
        print(f"Early stopping triggered after {epochs_no_improve} epochs.")
        break

total_time = time.time() - start_time
print(f"\nTraining finished in {total_time/60:.2f} minutes. Best val_recall: {best_val_recall:.4f}")

# Load best model for testing
checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for bx, by in test_loader:
        bx = bx.to(device)
        by = by.to(device)
        logits = model(bx)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(by.cpu().numpy())

acc = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)
f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

print("\n" + "="*60)
print("TEST SET RESULTS")
print(f"Accuracy: {acc:.4f}")
print(f"Precision (macro): {prec:.4f}")
print(f"Recall (macro): {rec:.4f}")
print(f"F1 (macro): {f1:.4f}")
print("="*60)

class_names = list(label_encoder.classes_)
report = classification_report(all_labels, all_preds, target_names=class_names, digits=4, zero_division=0)
print("\nClassification Report:\n", report)

p, r, f, s = precision_recall_fscore_support(all_labels, all_preds, average=None, labels=range(NUM_CLASSES), zero_division=0)
df_perf = pd.DataFrame({"Class": class_names, "Precision": p, "Recall": r, "F1-Score": f, "Support": s})
perf_csv = os.path.join(RESULTS_DIR, f"dnn_per_class_performance_{now_ts()}.csv")
df_perf.to_csv(perf_csv, index=False)
print("Saved per-class performance to:", perf_csv)

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(12, 10))
plt.title("DNN - Confusion Matrix")
plt.imshow(cm, interpolation="nearest", aspect="auto")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
cm_img_path = os.path.join(RESULTS_DIR, f"dnn_confusion_matrix_{now_ts()}.png")
plt.savefig(cm_img_path, dpi=300, bbox_inches="tight")
plt.close()
print("Saved confusion matrix to:", cm_img_path)

plt.figure()
plt.plot(history["train_loss"], label="train_loss")
plt.plot(history["val_loss"], label="val_loss")
plt.legend()
plt.title("Loss curves")
plt.savefig(os.path.join(RESULTS_DIR, f"dnn_loss_{now_ts()}.png"))
plt.close()

plt.figure()
plt.plot(history["val_recall"], label="val_recall")
plt.plot(history["val_f1"], label="val_f1")
plt.legend()
plt.title("Val recall / F1")
plt.savefig(os.path.join(RESULTS_DIR, f"dnn_val_metrics_{now_ts()}.png"))
plt.close()

final_model_path = os.path.join(MODELS_DIR, f"dnn_model_{now_ts()}.pth")
torch.save({
    "model_state_dict": model.state_dict(),
    "label_encoder": label_encoder,
    "config": {"hidden_sizes": HIDDEN_SIZES, "dropout": DROPOUT, "batch_size": BATCH_SIZE, "lr": LR}
}, final_model_path)
print("Saved final model to:", final_model_path)
print("\nAll done. Review results in 'results/' and model in 'models/'.")
