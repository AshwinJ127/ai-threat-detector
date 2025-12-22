# train_dnn2.py (improved for overfitting control & unique model saving)
import os
import time
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.optim import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import recall_score, f1_score, accuracy_score, precision_score
from sklearn.utils.class_weight import compute_class_weight

# ------------------------------
# Directories
# ------------------------------
DATA_DIR = "data/processed"
RESULTS_DIR = "results"
MODELS_DIR = "models"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ------------------------------
# Training configuration
# ------------------------------
BATCH_SIZE = 1024
EPOCHS = 50
LR = 1e-4
WEIGHT_DECAY = 1e-4
HIDDEN_SIZES = [256, 128, 64]
DROPOUT = 0.5
PRINT_EVERY_BATCHES = 200
PATIENCE = 5
USE_FOCAL_LOSS = True

def now_ts():
    return time.strftime("%Y%m%d_%H%M%S")

BEST_MODEL_PATH = os.path.join(MODELS_DIR, f"dnn2_best_{now_ts()}.pth")

# ------------------------------
# Focal Loss
# ------------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, ignore_index=-100):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    def forward(self, input, target):
        logpt = -self.ce(input, target)
        pt = torch.exp(logpt)
        loss = ((1 - pt) ** self.gamma) * (-logpt)
        return loss.mean()

# ------------------------------
# DNN Model
# ------------------------------
class TabularDNN(nn.Module):
    def __init__(self, in_features, hidden_sizes=[256,128,64], dropout=0.5, num_classes=2):
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

# ------------------------------
# Load data
# ------------------------------
print("Loading processed data from:", DATA_DIR)
X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train_scaled.csv"))
X_test  = pd.read_csv(os.path.join(DATA_DIR, "X_test_scaled.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv"))["Label"]
y_test  = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv"))["Label"]

# Train/validation split
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

# Label encoding
all_labels = pd.concat([y_train_final, y_val, y_test]).astype(str)
label_encoder = LabelEncoder()
label_encoder.fit(all_labels)
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

y_train_encoded = label_encoder.transform(y_train_final.astype(str))
y_val_encoded   = label_encoder.transform(y_val.astype(str))
y_test_encoded  = label_encoder.transform(y_test.astype(str))
NUM_CLASSES = len(label_encoder.classes_)

X_tr_np = X_train_final.values.astype(np.float32)
X_val_np = X_val.values.astype(np.float32)
X_test_np = X_test.values.astype(np.float32)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------
# Class balancing
# ------------------------------
class_sample_weight = compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=y_train_encoded)
sample_weights = np.array([class_sample_weight[c] for c in y_train_encoded])
weighted_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

# ------------------------------
# Datasets and loaders
# ------------------------------
train_dataset = TensorDataset(torch.from_numpy(X_tr_np), torch.from_numpy(y_train_encoded.astype(np.int64)))
val_dataset   = TensorDataset(torch.from_numpy(X_val_np), torch.from_numpy(y_val_encoded.astype(np.int64)))
test_dataset  = TensorDataset(torch.from_numpy(X_test_np), torch.from_numpy(y_test_encoded.astype(np.int64)))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=weighted_sampler, drop_last=False, pin_memory=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

# ------------------------------
# Model, loss, optimizer
# ------------------------------
model = TabularDNN(in_features=X_tr_np.shape[1], hidden_sizes=HIDDEN_SIZES, dropout=DROPOUT, num_classes=NUM_CLASSES).to(device)
class_weights_tensor = torch.tensor(class_sample_weight, dtype=torch.float32).to(device)
criterion = FocalLoss(gamma=2.0, weight=class_weights_tensor) if USE_FOCAL_LOSS else nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

# ------------------------------
# Training loop
# ------------------------------
best_val_recall = 0.0
epochs_no_improve = 0
history = {"train_loss": [], "val_loss": [], "val_recall": [], "val_f1": []}

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    n_batches = 0
    t0 = time.time()

    for batch_idx, (bx, by) in enumerate(train_loader, 1):
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        logits = model(bx)
        loss = criterion(logits, by)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        n_batches += 1
        if batch_idx % PRINT_EVERY_BATCHES == 0:
            print(f"Epoch {epoch}/{EPOCHS} | Batch {batch_idx}/{len(train_loader)} | Avg Loss: {running_loss/n_batches:.6f}")

    train_loss = running_loss / max(1, n_batches)

    # Validation
    model.eval()
    all_preds, all_labels = [], []
    val_loss = 0.0
    with torch.no_grad():
        for bx, by in val_loader:
            bx, by = bx.to(device), by.to(device)
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
    print(f"--- Epoch {epoch} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val Recall: {val_recall:.4f} | Val F1: {val_f1:.4f} | Time: {time.time()-t0:.1f}s ---")

    # Early stopping & best model
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

# ------------------------------
# Test evaluation
# ------------------------------
checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for bx, by in test_loader:
        bx, by = bx.to(device), by.to(device)
        logits = model(bx)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(by.cpu().numpy())

acc = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)
f1 = f1_score(all_preds, all_preds, average="macro", zero_division=0)

print("\nTEST SET RESULTS")
print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
