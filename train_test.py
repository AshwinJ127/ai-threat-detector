import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -------------------------------
# 1. Load preprocessed dataset
# -------------------------------
df = pd.read_csv("data/combined_dataset.csv")

# Clean up column names and labels
df.columns = df.columns.str.strip()
df['is_attack'] = (df['Label'] != 'BENIGN').astype(int)

# Drop non-feature columns
feature_cols = [c for c in df.columns if c not in ['Label', 'day', 'source_file', 'is_attack']]
X = df[feature_cols].astype(np.float32)

# Replace bad values
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(0, inplace=True)
X = X.clip(-1e6, 1e6)

y = df['is_attack'].values

# Standardize numeric features
scaler = StandardScaler()
X[X.columns] = scaler.fit_transform(X)

# -------------------------------
# 2. Train/test split by day
# -------------------------------
train_mask = df['day'].isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday'])
X_train, X_test = X[train_mask], X[~train_mask]
y_train, y_test = y[train_mask], y[~train_mask]

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# -------------------------------
# 3. Dataset class
# -------------------------------
SEQ_LEN = 5

class FlowSequenceDataset(Dataset):
    def __init__(self, X, y, seq_len=SEQ_LEN):
        self.X = X.values
        self.y = y
        self.seq_len = seq_len
        self.n_sequences = len(X) - seq_len + 1

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, idx):
        seq_x = self.X[idx:idx+self.seq_len]
        seq_y = self.y[idx:idx+self.seq_len]
        label = 1 if seq_y.sum() > 0 else 0
        return torch.tensor(seq_x, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

train_dataset = FlowSequenceDataset(X_train, y_train)
test_dataset = FlowSequenceDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# -------------------------------
# 4. Transformer model
# -------------------------------
class TransformerClassifier(nn.Module):
    def __init__(self, num_features, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(num_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = x.permute(1, 0, 2)  # seq_len, batch, d_model
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)
        x = self.classifier(x)
        return x  # logits (no sigmoid)

num_features = X_train.shape[1]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = TransformerClassifier(num_features).to(DEVICE)

# -------------------------------
# 5. Weighted loss and optimizer
# -------------------------------
# Compute positive class weight to handle imbalance
pos_weight_value = (len(y_train) - y_train.sum()) / y_train.sum()
pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32).to(DEVICE)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

# -------------------------------
# 6. Training loop
# -------------------------------
EPOCHS = 10
PRINT_EVERY = 5000

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch_idx, (seq_x, seq_y) in enumerate(train_loader, 1):
        seq_x, seq_y = seq_x.to(DEVICE), seq_y.to(DEVICE).unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(seq_x)
        loss = criterion(outputs, seq_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % PRINT_EVERY == 0:
            avg_batch_loss = total_loss / batch_idx
            print(f"Epoch {epoch+1}/{EPOCHS} | Batch {batch_idx}/{len(train_loader)} | Avg Loss: {avg_batch_loss:.6f}")

    avg_loss = total_loss / len(train_loader)
    print(f"--- Epoch {epoch+1} completed | Avg Loss: {avg_loss:.6f} ---")

    # Evaluate each epoch
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for seq_x, seq_y in test_loader:
            seq_x = seq_x.to(DEVICE)
            logits = model(seq_x)
            preds = torch.sigmoid(logits)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(seq_y.numpy())

    # Use lower threshold to improve recall
    threshold = 0.3
    pred_labels = [1 if p >= threshold else 0 for p in all_preds]

    acc = accuracy_score(all_labels, pred_labels)
    prec = precision_score(all_labels, pred_labels)
    rec = recall_score(all_labels, pred_labels)
    f1 = f1_score(all_labels, pred_labels)

    print(f"[Epoch {epoch+1}] Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

# -------------------------------
# 7. Final evaluation summary
# -------------------------------
print("\nFinal model evaluation complete.")
