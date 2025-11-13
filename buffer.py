import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm
from pathlib import Path

# ==========================================================
# CONFIGURATION
# ==========================================================
np.random.seed(42)
torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔧 Device: {device}")

# Paths
ROOT = Path("Amazon_products")
TRAIN_CORPUS_PATH = ROOT / "train" / "train_corpus.txt"
TEST_CORPUS_PATH  = ROOT / "test" / "test_corpus.txt"
CLASS_PATH        = ROOT / "classes.txt"

EMB_DIR          = Path("Embeddings")
X_ALL_PATH       = EMB_DIR / "X_train_test.pt"
LABEL_EMB_PATH   = EMB_DIR / "labels_true.pt"

MODEL_SAVE = Path("Models")
MODEL_SAVE.mkdir(exist_ok=True)
MODEL_PATH = MODEL_SAVE / "silver_classifier.pt"

# ==========================================================
# LOAD IDS
# ==========================================================
def load_ids(path):
    ids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            pid, _ = line.strip().split("\t", 1)
            ids.append(int(pid))
    return ids

train_ids = load_ids(TRAIN_CORPUS_PATH)
test_ids  = load_ids(TEST_CORPUS_PATH)
n_train = len(train_ids)
n_test  = len(test_ids)

print(f"Train IDs: {n_train} | Test IDs: {n_test}")

# ==========================================================
# LOAD SILVER LABELS
# ==========================================================
with open("Silver/silver_train_true.json", "r", encoding="utf-8") as f:
    raw = json.load(f)

silver_labels = {int(pid): data["labels"] for pid, data in raw.items()}

# ==========================================================
# LOAD X_all
# ==========================================================
print("\n🧠 Loading X_all.pt ...")
data = torch.load(X_ALL_PATH, weights_only=False)

if isinstance(data, np.ndarray):
    data = torch.from_numpy(data)
elif isinstance(data, list):
    data = torch.stack(data)

X_all = data.float().to(device)
assert X_all.shape[0] == n_train + n_test

X_train = X_all[:n_train]
X_test  = X_all[n_train:]
print(f"✓ X_train: {X_train.shape} | X_test: {X_test.shape}")

# ==========================================================
# LOAD LABEL EMBEDDINGS
# ==========================================================
tmp = torch.load(LABEL_EMB_PATH, weights_only=False)

# Convertir numpy → tensor si nécessaire
if isinstance(tmp, np.ndarray):
    tmp = torch.from_numpy(tmp)

label_emb = tmp.float().to(device)
print(f"✓ Label embeddings: {label_emb.shape}")

# ==========================================================
# LOAD CLASS NAMES
# ==========================================================
classes = {}
with open(CLASS_PATH, "r", encoding="utf-8") as f:
    for line in f:
        cid, cname = line.strip().split("\t")
        classes[int(cid)] = cname

n_classes = len(classes)

# ==========================================================
# DATASET
# ==========================================================
pid2idx = {pid: i for i, pid in enumerate(train_ids)}

class MultiLabelDataset(Dataset):
    def __init__(self, pids, labels_dict):
        self.pids = pids
        self.labels = labels_dict

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, idx):
        pid = self.pids[idx]
        emb = X_train[pid2idx[pid]]

        y = torch.zeros(n_classes)
        for c in self.labels[pid]:
            if 0 <= c < n_classes:
                y[c] = 1.0

        return {"X": emb, "y": y}

# SPLIT
train_p, val_p = train_test_split(
    list(silver_labels.keys()), test_size=0.2, random_state=42
)

train_dataset = MultiLabelDataset(train_p, silver_labels)
val_dataset   = MultiLabelDataset(val_p, silver_labels)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=64)

# ==========================================================
# MODEL : InnerProductClassifier
# ==========================================================
class InnerProductClassifier(nn.Module):
    def __init__(self, input_dim, label_embeddings, dropout=0.2, trainable_label_emb=False):
        super().__init__()

        D = label_embeddings.size(1)

        self.proj = nn.Linear(input_dim, D)
        self.dropout = nn.Dropout(dropout)

        if trainable_label_emb:
            self.label_emb = nn.Parameter(label_embeddings.clone())
        else:
            self.register_buffer("label_emb", label_embeddings.clone())

    def forward(self, x, use_dropout=True):
        if use_dropout:
            x = self.dropout(x)

        x_proj = self.proj(x)                 # (B, D)
        logits = x_proj @ self.label_emb.T    # (B, C)

        return logits

model = InnerProductClassifier(
    input_dim = X_train.size(1),
    label_embeddings = label_emb,
    dropout = 0.2,
    trainable_label_emb = False
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

# ==========================================================
# EVAL
# ==========================================================
def evaluate(model, loader, thr=0.25):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for batch in loader:
            X = batch["X"]
            y = batch["y"].numpy()

            prob = torch.sigmoid(model(X)).cpu().numpy()
            pred = (prob > thr).astype(int)

            preds.extend(pred)
            labels.extend(y)

    f1s = f1_score(labels, preds, average="samples")
    f1m = f1_score(labels, preds, average="macro")
    return f1s, f1m

# ==========================================================
# TRAIN LOOP
# ==========================================================
print("\n🚀 Training...")
best = 0
epochs = 25

for epoch in range(1, epochs+1):
    model.train()
    total = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
        X = batch["X"]
        y = batch["y"].to(device)

        logits = model(X)
        loss = F.binary_cross_entropy_with_logits(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += loss.item()

    f1s, f1m = evaluate(model, val_loader)
    print(f"[Epoch {epoch}] loss={total/len(train_loader):.4f} | F1={f1s:.4f}")

    if f1s > best:
        best = f1s
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"🔥 New best model saved ({best:.4f})")

print(f"\n🎉 Best validation F1 = {best:.4f}")
print(f"📦 Model saved at: {MODEL_PATH}")

import csv

print("\n📝 Generating submission...")

# Reload best model
best_model = InnerProductClassifier(
    input_dim=X_train.size(1),
    label_embeddings=label_emb,
    dropout=0.2,
    trainable_label_emb=False
).to(device)

best_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
best_model.eval()

# Parameters from your ORIGINAL code
THR = 0.5
MIN_L = 2
MAX_L = 3

X_test = X_test.to(device)

preds = []

with torch.no_grad():
    for start in tqdm(range(0, len(X_test), 64)):
        batch = X_test[start:start+64]

        # disable dropout
        logits = best_model(batch, use_dropout=False)

        probs = torch.sigmoid(logits).cpu().numpy()

        for p in probs:
            pred = (p > THR).astype(int)

            # ===== YOUR POST-PROCESSING RULES =====
            if pred.sum() == 0:
                pred[np.argsort(p)[-MIN_L:]] = 1
            elif pred.sum() == 1:
                pred[np.argsort(p)[-2:]] = 1
            elif pred.sum() > MAX_L:
                pred = np.zeros_like(pred)
                pred[np.argsort(p)[-MAX_L:]] = 1

            labels = [str(i) for i, v in enumerate(pred) if v == 1]
            preds.append(labels)

# ==========================================================
# SAVE CSV
# ==========================================================

OUT_DIR = Path("Submission")
OUT_DIR.mkdir(exist_ok=True)
OUT_PATH = OUT_DIR / "submission.csv"

with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["id", "label"])
    for pid, labels in zip(test_ids, preds):
        w.writerow([pid, ",".join(labels)])

print(f"🎉 Submission saved → {OUT_PATH}")