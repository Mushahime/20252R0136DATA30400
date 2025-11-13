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
CLASS_PATH = ROOT / "classes.txt"
HIERARCHY_PATH = ROOT / "class_hierarchy.txt"

# Output paths
OUTPUT_DIR = Path("Silver")
OUTPUT_DIR.mkdir(exist_ok=True)
SILVER_LABELS_PATH = OUTPUT_DIR / "silver_labels_v2.json"

EMBEDDINGS_DIR = Path("Embeddings")
#EMBEDDINGS_PATH = EMBEDDINGS_DIR / "X_trainMiniLM.pt"
#EMBEDDINGS_PATH = EMBEDDINGS_DIR / "X_trainMiniLM_roberta.pt"
EMBEDDINGS_PATH = EMBEDDINGS_DIR / "X_train_true.pt"
#LABEL_EMB_PATH = EMBEDDINGS_DIR / "label_embMiniLM.pt"
#LABEL_EMB_PATH = EMBEDDINGS_DIR / "label_embMiniLM_roberta.pt"
LABEL_EMB_PATH = EMBEDDINGS_DIR / "label_emb_true.pt"
#X_TEST_PATH = EMBEDDINGS_DIR / "X_testMiniLM.pt"
#X_TEST_PATH = EMBEDDINGS_DIR / "X_testMiniLM_roberta.pt"
X_TEST_PATH = EMBEDDINGS_DIR / "X_test_true.pt"

# ==========================================================
# LOAD DATA
# ==========================================================
print("\n📂 Loading data...")

def load_corpus(path):
    """Load corpus file"""
    id2text = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                pid, text = parts
                id2text[int(pid)] = text
    return id2text

def load_classes(path):
    """Load class names"""
    classes = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                cid, cname = parts
                classes[int(cid)] = cname
    return classes

def load_hierarchy(path):
    """Load hierarchy as parent->children dict"""
    hierarchy = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                parent, child = int(parts[0]), int(parts[1])
                if parent not in hierarchy:
                    hierarchy[parent] = []
                hierarchy[parent].append(child)
    return hierarchy

# Load all data
id2text = load_corpus(TRAIN_CORPUS_PATH)
classes = load_classes(CLASS_PATH)
hierarchy = load_hierarchy(HIERARCHY_PATH)

print(f"✓ Loaded {len(id2text)} documents")
print(f"✓ Loaded {len(classes)} classes")
print(f"✓ Loaded {len(hierarchy)} parent-child relations")



import json

# --- Charger ton fichier JSON ---
with open("Silver/silver_train_roberta.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# --- Récupérer uniquement les listes de labels ---
silver_labels = {k: v["labels"] for k, v in data.items() if "labels" in v}



print("\n🧠 Loading embeddings...")
data = torch.load(EMBEDDINGS_PATH)

if isinstance(data, list):
    print("🧩 Detected list of tensors — stacking them...")
    data = torch.stack(data)  # assemble (N, D)

X_train = data.to(device)

label_emb_data = torch.load(LABEL_EMB_PATH)
if isinstance(label_emb_data, list):
    label_emb_data = torch.stack(label_emb_data)
label_emb = label_emb_data.to(device)

print(f"✓ Train embeddings: {X_train.shape}")
print(f"✓ Label embeddings: {label_emb.shape}")

print("🧠 Loading test embeddings...")

X_data = torch.load(X_TEST_PATH)
if isinstance(X_data, list):
    X_data = torch.stack(X_data)
X_test = X_data.to(device)



# Create pid to index mapping
train_ids = list(id2text.keys())
pid2idx = {pid: i for i, pid in enumerate(train_ids)}
pid2idx = {int(k): v for k, v in pid2idx.items()}


# ==========================================================
# VALIDATE SILVER LABELS WITH GNN
# ==========================================================

import copy

print("\n🧩 Validating silver labels using GCN-enhanced classifier...")

# ----------------------------------------------------------
# Adjacency Matrix Construction
# ----------------------------------------------------------
num_classes = len(classes)
A = torch.zeros((num_classes, num_classes), dtype=torch.float32, device=device)

for parent_id, children_ids in hierarchy.items():
    if parent_id >= num_classes:
        continue
    for child_id in children_ids:
        if child_id >= num_classes:
            continue
        A[parent_id, child_id] = 1.0
        A[child_id, parent_id] = 1.0

# Add self-loops
A = A + torch.eye(num_classes, device=device)

# Degree normalization
deg = A.sum(dim=1)
deg_inv_sqrt = torch.pow(deg, -0.5)
deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
D_inv_sqrt = torch.diag(deg_inv_sqrt)

A_hat = D_inv_sqrt @ A @ D_inv_sqrt

print(f"✓ Adjacency matrix: {A.shape} | Normalized: {A_hat.shape}")

# ----------------------------------------------------------
# Dataset class
# ----------------------------------------------------------
class ProductCategoryDataset(Dataset):
    """Dataset using precomputed embeddings (robuste à int/str mismatches)"""
    def __init__(self, pid2label, pid2idx, embeddings, num_classes):
        self.embeddings = embeddings
        self.num_classes = num_classes

        # 🔧 Conversion des clés en int (sécurisée)
        self.pid2idx = {int(k): v for k, v in pid2idx.items()}
        self.pid2label = (
            {int(k): v for k, v in pid2label.items()} if pid2label is not None else None
        )

        # 🔎 Sélection des IDs valides
        if self.pid2label is not None:
            self.pids = [pid for pid in self.pid2label.keys() if pid in self.pid2idx]
            self.has_labels = True
        else:
            self.pids = list(self.pid2idx.keys())
            self.has_labels = False

        # ⚙️ On garde uniquement les embeddings alignés
        self.indices = [self.pid2idx[pid] for pid in self.pids]

        # 💬 Log utile
        print(f"✓ Dataset ready: {len(self.pids)} samples ({'train' if self.has_labels else 'test'})")

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, idx):
        emb = self.embeddings[self.indices[idx]]
        if self.has_labels:
            y = torch.zeros(self.num_classes, dtype=torch.float32)
            for label in self.pid2label[self.pids[idx]]:
                if 0 <= label < self.num_classes:
                    y[label] = 1.0
            return {"X": emb, "y": y}
        else:
            return {"X": emb}


# ----------------------------------------------------------
# Label GCN + Classifier
# ----------------------------------------------------------
class LabelGCN(nn.Module):
    def __init__(self, emb_dim, num_layers=2, dropout=0.5):
        super().__init__()
        self.W_list = nn.ParameterList([
            nn.Parameter(torch.empty(emb_dim, emb_dim))
            for _ in range(num_layers)
        ])
        for W in self.W_list:
            nn.init.xavier_uniform_(W)
        self.dropout = dropout
        self.num_layers = num_layers

    def forward(self, H, A_hat):
        for i, W in enumerate(self.W_list):
            H = A_hat @ H @ W
            if i < self.num_layers - 1:
                H = F.relu(H)
                H = F.dropout(H, p=self.dropout, training=self.training)
        return H


class GCNEnhancedClassifier(nn.Module):
    def __init__(self, input_dim, label_init_emb, A_hat, num_layers=2, dropout=0.3):
        super().__init__()
        self.proj = nn.Linear(input_dim, label_init_emb.size(1))
        self.encoder = LabelGCN(label_init_emb.size(1), num_layers, dropout)
        self.label_emb = nn.Parameter(label_init_emb.clone())
        self.register_buffer("A_hat", A_hat)
        self.dropout = dropout

    def forward(self, x):
        # Refine label embeddings through GCN
        E_refine = self.encoder(self.label_emb, self.A_hat)
        x_proj = F.dropout(self.proj(x), p=self.dropout, training=self.training)
        logits = torch.matmul(x_proj, E_refine.T)
        return logits

# ----------------------------------------------------------
# Train / Validation split
# ----------------------------------------------------------
train_pids, val_pids = train_test_split(list(silver_labels.keys()), test_size=0.2, random_state=42)
train_labels = {pid: silver_labels[pid] for pid in train_pids}
val_labels = {pid: silver_labels[pid] for pid in val_pids}

train_dataset = ProductCategoryDataset(train_labels, pid2idx, X_train, num_classes)
val_dataset = ProductCategoryDataset(val_labels, pid2idx, X_train, num_classes)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

print(f"Train: {len(train_labels)} | Val: {len(val_labels)}")

# ----------------------------------------------------------
# Model setup
# ----------------------------------------------------------
input_dim = X_train.shape[1]

model = GCNEnhancedClassifier(
    input_dim=input_dim,
    label_init_emb=label_emb,
    A_hat=A_hat,
    num_layers=2,
    dropout=0.3
).to(device)

teacher = copy.deepcopy(model).to(device)
for p in teacher.parameters():
    p.requires_grad = False

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
criterion = nn.BCEWithLogitsLoss()

# ----------------------------------------------------------
# Evaluation
# ----------------------------------------------------------
def evaluate(model, dataloader, threshold=0.5):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            X = batch["X"].to(device)
            y = batch["y"].cpu().numpy()
            logits = model(X)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > threshold).astype(int)
            all_preds.append(preds)
            all_labels.append(y)
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    return {
        "f1_macro": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "f1_micro": f1_score(all_labels, all_preds, average="micro", zero_division=0),
        "f1_samples": f1_score(all_labels, all_preds, average="samples", zero_division=0)
    }

# ----------------------------------------------------------
# Training Loop (with EMA teacher)
# ----------------------------------------------------------
alpha_ema = 0.995
lambda_cons = 0.5
best_val_f1 = -1
best_state = None
patience = 5
patience_counter = 0
EPOCHS = 25

print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

for epoch in range(1, EPOCHS + 1):
    print(f"\n=== Epoch {epoch}/{EPOCHS} ===")
    model.train()
    total_loss = 0.0

    for batch in tqdm(train_loader, desc="Training"):
        X = batch["X"].to(device)
        y = batch["y"].to(device)

        logits_student = model(X)
        loss_sup = criterion(logits_student, y)

        with torch.no_grad():
            logits_teacher = teacher(X)

        loss_cons = F.mse_loss(
            torch.sigmoid(logits_student),
            torch.sigmoid(logits_teacher)
        )
        loss = loss_sup + lambda_cons * loss_cons

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # EMA update
        with torch.no_grad():
            for p_t, p_s in zip(teacher.parameters(), model.parameters()):
                p_t.data = alpha_ema * p_t.data + (1 - alpha_ema) * p_s.data

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    val_metrics = evaluate(teacher, val_loader)
    val_f1 = val_metrics["f1_samples"]

    print(f"Loss: {avg_loss:.4f} | Val F1(s): {val_f1:.4f} | F1(macro): {val_metrics['f1_macro']:.4f}")

    # Save best model
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_state = copy.deepcopy(teacher.state_dict())
        patience_counter = 0
        print("✅ New best model saved.")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("⏹️ Early stopping triggered.")
            break

    scheduler.step()

print("\n" + "="*60)
print(f"🏆 Best Validation F1 (samples): {best_val_f1:.4f}")
print("="*60)

# ==========================================================
# 🧾 FINAL EVALUATION & SUBMISSION (GCN VERSION)
# ==========================================================
print("\n📊 Evaluating best model on validation set...")

# Charger le meilleur modèle (teacher stable)
model.load_state_dict(best_state)
model.eval()

final_result = evaluate(model, val_loader)

print("\n" + "="*60)
print("📊 FINAL VALIDATION RESULTS (GCN)")
print("="*60)
print(f"F1 Macro:   {final_result['f1_macro']:.4f}")
print(f"F1 Micro:   {final_result['f1_micro']:.4f}")
print(f"F1 Samples: {final_result['f1_samples']:.4f}")
print(f"Coverage:   {len(silver_labels)}/{len(train_ids)} ({len(silver_labels)/len(train_ids):.1%})")
print("="*60)

best_f1 = final_result['f1_samples']
if best_f1 > 0.25:
    print("✅ Silver labels are of GOOD quality – ready for full supervised training!")
elif best_f1 > 0.15:
    print("⚠️ Silver labels are of MEDIUM quality – may need minor tuning.")
else:
    print("❌ Silver labels are noisy – recheck similarity thresholds or hierarchy injection.")

print(f"\n💾 Silver labels saved to: {SILVER_LABELS_PATH}")
print("✨ Done validating GCN model!")

# ==========================================================
# 🧾 BASELINE SUBMISSION USING TRAINED GCN TEACHER
# ==========================================================
print("\n📦 Generating submission with trained GCN model...")

import csv, os
from tqdm import tqdm

# --- Load test embeddings ---
X_TEST_PATH = Path("Embeddings") / "X_test_nli_roberta.pt"
print("🧠 Loading test embeddings...")
X_test = torch.load(X_TEST_PATH).to(device)
print(f"✓ Test embeddings loaded: {X_test.shape}")

# --- Load test IDs ---
TEST_CORPUS_PATH = Path("Amazon_products") / "test" / "test_corpus.txt"
test_ids = []
with open(TEST_CORPUS_PATH, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t", 1)
        if len(parts) == 2:
            pid, _ = parts
            test_ids.append(int(pid))

assert len(test_ids) == len(X_test), f"Mismatch: {len(test_ids)} IDs vs {len(X_test)} embeddings"

# --- Parameters ---
THRESHOLD = 0.5
MIN_LABELS = 2
MAX_LABELS = 3
OUTPUT_PATH = Path("Submission") / "submission_gcn.csv"
os.makedirs(OUTPUT_PATH.parent, exist_ok=True)

# --- Generate predictions ---
print("\n⚙️ Generating predictions with GCN teacher model...")
model.eval()
all_pids, all_pred_labels = [], []

with torch.no_grad():
    for start in tqdm(range(0, len(X_test), 64), desc="Predicting"):
        end = start + 64
        batch = X_test[start:end]
        batch_pids = test_ids[start:end]

        logits = model(batch)
        probs = torch.sigmoid(logits).cpu().numpy()

        for pid, prob in zip(batch_pids, probs):
            # Binary threshold
            pred_row = (prob > THRESHOLD).astype(int)

            # No label → top-2
            if pred_row.sum() == 0:
                topk_idx = np.argsort(prob)[-MIN_LABELS:][::-1]
                pred_row = np.zeros_like(pred_row)
                pred_row[topk_idx] = 1

            # Only 1 label → add one more
            elif pred_row.sum() == 1:
                top2_idx = np.argsort(prob)[-2:][::-1]
                pred_row = np.zeros_like(pred_row)
                pred_row[top2_idx] = 1

            # Too many → keep top-3
            elif pred_row.sum() > MAX_LABELS:
                topk_idx = np.argsort(prob)[-MAX_LABELS:][::-1]
                pred_row = np.zeros_like(pred_row)
                pred_row[topk_idx] = 1

            labels = [str(j) for j, v in enumerate(pred_row) if v == 1]
            all_pids.append(pid)
            all_pred_labels.append(labels)

print(f"✓ Generated predictions for {len(all_pids)} samples.")

# --- Save submission ---
with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "label"])
    for pid, labels in zip(all_pids, all_pred_labels):
        writer.writerow([pid, ",".join(labels)])

print(f"\n✅ Submission file saved: {OUTPUT_PATH}")
print("🎯 Ready for Kaggle upload!")
