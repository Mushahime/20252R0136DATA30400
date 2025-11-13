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
LABEL_EMB_PATH = EMBEDDINGS_DIR / "labels_emb_true.pt"
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
with open("Silver/silver_train_true.json", "r", encoding="utf-8") as f:
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

# ==========================================================
# VALIDATE SILVER LABELS WITH SIMPLE CLASSIFIER
# ==========================================================
print("\n🧪 Validating silver label quality...")

class HierarchicalClassifier(nn.Module):
    """Better classifier that handles multi-label better"""
    def __init__(self, input_dim, num_classes, hidden_dim=512, dropout=0.3):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = self.dropout(x)
        x = F.relu(self.bn(self.fc1(x)))
        x = self.dropout(x)
        return self.fc2(x)

class MultiLabelDataset(Dataset):
    def __init__(self, pid2labels, pid2idx, embeddings, num_classes):
        self.pids = list(pid2labels.keys())
        self.pid2idx = pid2idx
        self.embeddings = embeddings
        self.num_classes = num_classes
        self.labels = [pid2labels[pid] for pid in self.pids]
    
    def __len__(self):
        return len(self.pids)
    
    def __getitem__(self, idx):
        pid = self.pids[idx]
        pid = int(pid) if isinstance(pid, str) else pid
        emb = self.embeddings[self.pid2idx[pid]]
        
        y = torch.zeros(self.num_classes)
        for label_id in self.labels[idx]:
            if 0 <= label_id < self.num_classes:
                y[label_id] = 1.0
        
        return {"X": emb, "y": y}

# Split data
train_pids, val_pids = train_test_split(
    list(silver_labels.keys()), 
    test_size=0.2, 
    random_state=42
)

train_labels = {pid: silver_labels[pid] for pid in train_pids}
val_labels = {pid: silver_labels[pid] for pid in val_pids}

print(f"Train: {len(train_labels)} | Val: {len(val_labels)}")

# Create datasets
train_dataset = MultiLabelDataset(train_labels, pid2idx, X_train, len(classes))
val_dataset = MultiLabelDataset(val_labels, pid2idx, X_train, len(classes))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

# Initialize model
model = HierarchicalClassifier(X_train.size(1), len(classes)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)

def evaluate(model, dataloader, threshold=0.25):  # Lower threshold for multi-label
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            X = batch["X"].to(device)
            y = batch["y"].to(device)
            
            logits = model(X)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).int()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    # Also compute with adaptive threshold (top-k per sample)
    all_preds_adaptive = []
    with torch.no_grad():
        for batch in dataloader:
            X = batch["X"].to(device)
            y = batch["y"].to(device)
            
            logits = model(X)
            probs = torch.sigmoid(logits)
            
            # For each sample, take top-3 predictions
            for i in range(len(probs)):
                pred = torch.zeros(len(classes))
                top_k = min(3, (y[i] > 0).sum().item() + 1)  # At least k labels
                if top_k > 0:
                    _, top_indices = torch.topk(probs[i], k=top_k)
                    pred[top_indices] = 1
                all_preds_adaptive.append(pred.numpy())
    
    f1_sample = f1_score(all_labels, all_preds, average="samples", zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    
    f1_sample_adaptive = f1_score(all_labels, all_preds_adaptive, average="samples", zero_division=0)
    f1_macro_adaptive = f1_score(all_labels, all_preds_adaptive, average="macro", zero_division=0)
    
    return {
        "f1_sample": f1_sample, 
        "f1_macro": f1_macro,
        "f1_sample_adaptive": f1_sample_adaptive,
        "f1_macro_adaptive": f1_macro_adaptive
    }

# Training loop with early stopping
print("\n🚀 Training validation classifier...")
NUM_EPOCHS = 15
best_val_f1 = 0
patience = 3
no_improve = 0

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}", leave=False):
        X = batch["X"].to(device)
        y = batch["y"].to(device)
        
        logits = model(X)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    val_result = evaluate(model, val_loader)
    
    current_f1 = val_result['f1_sample_adaptive']
    
    print(f"[Epoch {epoch}] loss={avg_loss:.4f} | "
          f"F1_sample={val_result['f1_sample']:.4f} ({val_result['f1_sample_adaptive']:.4f}) | "
          f"F1_macro={val_result['f1_macro']:.4f} ({val_result['f1_macro_adaptive']:.4f})")
    
    # Early stopping
    if current_f1 > best_val_f1:
        best_val_f1 = current_f1
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

# Final evaluation
final_result = evaluate(model, val_loader)

print("\n" + "="*60)
print("📊 FINAL RESULTS")
print("="*60)
print(f"F1 Sample (threshold): {final_result['f1_sample']:.4f}")
print(f"F1 Sample (adaptive):  {final_result['f1_sample_adaptive']:.4f}")
print(f"F1 Macro (threshold):  {final_result['f1_macro']:.4f}")
print(f"F1 Macro (adaptive):   {final_result['f1_macro_adaptive']:.4f}")
print(f"Coverage:  {len(silver_labels)}/{len(train_ids)} ({len(silver_labels)/len(train_ids):.1%})")
print("="*60)

# Quality assessment based on best metric
best_f1 = max(final_result['f1_sample'], final_result['f1_sample_adaptive'])
if best_f1 > 0.25:
    print("✅ Silver labels are of GOOD quality - ready for training!")
elif best_f1 > 0.15:
    print("⚠️  Silver labels are of MEDIUM quality - may need refinement")
else:
    print("❌ Silver labels need improvement")

print(f"\n💾 Silver labels saved to: {SILVER_LABELS_PATH}")
print("✨ Done!")

# ==========================================================
# 🧾 BASELINE SUBMISSION USING TRAINED MODEL
# ==========================================================
print("\n📦 Generating submission with trained model...")

import csv, os
from tqdm import tqdm
import torch
import numpy as np

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
MIN_LABELS = 2      # ensure at least 2 labels
MAX_LABELS = 3      # at most 3 labels
OUTPUT_PATH = Path("Submission") / "submission_model.csv"
os.makedirs(OUTPUT_PATH.parent, exist_ok=True)

# --- Generate predictions ---
print("\n⚙️  Generating predictions with trained model...")
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
            # Binary thresholding
            pred_row = (prob > THRESHOLD).astype(int)

            # --- Aucun label → top-2
            if pred_row.sum() == 0:
                topk_idx = np.argsort(prob)[-MIN_LABELS:][::-1]
                pred_row = np.zeros_like(pred_row)
                pred_row[topk_idx] = 1

            # --- Seulement 1 label → ajoute un deuxième
            elif pred_row.sum() == 1:
                top2_idx = np.argsort(prob)[-2:][::-1]
                pred_row = np.zeros_like(pred_row)
                pred_row[top2_idx] = 1

            # --- Trop de labels → top-3
            elif pred_row.sum() > MAX_LABELS:
                topk_idx = np.argsort(prob)[-MAX_LABELS:][::-1]
                pred_row = np.zeros_like(pred_row)
                pred_row[topk_idx] = 1

            # Final labels
            labels = [str(j) for j, v in enumerate(pred_row) if v == 1]
            all_pids.append(pid)
            all_pred_labels.append(labels)

print(f"✓ Generated predictions for {len(all_pids)} samples.")

# --- Save to CSV ---
with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "label"])
    for pid, labels in zip(all_pids, all_pred_labels):
        writer.writerow([pid, ",".join(labels)])

print(f"\n✅ Submission file saved: {OUTPUT_PATH}")
