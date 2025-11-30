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

import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

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
CLASS_HIERARCHY_PATH = ROOT / "class_hierarchy.txt" 

EMB_DIR          = Path("Embeddings")
X_ALL_PATH       = EMB_DIR / "X_train_test_mpn.pt"
LABEL_EMB_PATH   = EMB_DIR / "labels_hierarchical_new_mpn.pt"

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

def load_multilabel(path):
    """Load multi-label data into {id: [labels]} dictionary."""
    id2labels = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                pid, label = parts
                pid = int(pid)
                label = int(label)

                if pid not in id2labels:
                    id2labels[pid] = []

                id2labels[pid].append(label)
    return id2labels

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

pid2idx = {pid: i for i, pid in enumerate(train_ids)}

class2hierarchy = load_multilabel(CLASS_HIERARCHY_PATH)
print(class2hierarchy)

import json

# Load JSON
with open("Silver/silver_train_new_mpn.json", "r", encoding="utf-8") as f:
    raw = json.load(f)

confidence_threshold = 0.62

pid2labelids_silver_filtered = {}

for pid_str, data in raw.items():
    pid = int(pid_str)

    labels = data["labels"]
    probs = data["probs"]

    # Si AU MOINS un score dépasse le seuil → on garde TOUTE la liste
    if any(score > confidence_threshold for score in probs):
        pid2labelids_silver_filtered[pid] = labels

print(f"Filtered: {len(pid2labelids_silver_filtered)} / {len(raw)}")

silver_labels = pid2labelids_silver_filtered


class MultiLabelDataset(Dataset):
    def __init__(self, pids, labels_dict):
        self.pids = pids
        self.labels = labels_dict

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, idx):
        pid = self.pids[idx]
        emb = X_train[pid2idx[pid]]

        labels = self.labels[pid]

        # soft labels (numpy / list of floats)
        if isinstance(labels, (list, np.ndarray)) and len(labels) == n_classes:
            y = torch.tensor(labels, dtype=torch.float)

        # hard labels (list of class indices)
        else:
            y = torch.zeros(n_classes, dtype=torch.float)
            for c in labels:
                if 0 <= c < n_classes:
                    y[c] = 1.0

        return {"X": emb, "y": y}
    

class UnlabeledEmbeddingDataset(Dataset):
    def __init__(self, pids, pid2idx, embeddings):
        self.pids = pids                
        self.pid2idx = pid2idx             
        self.embeddings = embeddings      

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, idx):
        pid = self.pids[idx]
        emb = self.embeddings[self.pid2idx[pid]]

        return {"X": emb, "pid": pid}


train_p, val_p = train_test_split(
    list(silver_labels.keys()), test_size=0.2, random_state=42
)

train_dataset = MultiLabelDataset(train_p, silver_labels)
val_dataset   = MultiLabelDataset(val_p, silver_labels)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=64)

class LabelGCN(nn.Module):
    def __init__(self, emb_dim, num_layers=1, dropout=0.5):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.W_list = nn.ParameterList()
        for _ in range(num_layers):
            W = nn.Parameter(torch.empty(emb_dim, emb_dim))
            nn.init.xavier_uniform_(W)
            self.W_list.append(W)

    def forward(self, H, A_hat):
        for i, W in enumerate(self.W_list):
            H_input = H  # skip connection

            H_msg = A_hat @ H_input
            H_msg = H_msg @ W

            # residual connection
            H = H_input + H_msg

            if i < self.num_layers - 1:
                H = F.relu(H)
                H = F.dropout(H, p=self.dropout, training=self.training)

        return H


class GCNEnhancedClassifier(nn.Module):
    def __init__(self, input_dim, label_init_emb, A_hat, num_layers=1, dropout=0.2):
        super().__init__()
        emb_dim = label_init_emb.size(1)

        # proj docs -> label space
        self.proj = nn.Linear(input_dim, emb_dim)
        self.dropout = dropout

        # GNN sur les labels
        self.encoder = LabelGCN(emb_dim, num_layers=num_layers, dropout=dropout)

        # label embeddings trainables
        self.label_emb = nn.Parameter(label_init_emb.clone())

        # matrice d’adjacence (buffer, pas un paramètre)
        self.register_buffer("A_hat", A_hat)

    def forward(self, x, use_dropout=True):
        # 1) raffiner les embeddings de labels
        E_refine = self.encoder(self.label_emb, self.A_hat)   # (C, D)

        # 2) projeter les docs
        x_proj = self.proj(x)
        if use_dropout:
            x_proj = F.dropout(x_proj, p=self.dropout, training=self.training)

        # 3) logits = produit scalaire
        logits = x_proj @ E_refine.T    # (B, C)
        return logits
    
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
    f1mic = f1_score(labels, preds, average="micro")
    return f1s, f1m, f1mic


def build_adj_from_hierarchy(class2hierarchy, n_classes, w_parent=1.0, w_sibling=0.1):
    """
    Construit A_hat pour GCN en utilisant EXCLUSIVEMENT class2hierarchy.

    - parent <-> enfant : poids = w_parent
    - frères/soeurs : poids = w_sibling
    - auto-boucle : 1.0 (standard GCN)
    """

    A = torch.zeros((n_classes, n_classes))

    # ---- liens parent/enfant + siblings ----
    for parent, children in class2hierarchy.items():

        # parent <-> enfant
        for c in children:
            A[parent, c] = w_parent
            A[c, parent] = w_parent

        # siblings (enfants du même parent)
        for i in range(len(children)):
            for j in range(i + 1, len(children)):
                c1, c2 = children[i], children[j]
                A[c1, c2] = w_sibling
                A[c2, c1] = w_sibling

    # ---- self-loops ----
    A = A + torch.eye(n_classes)

    # ---- normalisation GCN ----
    D = A.sum(dim=1)
    D_inv_sqrt = torch.pow(D, -0.5)
    D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0
    D_mat = torch.diag(D_inv_sqrt)

    A_hat = D_mat @ A @ D_mat
    return A_hat

def load_multilabel(path):
    """
    Charge un fichier parent-enfant du type :
    parent_id \t child_id

    Retourne :
    {parent: [child, ...]}
    """
    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            p, c = line.strip().split("\t")
            p, c = int(p), int(c)

            if p not in mapping:
                mapping[p] = []

            mapping[p].append(c)

    return mapping

# ---------- CHARGEMENT HIÉRARCHIE ----------
CLASS_HIERARCHY_PATH = ROOT / "class_hierarchy.txt"
class2hierarchy = load_multilabel(CLASS_HIERARCHY_PATH)

A_hat = build_adj_from_hierarchy(class2hierarchy, n_classes).to(device)

import copy

print("\nTraining...")
best = 0
epochs = 5
wait = 0
patience = 5

model = GCNEnhancedClassifier(
    input_dim=X_train.size(1),
    label_init_emb=label_emb,
    A_hat=A_hat,
    num_layers=3,     
    dropout=0.2
).to(device)

best_model = copy.deepcopy(model.state_dict())

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

for epoch in range(1, epochs+1):
    model.train()
    total = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
        X = batch["X"].to(device)
        y = batch["y"].to(device)

        logits = model(X)
        loss = F.binary_cross_entropy_with_logits(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += loss.item()

    f1s, f1m, f1mic = evaluate(model, val_loader)
    print(f"[Epoch {epoch}] loss={total/len(train_loader):.4f} | F1={f1s:.4f}")

    if f1s > best:
        best = f1s
        wait = 0
        best_model = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"New best model saved ({best:.4f})")
    else:
        wait += 1
    
    if wait >= patience:
        print("\nEarly stopping triggered!")
        break


print(f"\nBest validation F1 = {best:.4f}")
print(f"Model saved at: {MODEL_PATH}")

# === Load best student before test ===
model.load_state_dict(best_model)


student = model

teacher = GCNEnhancedClassifier(
    input_dim=X_train.size(1),
    label_init_emb=label_emb,
    A_hat=A_hat,
    num_layers=3,     
    dropout=0.0
).to(device)

teacher.load_state_dict(best_model) 
teacher.eval()

def ema_update(teacher, student, alpha=0.999):
    for t_param, s_param in zip(teacher.parameters(), student.parameters()):
        t_param.data.mul_(alpha).add_(s_param.data * (1 - alpha))


EPOCHS = 100
pseudo_update_freq = 5

threshold_start = 0.7
threshold_end   = 0.85

alpha_ema = 0.99
lambda_cons = 1.5

current_labels = dict(silver_labels) 
all_train_pids = list(pid2idx.keys())

optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

wait = 0
patience = 5
best_f1 = 0.
best_teacher = copy.deepcopy(teacher.state_dict())

def consistency_loss(log_s, log_t):
    ps = torch.sigmoid(log_s)
    pt = torch.sigmoid(log_t)
    return F.mse_loss(ps, pt)


def expand_with_hierarchy(labels, hierarchy):
    """
    Expand a list of core labels by adding ALL their ancestors
    (parents, parents of parents, etc.), recursively.
    This guarantees 100% hierarchy consistency.
    """
    expanded = set(labels)
    stack = list(labels)
    child2parents = {}
    for parent, children in hierarchy.items():
        for child in children:
            child2parents.setdefault(child, []).append(parent)

    # DFS / BFS upward through ancestors
    while stack:
        node = stack.pop()

        if node not in child2parents:
            continue

        for parent in child2parents[node]:
            if parent not in expanded:
                expanded.add(parent)
                stack.append(parent)  

    return sorted(expanded)[-3:]


def select_labels_hierarchical(probs, threshold, hierarchy, max_k=3):
    # 1) labels surpassant le seuil
    cand = [i for i in range(len(probs)) if probs[i] > threshold]

    if len(cand) == 0:
        return []

    # ordonner par probas
    cand = sorted(cand, key=lambda c: -probs[c])
    # prendre au maximum 3 feuilles candidates
    cand = cand[:max_k]

    # 2) expansion hiérarchique
    expanded = set(cand)
    for c in cand:
        if str(c) in hierarchy:
            parents = hierarchy[str(c)]
            for p in parents:
                expanded.add(p)

    expanded = list(expanded)

    # 3) garder max 3 labels au total
    expanded = sorted(expanded, key=lambda c: -probs[c])[:max_k]

    return expanded


def invert_hierarchy(class2hierarchy):
    inv = {}
    for parent, children in class2hierarchy.items():
        for child in children:
            inv.setdefault(child, []).append(parent)
    return inv

inv_hierarchy = invert_hierarchy(class2hierarchy)


def generate_pseudo_labels(threshold):
    teacher.eval()
    new_pseudo = {}

    labeled = set(current_labels.keys())
    unlabeled_pids = [pid for pid in all_train_pids if pid not in labeled]

    unlabeled_ds = UnlabeledEmbeddingDataset(unlabeled_pids, pid2idx, X_train)
    unlabeled_ld = DataLoader(unlabeled_ds, batch_size=64)

    with torch.no_grad():
        for batch in unlabeled_ld:
            X = batch["X"].to(device)
            pids = batch["pid"]

            logits = teacher(X)
            probs_batch = torch.sigmoid(logits).cpu().numpy()

            for pid, p in zip(pids, probs_batch):

                final_labels = select_labels_hierarchical(
                    probs=p,
                    threshold=threshold,
                    hierarchy=inv_hierarchy,
                    max_k=3
                )

                if len(final_labels) < 2:
                    continue

                new_pseudo[int(pid)] = final_labels

    return new_pseudo

def generate_soft_pseudo_labels(threshold):
    teacher.eval()
    new_pseudo = {}

    labeled = set(current_labels.keys())
    unlabeled_pids = [pid for pid in all_train_pids if pid not in labeled]

    unlabeled_ds = UnlabeledEmbeddingDataset(unlabeled_pids, pid2idx, X_train)
    unlabeled_ld = DataLoader(unlabeled_ds, batch_size=64)

    with torch.no_grad():
        for batch in unlabeled_ld:
            X = batch["X"].to(device)
            pids = batch["pid"]

            logits = teacher(X)
            probs = torch.sigmoid(logits).cpu()  # (B, C)

            for pid, p in zip(pids, probs):
                # ne garde que si au moins une classe dépasse le seuil
                if p.max().item() < threshold:
                    continue

                # garder uniquement les top-3 (pour rester compatible avec ton code)
                topk = torch.topk(p, 3)
                mask = torch.zeros_like(p)
                mask[topk.indices] = topk.values   # soft labels

                new_pseudo[int(pid)] = mask.numpy()

    return new_pseudo



total = 0

for epoch in range(1, EPOCHS + 1):

    if epoch % pseudo_update_freq == 1 and epoch > 1:
        progress = epoch / EPOCHS
        thr = threshold_start + (threshold_end - threshold_start) * (epoch / EPOCHS)**2

        new_pseudo = generate_soft_pseudo_labels(thr)

        if len(new_pseudo) > 0:
            before = len(current_labels)
            current_labels.update(new_pseudo)
            print(f"Added {len(current_labels) - before} pseudo-labeled examples")
            total += len(current_labels) - before

    train_ds = MultiLabelDataset(list(current_labels.keys()), current_labels)
    train_ld = DataLoader(train_ds, batch_size=64, shuffle=True)

    student.train()
    teacher.eval()

    epoch_loss = 0.

    for batch in train_ld:
        X = batch["X"].to(device)
        y = batch["y"].to(device)

        # student (noisy)
        noise = torch.randn_like(X) * 0.05
        logits_s = student(X + noise)

        # teacher (clean)
        with torch.no_grad():
            logits_t = teacher(X)

        # Multi-label supervised loss
        loss_sup = F.binary_cross_entropy_with_logits(logits_s, y)

        # Consistency loss
        loss_cons = consistency_loss(logits_s, logits_t)
        loss = loss_sup + lambda_cons * loss_cons

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # EMA teacher update
        ema_update(teacher, student, alpha_ema)

        epoch_loss += loss.item()

    scheduler.step()

    # Validation with teacher
    teacher.eval()
    f1s, f1m, f1mic = evaluate(teacher, val_loader)

    print(f"Epoch {epoch} | Loss={epoch_loss/len(train_ld):.4f} | F1={f1s:.4f}")

    if f1s > best_f1:
        best_f1 = f1s
        best_teacher = copy.deepcopy(teacher.state_dict())
        wait = 0
        print(f"New best F1 = {best_f1:.4f}")
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping triggered")
            print(f"Number of PS added {total}")
            break

# Load best teacher
teacher.load_state_dict(best_teacher)
print("\nFinal teacher F1:", best_f1)


import csv
import numpy as np
from pathlib import Path

print("\n📝 Generating submission...")

teacher.eval()
X_test = X_test.to(device)

def select_k(prob, min_k=2, max_k=3):
    idx = np.argsort(prob)[::-1]    # sorted descending

    # Always take the best 3 candidates
    top3 = idx[:max_k]

    # If the 3rd is much weaker → keep only 2
    if prob[top3[2]] < 0.25 * prob[top3[1]]:
        return top3[:2]

    return top3

preds = []

with torch.no_grad():
    for start in tqdm(range(0, len(X_test), 64)):
        batch = X_test[start:start+64]

        logits = teacher(batch, use_dropout=False)
        probs = torch.sigmoid(logits).cpu().numpy()

        for p in probs:
            labels = select_k(p)
            preds.append([str(x) for x in labels])

# ==========================================================
# SAVE CSV
# ==========================================================

OUT_DIR = Path("Submission")
OUT_DIR.mkdir(exist_ok=True)
OUT_PATH = OUT_DIR / "submission_selfGNN.csv"

with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["id","label"])
    for pid, labels in zip(test_ids, preds):
        w.writerow([pid, ",".join(labels)])

print(f"🎉 Submission saved → {OUT_PATH}")

