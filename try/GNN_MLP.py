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
from utils import *


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


# ==========================================================
# LOAD SILVER LABELS
# ==========================================================
with open("Silver/silver_train_new_mpn.json", "r", encoding="utf-8") as f:
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

train_p, val_p = train_test_split(
    list(silver_labels.keys()), test_size=0.2, random_state=42
)

train_dataset = MultiLabelDataset(train_p, silver_labels)
val_dataset   = MultiLabelDataset(val_p, silver_labels)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=64)


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
        #self.proj = nn.Linear(input_dim, emb_dim)

        self.proj = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(emb_dim, emb_dim)   # ← MLP supplémentaire
        )
        

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

print("A_hat:", A_hat.shape, "Non-zero =", (A_hat > 0).sum().item())
print("A_hat type:", A_hat.dtype)
print("A_hat device:", A_hat.device)


import copy

# ----------------------------
# Hyperparameters
# ----------------------------
epochs = 100
patience = 5
wait = 0
best_f1 = 0

alpha_ema = 0.99       # teacher EMA speed
lambda_cons = 0.5     # weight for consistency loss
noise_std = 0.05        # noise on student input

# ----------------------------
# Init student + teacher
# ----------------------------
student = GCNEnhancedClassifier(
    input_dim=X_train.size(1),
    label_init_emb=label_emb,
    A_hat=A_hat,
    num_layers=3,
    dropout=0.2        # student = bruit
).to(device)

teacher = GCNEnhancedClassifier(
    input_dim=X_train.size(1),
    label_init_emb=label_emb,
    A_hat=A_hat,
    num_layers=3,
    dropout=0.0        # teacher = STABLE
).to(device)

teacher.load_state_dict(student.state_dict())

optimizer = torch.optim.AdamW(student.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

best_teacher = copy.deepcopy(teacher.state_dict())

# ----------------------------
# Consistency loss
# ----------------------------
def consistency_loss(log_s, log_t):
    ps = torch.sigmoid(log_s)
    pt = torch.sigmoid(log_t)
    return F.mse_loss(ps, pt)

# ----------------------------
# Training loop
# ----------------------------
for epoch in range(1, epochs + 1):
    student.train()
    teacher.eval()

    total_loss = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
        X = batch["X"].to(device)
        y = batch["y"].to(device)

        # Add noise to student
        noisy_X = X + noise_std * torch.randn_like(X)

        # student forward
        logits_s = student(noisy_X)

        # teacher forward (no gradient)
        with torch.no_grad():
            logits_t = teacher(X)

        # supervised = main objective
        loss_sup = F.binary_cross_entropy_with_logits(logits_s, y)

        # consistency = stability objective
        loss_cons = consistency_loss(logits_s, logits_t)

        # total loss
        loss = loss_sup + lambda_cons * loss_cons

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # EMA update (teacher → student)
        for t_param, s_param in zip(teacher.parameters(), student.parameters()):
            t_param.data = alpha_ema * t_param.data + (1 - alpha_ema) * s_param.data

        total_loss += loss.item()

    scheduler.step()


    teacher.eval()
    f1_sample, f1_macro = evaluate(teacher, val_loader)

    print(f"[Epoch {epoch}] Loss={total_loss/len(train_loader):.4f} | F1={f1_sample:.4f}")

    if f1_sample > best_f1:
        best_f1 = f1_sample
        best_teacher = copy.deepcopy(teacher.state_dict())
        wait = 0
        print(f"New best model saved (F1={best_f1:.4f})")
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping triggered")
            break


teacher.load_state_dict(best_teacher)
print("\n🎉 Final best F1:", best_f1)

import csv
import numpy as np
from pathlib import Path

print("\nGenerating submission...")

teacher.eval()

X_test = X_test.to(device)

def select_k(prob, min_k=2, max_k=3):
    idx = np.argsort(prob)[::-1]  # descend
    top3 = idx[:max_k]

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
OUT_PATH = OUT_DIR / "submission_GNN.csv"

with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["id", "label"])
    for pid, labels in zip(test_ids, preds):
        w.writerow([pid, ",".join(labels)])

print(f"🎉 Submission saved → {OUT_PATH}")