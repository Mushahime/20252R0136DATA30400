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
import copy
from utils import *
import csv
import random

from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

np.random.seed(42)
torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Paths
ROOT = Path("Amazon_products")
TRAIN_CORPUS_PATH = ROOT / "train" / "train_corpus.txt"
TEST_CORPUS_PATH  = ROOT / "test" / "test_corpus.txt"
CLASS_PATH        = ROOT / "classes.txt"

EMB_DIR      = Path("Embeddings")
X_ALL_PATH   = EMB_DIR / "X_train_test_mpnet.pt" # Train + Test embeddings
LABEL_EMB_PATH = EMB_DIR / "labels_mpnet.pt"

MODEL_SAVE = Path("Models")
MODEL_SAVE.mkdir(exist_ok=True)
MODEL_PATH = MODEL_SAVE / "classifierInner.pt"
CLASS_HIERARCHY_PATH = ROOT / "class_hierarchy.txt"

label_emb = torch.load(LABEL_EMB_PATH).float().to(device)
print("Label embeddings:", label_emb.shape)

# Load useful data (like silver gen)
def load_classic(path):
    id2text = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            pid, text = line.strip().split("\t", 1)
            id2text[int(pid)] = text
    return id2text

def load_multilabel(path):
    """Load multi-label data into {id: [labels]} dictionary -> for class_hierarchy"""
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

id2text_train = load_classic(TRAIN_CORPUS_PATH)
id2text_test  = load_classic(TEST_CORPUS_PATH)
train_ids = list(id2text_train.keys())
test_ids  = list(id2text_test.keys())
n_train = len(train_ids)
n_test  = len(test_ids)
print(f"Train IDs: {n_train} | Test IDs: {n_test}")

# Load silver label (hier & nohier)
with open("Silver/silver_train_new_mpnet.json", "r", encoding="utf-8") as f:
    raw = json.load(f)
silver_labels = {int(pid): data["labels"] for pid, data in raw.items()}

# Load X_all + split into X_train & X_test
data = torch.load(X_ALL_PATH, weights_only=False)

# ensure tensor (check)
if isinstance(data, np.ndarray):
    data = torch.from_numpy(data)
elif isinstance(data, list):
    data = torch.stack(data)

X_all = data.float().to(device)
X_train = X_all[:n_train]
X_test  = X_all[n_train:]
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

# Load Class names
classes = {}
with open(CLASS_PATH, "r", encoding="utf-8") as f:
    for line in f:
        cid, cname = line.strip().split("\t")
        classes[int(cid)] = cname

n_classes = len(classes) # 531
print(n_classes)

pid2idx = {pid: i for i, pid in enumerate(train_ids)}

class2hierarchy = load_multilabel(CLASS_HIERARCHY_PATH)
print(class2hierarchy)


# Load Silver labels
with open("Silver/silver_train_new_mpnet.json", "r", encoding="utf-8") as f:
    raw = json.load(f)

silver_labels = {int(pid): data["labels"] for pid, data in raw.items()}

# Compute similarities
X_train_norm = F.normalize(X_train, p=2, dim=1).to(device)
label_emb_norm = F.normalize(label_emb, p=2, dim=1).to(device)
all_similarities = torch.matmul(X_train_norm, label_emb_norm.T).cpu()

# Map scores to silver labels
silver_with_scores = {}
for idx, pid in enumerate(train_ids):
    if pid not in silver_labels:
        continue

    labels = silver_labels[pid]
    sims = all_similarities[idx]

    label_scores = [float(sims[label]) for label in labels]

    silver_with_scores[pid] = {
        "labels": labels,
        "scores": label_scores,
        "max_score": max(label_scores),
    }

# Extract quartiles
max_scores = [info["max_score"] for info in silver_with_scores.values()]

Q1 = np.quantile(max_scores, 0.25)
Q2 = np.quantile(max_scores, 0.50)
Q3 = np.quantile(max_scores, 0.75)

print(f"Q1(25%): {Q1:.4f}")
print(f"Q2(50%): {Q2:.4f}")
print(f"Q3(75%): {Q3:.4f}")

# Choose threshold
threshold = Q3  # Top 25% more confident
print(f"\nUsing threshold: {threshold:.4f}")

# Split pseudo-labeled vs unlabeled
pseudo_pids = [pid for pid, info in silver_with_scores.items() if info["max_score"] >= threshold]

unlabeled_pids = [pid for pid, info in silver_with_scores.items() if info["max_score"] < threshold]

print(f"\nPseudo-labeled count : {len(pseudo_pids)}")
print(f"Unlabeled count : {len(unlabeled_pids)}")

# Build final datasets
silver_dataset = {pid: silver_with_scores[pid]["labels"] for pid in pseudo_pids}

unlabeled_dataset = unlabeled_pids

# Contrastive finetuning
print("finetuning...")

# Projection head to adapt and refine embedding space
class ProjectionHead(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        return self.linear(x)

proj_head = ProjectionHead(dim=X_train.shape[1]).to(device)

# Contrastive dataset construction (GPT helps me on this one)
class ContrastiveDataset(Dataset):
    """
    Builds pairs of (text embedding, label embedding) with:
    - target = 1 for positive label pairs
    - target = -1 for negative sampled label pairs
    """
    def __init__(self, train_ids, silver_dataset, X_train, label_emb, neg_ratio=4):
        self.samples = []
        all_labels = list(range(label_emb.shape[0]))

        for pid in train_ids:
            if pid not in silver_dataset:
                continue

            pos_labels = silver_dataset[pid]

            # Add positive pairs
            for lbl in pos_labels:
                self.samples.append((pid, lbl, 1))

            # Add negative sampled pairs
            neg_candidates = [l for l in all_labels if l not in pos_labels]
            negs = random.sample(neg_candidates, neg_ratio * len(pos_labels))

            for neg_lbl in negs:
                self.samples.append((pid, neg_lbl, -1))

        self.X_train = X_train
        self.label_emb = label_emb

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pid, lbl, target = self.samples[idx]

        text_vec = self.X_train[pid]
        label_vec = self.label_emb[lbl]

        return (
            torch.tensor(text_vec, dtype=torch.float),
            torch.tensor(label_vec, dtype=torch.float),
            torch.tensor(target, dtype=torch.float)
        )

dataset = ContrastiveDataset(
    train_ids=train_ids,
    silver_dataset=silver_dataset,
    X_train=X_train.detach().cpu().numpy(),
    label_emb=label_emb.detach().cpu().numpy(),
    neg_ratio=4
)
loader = DataLoader(dataset, batch_size=64, shuffle=True)
optimizer = torch.optim.AdamW(proj_head.parameters(), lr=3e-4)

# We use cosine distance instead of BCE or 
# softmax because the model outputs embeddings, not probabilities, and the
# task is metric learning (similarity-based), not classification.
loss_fn = nn.CosineEmbeddingLoss(margin=0.5)

# Training loop (update proj head)
EPOCHS = 5

for epoch in range(EPOCHS):
    total_loss = 0

    for x, y, target in loader:
        x, y, target = x.to(device), y.to(device), target.to(device)

        # Apply projection head to both embeddings
        x_proj = proj_head(x)
        y_proj = proj_head(y)

        # Normalize outputs
        x_proj = F.normalize(x_proj, dim=1)
        y_proj = F.normalize(y_proj, dim=1)

        # Compute contrastive loss
        loss = loss_fn(x_proj, y_proj, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")

print("\nFinetuning completed.\n")


# Apply projection head to all embeddings
X_train_norm = F.normalize(proj_head(X_train.to(device)), dim=1)
X_test_norm = F.normalize(proj_head(X_test.to(device)), dim=1)
label_emb_norm = F.normalize(proj_head(label_emb.to(device)), dim=1)

# Recompute PL
# Pairwise similarity matrix
all_similarities = torch.matmul(X_train_norm, label_emb_norm.T).cpu()

# Store similarity scores for the original silver labels
silver_with_scores = {}

for idx, pid in enumerate(train_ids):
    if pid not in silver_labels:
        continue

    labels = silver_labels[pid]
    sims = all_similarities[idx]

    label_scores = [float(sims[label]) for label in labels]

    silver_with_scores[pid] = {
        "labels": labels,
        "scores": label_scores,
        "max_score": max(label_scores),
    }

# Compute quartiles
max_scores = [info["max_score"] for info in silver_with_scores.values()]

Q1 = np.quantile(max_scores, 0.25)
Q2 = np.quantile(max_scores, 0.50)
Q3 = np.quantile(max_scores, 0.75)

print(f"Q1(25%): {Q1:.4f}")
print(f"Q2(50%): {Q2:.4f}")
print(f"Q3(75%): {Q3:.4f}")

# Threshold based on top quartile
threshold = Q3
print(f"\nUsing threshold: {threshold:.4f}")

# Split into pseudo-labeled and unlabeled subsets
pseudo_pids = [pid for pid, info in silver_with_scores.items() if info["max_score"] >= threshold]

unlabeled_pids = [pid for pid, info in silver_with_scores.items() if info["max_score"] < threshold]

print(f"\nPseudo-labeled count : {len(pseudo_pids)}")
print(f"Unlabeled count : {len(unlabeled_pids)}")

# Build final datasets
silver_dataset_finetuned = {
    pid: silver_with_scores[pid]["labels"]
    for pid in pseudo_pids
}

unlabeled_dataset_finetuned = unlabeled_pids

X_train_norm = X_train_norm.detach()
X_test_norm = X_test_norm.detach()
label_emb_norm = label_emb_norm.detach()
X_train_norm = torch.stack([X_train_norm[pid2idx[pid]] for pid in train_ids])

# Rebuild pid2idx correctly
pid2idx = {pid: i for i, pid in enumerate(train_ids)}

class MultiLabelDataset(Dataset):
    """
    Simple PyTorch dataset for multi-label classification.
    Takes a list of product IDs and a dict pid -> list of labels,
    and returns (embedding, multi-hot label vector) for each item.
    """
    def __init__(self, pids, labels_dict):
        self.pids = pids
        self.labels = labels_dict

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, idx):
        pid = self.pids[idx]
        emb = X_train_norm[pid2idx[pid]]

        y = torch.zeros(n_classes)
        for c in self.labels[pid]:
            if 0 <= c < n_classes:
                y[c] = 1.0 # one-hot multi-label vector

        return {"X": emb, "y": y}
    

class UnlabeledEmbeddingDataset(Dataset):
    """
    Dataset for unlabeled samples used in self-training.
    """
    def __init__(self, pids):
        self.pids = pids  # list of unlabeled product IDs

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, idx):
        pid = self.pids[idx]
        emb = X_train_norm[pid2idx[pid]]

        return {"X": emb, "pid": pid}    

# TRAIN / VAL splits
train_p, val_p = train_test_split(list(silver_dataset.keys()),test_size=0.15,random_state=42)

train_dataset = MultiLabelDataset(train_p, silver_dataset)
val_dataset   = MultiLabelDataset(val_p,   silver_dataset)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=64)

print(len(train_dataset))

class InnerProductClassifier(nn.Module):
    """
    A simple classifier that projects input embeddings and
    scores each label via inner product with its embedding.
    """
    def __init__(self, input_dim, label_embeddings, dropout=0.2, trainable_label_emb=False):
        super().__init__()

        D = label_embeddings.size(1)

        self.proj = nn.Linear(input_dim, D)
        self.dropout = nn.Dropout(dropout)

        # Label embeddings can be fixed or trainable (for innerproduct we'll prefer fixed cause 
        # it preserves semantic meaning from pretrained embeddings and reduces overfitting risk 
        # on noisy silver labels)        
        if trainable_label_emb:
            self.label_emb = nn.Parameter(label_embeddings.clone())
        else:
            self.register_buffer("label_emb", label_embeddings.clone())

    def forward(self, x, use_dropout=True):
        # project input
        x_proj = self.proj(x)
        
        if use_dropout:
            x_proj = self.dropout(x_proj)
        
        # normalize
        x_proj = F.normalize(x_proj, dim=1)
        label_emb = F.normalize(self.label_emb, dim=1)
        
        # Scale to expand sigmoid range: normalized inner products ∈ [-1,1] -> need scaling for confident predictions
        logits = (x_proj @ label_emb.T) * 30.0
        return logits

# Evaluation metrics -> see paper given
def precision_at_1(y_true, scores):
    """
    Computes Precision@1 for multi-label classification.
    For each sample, we check if the top-1 predicted class is actually a true label.
    """
    correct = 0
    for yt, sc in zip(y_true, scores):
        top1 = sc.argmax()
        if yt[top1] == 1:
            correct += 1
    return correct / len(y_true)


def precision_at_3(y_true, scores):
    """
    Computes Precision@3 for multi-label classification.
    For each sample, we check how many of the top-3 predicted classes
    match the true labels, and average over all samples.
    """
    total_correct = 0
    for yt, sc in zip(y_true, scores):
        top3 = sc.argsort()[-3:][::-1]   # indices of best 3 scores
        total_correct += yt[top3].sum()
    return total_correct / (3 * len(y_true))


# MLPs generate high-magnitude logits (leading to sigmoid outputs near 0 or 1), so thresholds around 0.4–0.6 perform best.
# In contrast, InnerProduct and GNN models use cosine-like similarity scores,which are naturally compressed in the range 0.15–0.40 after sigmoid, thus requiring lower thresholds (≈0.20–0.30).
def evaluate(model, loader, thr=0.15):
    """
    Evaluate a multi-label classifier on a dataloader.
    Applies sigmoid to get probabilities, turns them into binary labels
    using a threshold, and computes sample-wise and macro F1 scores.
    """
    model.eval()
    preds, labels = [], []
    all_scores = []   # predicted scores for each example
    all_true = []     # true binary vectors

    with torch.no_grad():
        for batch in loader:
            X = batch["X"]
            y = batch["y"].numpy()

            logits = model(X)
            prob = torch.sigmoid(logits).cpu().numpy()

            # For threshold metrics
            pred = (prob > thr).astype(int)
            preds.extend(pred)
            labels.extend(y)

            # For ranking metrics
            all_scores.extend(prob)
            all_true.extend(y)

    # F1 metrics
    f1s = f1_score(labels, preds, average="samples")
    f1m = f1_score(labels, preds, average="macro")

    # Ranking metrics
    P1 = precision_at_1(all_true, all_scores)
    P3 = precision_at_3(all_true, all_scores)

    return f1s, f1m, P1, P3

# Classic training without techniques of regularization only early stopping and bestf1 improvement
model = InnerProductClassifier(X_train_norm.size(1), label_emb_norm, 0.2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

epochs = 100
val_f1_list = []
val_p1_list = []
val_p3_list = []
best_f1 = 0
best_state = None
patience = 5
wait = 0

for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0.0

    # Training
    for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
        X = batch["X"].to(device)
        y = batch["y"].to(device)

        logits = model(X)
        # We use BCE-with-logits because we are using a multi-label system: each class must have its own probability. 
        # This loss automatically applies a sigmoid to each class and calculates the bit error for each one.
        loss = F.binary_cross_entropy_with_logits(logits, y) 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    f1s, f1m, p1, p3 = evaluate(model, val_loader)
    val_f1_list.append(f1s)
    val_p1_list.append(p1)
    val_p3_list.append(p3)

    print(f"[Epoch {epoch}] loss={avg_loss:.4f} | F1={f1s:.4f}")

    # Save best
    if f1s > best_f1: # we compare on f1 sample like kaggle
        best_f1 = f1s
        best_state = copy.deepcopy(model.state_dict())
        torch.save(best_state, MODEL_PATH)
        print(f"New best model (F1={best_f1:.4f})")
        wait = 0
    else:
        wait += 1 # no improvement this epoch
        print(f" No improvement for {wait} epoch(s).")
    
    if wait >= patience:
        print(f"\nEarly stopping triggered after {epoch} epochs.")
        break

print("\nTraining finished")
print(f"Best validation F1 = {best_f1:.4f}")

# Load best model
model.load_state_dict(best_state)

save_model = copy.deepcopy(model)
