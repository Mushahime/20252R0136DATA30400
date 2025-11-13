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
EMBEDDINGS_PATH = EMBEDDINGS_DIR / "X_train_nli_roberta.pt"
LABEL_EMB_PATH = EMBEDDINGS_DIR / "label_emb_nli_roberta.pt"

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

# ==========================================================
# TF-IDF
# ==========================================================

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("\n🧮 Building TF-IDF representations with class-related keywords...")

# ==========================================================
# LOAD CLASS-RELATED KEYWORDS
# ==========================================================
CLASS_RELATED_PATH = ROOT / "class_related_keywords.txt"
class2related = {}

if CLASS_RELATED_PATH.exists():
    with open(CLASS_RELATED_PATH, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                class_name, kws = parts[0], parts[1:]
                class2related[class_name] = kws
else:
    print("⚠️  No class_related_keywords.txt found — continuing without keyword enrichment.")

# ==========================================================
# TEXT PREPROCESSING
# ==========================================================
def preprocess_text(text):
    text = re.sub(r"[>&]", " ", text)
    text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

# ==========================================================
# BUILD TF-IDF INPUTS
# ==========================================================
NUM_CLASSES = len(classes)

# Cleaned label texts (class name + related keywords)
label_texts = []
for cid in range(NUM_CLASSES):
    class_name = classes[cid]
    class_name_clean = preprocess_text(class_name)

    # Add related keywords if any
    keywords = class2related.get(class_name, [])
    keywords_clean = " ".join(preprocess_text(kw) for kw in keywords)
    full_label_text = (class_name_clean + " " + keywords_clean).strip()

    label_texts.append(full_label_text)

# Clean training docs
train_ids = list(id2text.keys())
train_texts = [preprocess_text(id2text[pid]) for pid in train_ids]

# ==========================================================
# TF-IDF FITTING
# ==========================================================
print("Vectorizing with TF-IDF...")
vectorizer = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.7
)

all_texts = train_texts + label_texts
tfidf_matrix = vectorizer.fit_transform(all_texts)

n_train = len(train_texts)
n_labels = len(label_texts)

train_tfidf = tfidf_matrix[:n_train]
label_tfidf = tfidf_matrix[n_train:]

print(f"✓ TF-IDF matrices built:")
print(f"  Train:  {train_tfidf.shape}")
print(f"  Labels: {label_tfidf.shape}")

# ==========================================================
# LEXICAL SIMILARITY (TF-IDF COSINE)
# ==========================================================
print("Computing lexical similarity (train × labels)...")
lex_sim_train = cosine_similarity(train_tfidf, label_tfidf)

print(f"✓ Lexical similarity matrix shape: {lex_sim_train.shape}")


# ==========================================================
# LOAD EMBEDDINGS
# ==========================================================
print("\n🧠 Loading embeddings...")
X_train = torch.load(EMBEDDINGS_PATH).to(device)
label_emb = torch.load(LABEL_EMB_PATH).to(device)

print(f"✓ Train embeddings: {X_train.shape}")
print(f"✓ Label embeddings: {label_emb.shape}")

# Create pid to index mapping
train_ids = list(id2text.keys())
pid2idx = {pid: i for i, pid in enumerate(train_ids)}

# ==========================================================
# BUILD HIERARCHY STRUCTURES
# ==========================================================
print("\n🌳 Building hierarchy structures...")

# Find root and build child->parent mapping
child2parent = {}
all_children = set()
for parent, children in hierarchy.items():
    for child in children:
        child2parent[child] = parent
        all_children.add(child)

# Root is a class that has no parent
all_classes = set(classes.keys())
all_children = set(child2parent.keys())
root_classes = sorted(list(all_classes - all_children))
print(f"✓ Found {len(root_classes)} root classes: {[classes[r] for r in root_classes[:6]]}")



def get_ancestors(class_id):
    """Get all ancestors of a class (including itself)"""
    ancestors = [class_id]
    current = class_id
    while current in child2parent:
        current = child2parent[current]
        ancestors.append(current)
    return ancestors

def get_descendants(class_id):
    """Get all descendants of a class (including itself)"""
    descendants = [class_id]
    if class_id in hierarchy:
        for child in hierarchy[class_id]:
            descendants.extend(get_descendants(child))
    return descendants

# Get leaf classes (classes with no children)
leaf_classes = [c for c in all_classes if c not in hierarchy]
print(f"✓ Found {len(leaf_classes)} leaf classes")

# ==========================================================
# GENERATE SILVER LABELS
# ==========================================================
print("\n⚙️  Generating silver labels...")

"""def compute_similarity(doc_emb, class_embs):
    doc_emb = F.normalize(doc_emb.unsqueeze(0), dim=1)
    class_embs = F.normalize(class_embs, dim=1)
    return (doc_emb @ class_embs.T).squeeze(0)"""

def compute_similarity(doc_emb, class_embs, pid=None, alpha=0.8):
    """
    Combine semantic (embedding) similarity and lexical (TF-IDF) similarity.
    alpha = weight for embedding similarity.
    """
    # Semantic similarity
    doc_emb = F.normalize(doc_emb.unsqueeze(0), dim=1)
    class_embs = F.normalize(class_embs, dim=1)
    sim_emb = (doc_emb @ class_embs.T).squeeze(0).cpu().numpy()
    
    # Lexical similarity
    sim_lex = lex_sim_train[pid2idx[pid]] if pid is not None else np.zeros_like(sim_emb)
    
    # Combine
    sim_combined = alpha * sim_emb + (1 - alpha) * sim_lex
    return torch.tensor(sim_combined, device=device)

def get_level(class_id):
    """Get depth level of a class in hierarchy"""
    level = 0
    current = class_id
    while current in child2parent:
        current = child2parent[current]
        level += 1
    return level

# Build level structure
class_levels = {c: get_level(c) for c in all_classes}
max_level = max(class_levels.values())
levels_to_classes = {}
for c, level in class_levels.items():
    if level not in levels_to_classes:
        levels_to_classes[level] = []
    levels_to_classes[level].append(c)

print(f"✓ Hierarchy has {max_level + 1} levels")
for level in sorted(levels_to_classes.keys()):
    print(f"  Level {level}: {len(levels_to_classes[level])} classes")

def generate_hierarchical_multilabel(top_k_per_level=5, min_similarity=0.25, max_cores=3):
    """
    Hierarchical Multi-Label approach (multi-root version, inspired by TaxoClass):
    1. For each root class (no parent), run top-down selection
    2. Identify confident/core classes using similarity & relative confidence
    3. Add ancestors of core classes (path completion)
    """

    silver_labels = {}

    # ✅ 1. Identifier toutes les racines (classes sans parent)
    all_classes = set(classes.keys())
    all_children = set(child2parent.keys())
    root_classes = sorted(list(all_classes - all_children))
    print(f"✓ Found {len(root_classes)} root classes: {[classes[r] for r in root_classes[:6]]}")

    for pid in tqdm(train_ids, desc="Generating hierarchical multi-labels"):
        idx = pid2idx[pid]
        doc_emb = X_train[idx]
        similarities = compute_similarity(doc_emb, label_emb, pid).cpu().numpy()

        final_labels = set()

        # ✅ 2. Explorer CHAQUE racine séparément
        for root in root_classes:
            candidates = {root}
            path_scores = {root: 1.0}

            # --- Top-down exploration niveau par niveau ---
            for level in range(max_level):
                next_candidates = set()

                for parent in candidates:
                    if parent in hierarchy:
                        children = hierarchy[parent]

                        # Score enfants = score(parent) * similarité
                        child_scores = []
                        for child in children:
                            score = path_scores[parent] * similarities[child]
                            child_scores.append((child, score, similarities[child]))

                        # Garde top-k enfants pertinents
                        child_scores.sort(key=lambda x: x[1], reverse=True)
                        for child, path_score, sim in child_scores[:top_k_per_level]:
                            if sim >= min_similarity:
                                next_candidates.add(child)
                                path_scores[child] = path_score

                candidates = next_candidates
                if not candidates:
                    break

            # --- 3. Sélection des "core classes" ---
            high_sim_classes = [
                (c, similarities[c])
                for c in path_scores
                if c not in root_classes and similarities[c] >= min_similarity + 0.05
            ]
            high_sim_classes.sort(key=lambda x: x[1], reverse=True)

            confident_classes = []
            for c in path_scores:
                if c in root_classes:
                    continue
                sim_c = similarities[c]
                if sim_c < min_similarity:
                    continue

                parent = child2parent.get(c, None)
                max_competitor = 0
                if parent is not None:
                    max_competitor = similarities[parent]
                    if parent in hierarchy:
                        siblings = [s for s in hierarchy[parent] if s != c]
                        if siblings:
                            max_competitor = max(
                                max_competitor, max(similarities[s] for s in siblings)
                            )

                confidence = sim_c - max_competitor
                if confidence > 0.02:
                    confident_classes.append((c, sim_c, confidence))

            confident_classes.sort(key=lambda x: x[2], reverse=True)

            # --- 4. Combine les deux stratégies ---
            core_candidates = set()
            for c, sim in high_sim_classes[:max_cores]:
                core_candidates.add(c)
            for c, sim, conf in confident_classes[:max_cores]:
                core_candidates.add(c)

            # --- 5. Path completion ---
            for core in core_candidates:
                for anc in get_ancestors(core):
                    if anc not in root_classes:  # exclure toutes les racines
                        final_labels.add(anc)

        # --- 6. Sauvegarde des labels du doc ---
        if final_labels:
            silver_labels[pid] = sorted(list(final_labels))

    return silver_labels


print("\n🎯 Generating hierarchical multi-label silver labels...")
print("   Strategy: Dual approach (high similarity + confidence)")
silver_labels = generate_hierarchical_multilabel(top_k_per_level=5, min_similarity=0.25, max_cores=3)

# ==========================================================
# 🔧 ADJUST SILVER LABELS FOR KAGGLE (2–3 LABELS PER DOC)
# ==========================================================
print("\n🧩 Adjusting label counts to Kaggle format (2–3 per doc)...")

adjusted_labels = {}
for pid, labels in silver_labels.items():
    # Safety copy
    labels = list(labels)
    
    # Too few labels → add top similar classes
    if len(labels) < 2:
        idx = pid2idx[pid]
        doc_emb = X_train[idx]
        similarities = compute_similarity(doc_emb, label_emb, pid).cpu().numpy()
        best_candidates = np.argsort(similarities)[::-1]
        for c in best_candidates:
            if c not in labels and c not in root_classes:  # ✅ ici
                labels.append(int(c))
                if len(labels) >= 2:
                    break

    # Too many labels → keep top 3 by similarity
    elif len(labels) > 3:
        idx = pid2idx[pid]
        doc_emb = X_train[idx]
        similarities = compute_similarity(doc_emb, label_emb, pid).cpu().numpy()
        labels = sorted(labels, key=lambda x: similarities[x], reverse=True)[:3]

    adjusted_labels[pid] = labels

silver_labels = adjusted_labels

# Vérification de la distribution
import collections
dist = collections.Counter(len(v) for v in silver_labels.values())
print(f"📊 Label count distribution: {dict(dist)}")

# ==========================================================
# 🔍 Summary after adjustment
# ==========================================================
coverage = len(silver_labels) / len(train_ids)
avg_labels = np.mean([len(v) for v in silver_labels.values()]) if silver_labels else 0
print(f"\n✓ Adjusted silver labels:")
print(f"  Coverage: {len(silver_labels)}/{len(train_ids)} ({coverage:.1%})")
print(f"  Avg labels per doc: {avg_labels:.2f}")

# Show some examples
print("\n📋 Sample silver labels after adjustment:")
for i, (pid, labels) in enumerate(list(silver_labels.items())[:3]):
    print(f"\n  Doc {pid}: {id2text[pid][:100]}...")
    print(f"  Labels ({len(labels)}): {', '.join([classes[l] for l in labels])}")


print("\n🔍 Checking hierarchical consistency (same root branch)...")

def get_root(class_id):
    """Return the top-most ancestor (root) of a given class."""
    current = class_id
    while current in child2parent:
        current = child2parent[current]
    return current

same_root_count = 0
multi_root_count = 0

for pid, labels in silver_labels.items():
    if len(labels) <= 1:
        # Single label → trivially consistent
        same_root_count += 1
        continue

    # Get all root ancestors of the labels
    roots = {get_root(l) for l in labels}
    if len(roots) == 1:
        same_root_count += 1
    else:
        multi_root_count += 1

total_docs = len(silver_labels)
if total_docs > 0:
    consistency_ratio = same_root_count / total_docs * 100
else:
    consistency_ratio = 0.0

print(f"✓ Documents with all labels under the same root: {same_root_count}/{total_docs} "
      f"({consistency_ratio:.2f}%)")
print(f"✗ Documents with labels from multiple root branches: {multi_root_count}")


# Save silver labels
output_data = {
    "silver_labels": {str(k): v for k, v in silver_labels.items()},
    "min_similarity": 0.25,
    "top_k_per_level": 5,
    "max_cores": 3,
    "coverage": len(silver_labels) / len(train_ids),
    "avg_labels": avg_labels,
    "num_docs": len(silver_labels)
}

with open(SILVER_LABELS_PATH, "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=2)

print(f"\n✅ Silver labels saved to: {SILVER_LABELS_PATH}")
print(f"   Coverage: {len(silver_labels)}/{len(train_ids)} ({len(silver_labels)/len(train_ids):.1%})")

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
    """Dataset using precomputed embeddings"""
    def __init__(self, pid2label, pid2idx, embeddings, num_classes):
        self.pid2label = pid2label
        self.pid2idx = pid2idx
        self.embeddings = embeddings
        self.num_classes = num_classes

        if pid2label is not None:
            self.pids = list(pid2label.keys())
            self.has_labels = True
        else:
            self.pids = list(pid2idx.keys())
            self.has_labels = False

        self.indices = [pid2idx[pid] for pid in self.pids]

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, idx):
        emb = self.embeddings[self.indices[idx]]
        if self.has_labels:
            y = torch.zeros(self.num_classes, dtype=torch.float32)
            for label in self.pid2label[self.pids[idx]]:
                if label < self.num_classes:
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
