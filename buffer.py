import random
import numpy as np
import torch
import json
from tqdm import tqdm
from pathlib import Path
from utils import * 
import copy
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
import os
import csv
from tqdm import tqdm
from transformers import RobertaModel, RobertaTokenizer
from tqdm import tqdm
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
from sentence_transformers import SentenceTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Default paths
ROOT = Path("Amazon_products") # Root Amazon_products directory
TRAIN_DIR = ROOT / "train"
TEST_DIR = ROOT / "test"

TEST_CORPUS_PATH = os.path.join(TEST_DIR, "test_corpus.txt")  # product_id \t text
TRAIN_CORPUS_PATH = os.path.join(TRAIN_DIR, "train_corpus.txt")

CLASS_HIERARCHY_PATH = ROOT / "class_hierarchy.txt" 
CLASS_RELATED_PATH = ROOT / "class_related_keywords.txt" 
CLASS_PATH = ROOT / "classes.txt" 

SUBMISSION_PATH = "Submission/submission.csv"  # output file

# --- Constants ---
NUM_CLASSES = 531  # total number of classes (0–530)
MIN_LABELS = 1     # minimum number of labels per sample
MAX_LABELS = 3     # maximum number of labels per sample

# --- Load ---
def load_corpus(path):
    """Load test corpus into {id: text} dictionary."""
    id2text = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                id, text = parts
                id2text[id] = text
    return id2text

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

def load_class_keywords(path):
    """Load class keywords into {class_name: [keywords]} dictionary."""
    class2keywords = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if ":" not in line:
                continue
            classname, keywords = line.strip().split(":", 1)
            keyword_list = [kw.strip() for kw in keywords.split(",") if kw.strip()]
            class2keywords[classname] = keyword_list
    return class2keywords

id2text_test = load_corpus(TEST_CORPUS_PATH)
id_list_test = list(id2text_test.keys())

id2text_train = load_corpus(TRAIN_CORPUS_PATH)
id_list_train = list(id2text_train.keys())

id2class = load_corpus(CLASS_PATH)
class2hierarchy = load_multilabel(CLASS_HIERARCHY_PATH)
class2related = load_class_keywords(CLASS_RELATED_PATH)

def label_stats(name, silver):
    counts = [len(v) for v in silver.values()]
    print(f"\n{name}")
    print(f"  Documents: {len(counts)}")
    print(f"  Avg labels/doc: {np.mean(counts):.2f}")
    print(f"  Min labels: {np.min(counts)}")
    print(f"  Max labels: {np.max(counts)}")

def hierarchy_consistency(silver, hierarchy):
    ok = total = 0
    for labels in silver.values():
        L = set(labels)
        for parent, children in hierarchy.items():
            for child in children:
                if child in L:
                    total += 1
                    if parent in L:
                        ok += 1
    return ok / total if total > 0 else 0

def count_present_classes(silver, total_classes=531):
    # Collect all unique labels appearing in the dataset
    all_labels = set(label for labels in silver.values() for label in labels)
    
    # Count how many distinct classes are present
    n_present = len(all_labels)
    
    print(f"Present classes: {n_present}/{total_classes} ({n_present/total_classes*100:.2f}%)")
    return n_present

from collections import Counter
def analyze_coverage(silver, name):
    all_labels = []
    for info in silver.values():
        all_labels.extend(info)
    
    unique = len(set(all_labels))
    counter = Counter(all_labels)
    top5 = counter.most_common(5)
    
    print(f"\n{name}:")
    print(f"  Coverage: {unique}/531 ({unique/531*100:.1f}%)")
    print(f"  Top-5 most frequent:")
    for cls, count in top5:
        print(f"    Class {cls}: {count} times ({count/len(silver)*100:.1f}%)")

def expand_with_hierarchy(labels, hierarchy):
    """
    Expand a list of core labels by adding ALL their ancestors
    (parents, parents of parents, etc.), recursively.
    This guarantees 100% hierarchy consistency.
    """
    expanded = set(labels)
    stack = list(labels)

    # Build reverse parent → children mapping
    # hierarchy = { parent: [children] }
    # We need the reverse: child → parents
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
                stack.append(parent)   # continue climbing up

    return sorted(expanded)[-3:]


def propagate_hierarchy_simple(
    label_embeddings,
    class_hierarchy,
    alpha=0.7,
    include_children=False,
    normalize=True
):
    device = label_embeddings.device
    num_classes = label_embeddings.shape[0]
    updated = label_embeddings.clone()
    
    # Pass 1: Parents → Children
    for class_id in range(num_classes):
        class_id_str = str(class_id)
        
        if class_id_str not in class_hierarchy:
            continue
        
        parents = class_hierarchy[class_id_str].get("parents", [])
        valid_parents = [p for p in parents if 0 <= p < num_classes]
        
        if valid_parents:
            parent_vec = label_embeddings[valid_parents].mean(dim=0)
            updated[class_id] = (1 - alpha) * label_embeddings[class_id] + alpha * parent_vec
    
    # Pass 2: Children → Parents
    if include_children:
        temp = updated.clone()
        for class_id in range(num_classes):
            class_id_str = str(class_id)
            
            if class_id_str not in class_hierarchy:
                continue
            
            children = class_hierarchy[class_id_str].get("children", [])
            valid_children = [c for c in children if 0 <= c < num_classes]
            
            if valid_children:
                children_vec = updated[valid_children].mean(dim=0)
                temp[class_id] = (1 - alpha) * updated[class_id] + alpha * children_vec
        
        updated = temp
    
    # Normalize
    if normalize:
        norms = torch.norm(updated, dim=1, keepdim=True)
        updated = updated / (norms + 1e-8)
    
    return updated


def get_embeddings(texts, model, batch_size=64, save_path=None, force_recompute=False):

    # Load cache
    if save_path and os.path.exists(save_path) and not force_recompute:
        print(f"📦 Loading from {save_path}")
        emb = torch.load(save_path, map_location="cpu")
        if isinstance(emb, np.ndarray):
            emb = torch.from_numpy(emb)
        return emb

    print(f"⚙️ Encoding {len(texts)} texts on {model.device}...")

    # ---- ENCODE ----
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,  
        device=model.device
    )

    # ---- SAFETY CHECK ----
    if isinstance(emb, list):
        emb = torch.stack([e for e in emb])  

    elif isinstance(emb, np.ndarray):
        emb = torch.from_numpy(emb)

    # ---- CPU for saving ----
    emb = emb.cpu()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(emb, save_path)
        print(f"Saved to {save_path}")

    return emb

def get_enriched_category_with_hierarchy(class_id, id2class, class2related, class_hierarchy, max_keywords=10):
    class_name = id2class[str(class_id)]
    clean_name = class_name.replace('_', ' ')
    
    # Parents
    parents = class_hierarchy.get(str(class_id), {}).get("parents", [])
    parent_names = []
    for p in parents:
        if 0 <= p < 531:
            parent_name = id2class[str(p)].replace('_', ' ')
            if parent_name.lower() != "root":
                parent_names.append(parent_name)
    
    # Keywords
    keywords = class2related.get(class_name, [])[:max_keywords]
    
    # Combine
    parts = [clean_name]
    if parent_names:
        parts.extend(parent_names)
    if keywords:
        parts.extend(keywords)
    
    return " ".join(parts)

model_name = "paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(model_name)
model = model.to(device)


def generate_silver_labels_FAST(
    train_texts,
    train_ids,
    test_texts,
    test_ids,
    id2class,
    class2related,
    tokenizer,
    model,
    class_hierarchy,
    output_path_train="Silver/silver_train_new_para.json",
    output_path_test="Silver/silver_test_new_para.json"
):
    
    all_texts = train_texts + test_texts
    all_ids = train_ids + test_ids

    enriched_categories = [
        get_enriched_category_with_hierarchy(i, id2class, class2related, class_hierarchy)
        for i in tqdm(range(531), desc="Enriching")
    ]

    base_category_embeddings = get_embeddings(
        enriched_categories,
        model=model,
        batch_size=64,
        save_path="Embeddings/labels_base_new_para.pt",
        force_recompute=True
    )

    hierarchy_int = {}
    for cid, rel in class_hierarchy.items():
        parents = rel.get("parents", []) if isinstance(rel, dict) else []
        children = rel.get("children", []) if isinstance(rel, dict) else rel if isinstance(rel, list) else []
        hierarchy_int[cid] = {"parents": parents, "children": children}

    # Ensure tensor format
    if isinstance(base_category_embeddings, list):
        print("base_category_embeddings is list → stacking")
        base_category_embeddings = torch.stack(base_category_embeddings)
    elif isinstance(base_category_embeddings, np.ndarray):
        print("base_category_embeddings is numpy → converting")
        base_category_embeddings = torch.from_numpy(base_category_embeddings)
    elif isinstance(base_category_embeddings, torch.Tensor):
        print("base_category_embeddings is already torch")
    else:
        raise TypeError(f"Unexpected type: {type(base_category_embeddings)}")
    
    hierarchical_embeddings = propagate_hierarchy_simple(
        label_embeddings=base_category_embeddings,
        class_hierarchy=hierarchy_int,
        alpha=0.7,
        include_children=False,
        normalize=True
    )

    torch.save(hierarchical_embeddings, "Embeddings/labels_hierarchical_new_para.pt")

    review_embeddings = get_embeddings(
        all_texts,
        model=model,
        batch_size=64,
        save_path="Embeddings/X_train_test_para.pt",
        force_recompute=True
    )

    # Ensure both are tensors
    if isinstance(review_embeddings, np.ndarray):
        review_embeddings = torch.from_numpy(review_embeddings)
    if isinstance(hierarchical_embeddings, np.ndarray):
        hierarchical_embeddings = torch.from_numpy(hierarchical_embeddings)

    # Move to device for computation
    review_embeddings = review_embeddings.to(device)
    hierarchical_embeddings = hierarchical_embeddings.to(device)

    # Compute similarity on device
    all_similarities = torch.matmul(
        review_embeddings,
        hierarchical_embeddings.T
    )

    # Move back to CPU for numpy operations
    all_similarities = all_similarities.cpu()

    all_similarities2 = torch.matmul(
    review_embeddings,
    base_category_embeddings.to(device).T  # raw original embeddings
    )
    all_similarities2 = all_similarities2.cpu()

    silver_train, silver_test = {}, {}
    silver_train_nohier, silver_test_nohier = {}, {}

    n_train = len(train_ids)

    for idx, rid in enumerate(tqdm(all_ids, desc="Assigning")):

        sims = all_similarities[idx]
        topk_scores, topk_idx = torch.topk(sims, k=1)
        
        topk_idx = topk_idx.tolist()
        topk_scores = topk_scores.tolist()

        expanded = expand_with_hierarchy(topk_idx, class_hierarchy)

        expanded_scores = [float(sims[l]) for l in expanded]

        sorted_labels = [
            x for x, _ in sorted(
                zip(expanded, expanded_scores),
                key=lambda t: t[1],
                reverse=True
            )
        ]

        sorted_scores = [
            x for _, x in sorted(
                zip(expanded, expanded_scores),
                key=lambda t: t[1],
                reverse=True
            )
        ]

        final_labels = sorted_labels
        final_scores = sorted_scores
        final_probs = torch.sigmoid(torch.tensor(final_scores)).tolist()

        record = {
            "labels": final_labels,
            "scores": final_scores,
            "probs": final_probs
        }

        if idx < n_train:
            silver_train[rid] = record
        else:
            silver_test[rid] = record


        sims2 = all_similarities2[idx]
        topk_scores2, topk_idx2 = torch.topk(sims2, k=1)
        topk_idx2 = topk_idx2.tolist()

        expanded2 = expand_with_hierarchy(topk_idx2, class_hierarchy)
        expanded_scores2 = [float(sims2[l]) for l in expanded2]

        sorted_labels2 = [
            x for x, _ in sorted(
                zip(expanded2, expanded_scores2),
                key=lambda t: t[1],
                reverse=True
            )
        ]

        sorted_scores2 = [
            x for _, x in sorted(
                zip(expanded2, expanded_scores2),
                key=lambda t: t[1],
                reverse=True
            )
        ]

        record_nohier = {
            "labels": sorted_labels2,
            "scores": sorted_scores2,
            "probs": torch.sigmoid(torch.tensor(sorted_scores2)).tolist()
        }

        if idx < n_train:
            silver_train_nohier[rid] = record_nohier
        else:
            silver_test_nohier[rid] = record_nohier

    os.makedirs("Silver", exist_ok=True)

    json.dump(silver_train, open(output_path_train, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    json.dump(silver_test, open(output_path_test, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    json.dump(silver_train_nohier,
          open("Silver/silver_train_new_para_nohier.json", "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)

    json.dump(silver_test_nohier,
            open("Silver/silver_test_new_para_nohier.json", "w", encoding="utf-8"),
            indent=2, ensure_ascii=False)

    return silver_train, silver_test, silver_train_nohier, silver_test_nohier


# Exec
print("\n" + "="*50)
print("GENERATING TRAIN SILVER LABELS (FAST)")
print("="*50)

silver_train_safe, silver_test_safe, silver_train_safe_nohier, silver_test_safe_nohier = generate_silver_labels_FAST(
    list(id2text_train.values()),
    list(id2text_train.keys()),
    list(id2text_test.values()),
    list(id2text_test.keys()),
    id2class,
    class2related,
    None,
    model,
    class2hierarchy,
    output_path_train="Silver/silver_train_new_para.json",
    output_path_test="Silver/silver_test_new_para.json"
)

# Stats
print()
label_stats("Safe Train", silver_train_safe)

silver_train_labels_only = {
    pid: info["labels"]
    for pid, info in silver_train_safe.items()
}

label_stats("Safe Train", silver_train_safe_nohier)

silver_train_labels_only_nohier = {
    pid: info["labels"]
    for pid, info in silver_train_safe_nohier.items()
}


consistency = hierarchy_consistency(silver_train_labels_only, class2hierarchy)
print(f"\nHierarchy Consistency: {consistency:.2%}")

consistency = hierarchy_consistency(silver_train_labels_only_nohier, class2hierarchy)
print(f"\nHierarchy Consistency: {consistency:.2%}")


def label_coverage(silver_labels, num_classes=531):
    """
    silver_labels : { review_id: [label1, label2, ...] }
    returns coverage_ratio, covered_classes
    """
    covered = set()

    for _, labels in silver_labels.items():
        for lbl in labels:
            if 0 <= lbl < num_classes:
                covered.add(lbl)

    coverage_ratio = len(covered) / num_classes
    return coverage_ratio, sorted(list(covered))

coverage, classes = label_coverage(silver_train_labels_only)
print(f"Coverage: {coverage:.2%}")
print(f"Covered classes: {len(classes)}/{531}")

coverage, classes = label_coverage(silver_train_labels_only_nohier)
print(f"Coverage: {coverage:.2%}")
print(f"Covered classes: {len(classes)}/{531}")

