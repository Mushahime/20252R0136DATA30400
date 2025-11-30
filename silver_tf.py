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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import re

device = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Default paths
ROOT = Path("Amazon_products")
TRAIN_DIR = ROOT / "train"
TEST_DIR = ROOT / "test"

TEST_CORPUS_PATH = os.path.join(TEST_DIR, "test_corpus.txt")
TRAIN_CORPUS_PATH = os.path.join(TRAIN_DIR, "train_corpus.txt")

CLASS_HIERARCHY_PATH = ROOT / "class_hierarchy.txt" 
CLASS_RELATED_PATH = ROOT / "class_related_keywords.txt" 
CLASS_PATH = ROOT / "classes.txt" 

SUBMISSION_PATH = "Submission/submission.csv"

# --- Constants ---
NUM_CLASSES = 531
MIN_LABELS = 1
MAX_LABELS = 3

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

# --- Utility functions ---
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
    Expand a list of core labels by adding ALL their ancestors.
    This guarantees 100% hierarchy consistency.
    """
    expanded = set(labels)
    stack = list(labels)

    # Build reverse parent → children mapping
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

def preprocess_text(text):
    """
    Cleans text by removing special characters and normalizing whitespace.
    """
    cleaned = re.sub(r"[>&]", " ", text)
    cleaned = re.sub(r"[^a-zA-Z0-9 ]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

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

def propagate_hierarchy_tfidf(
    label_tfidf,
    class_hierarchy,
    alpha=0.7,
    include_children=False,
    normalize=True
):
    """
    Propagate hierarchy information through TF-IDF vectors.
    This is done on dense numpy arrays for flexibility.
    """
    # Convert sparse to dense for manipulation
    label_dense = label_tfidf.toarray() if hasattr(label_tfidf, 'toarray') else label_tfidf
    num_classes = label_dense.shape[0]
    updated = label_dense.copy()
    
    # Pass 1: Parents → Children
    for class_id in range(num_classes):
        class_id_str = str(class_id)
        
        if class_id_str not in class_hierarchy:
            continue
        
        parents = class_hierarchy[class_id_str].get("parents", [])
        valid_parents = [p for p in parents if 0 <= p < num_classes]
        
        if valid_parents:
            parent_vec = label_dense[valid_parents].mean(axis=0)
            updated[class_id] = (1 - alpha) * label_dense[class_id] + alpha * parent_vec
    
    # Pass 2: Children → Parents
    if include_children:
        temp = updated.copy()
        for class_id in range(num_classes):
            class_id_str = str(class_id)
            
            if class_id_str not in class_hierarchy:
                continue
            
            children = class_hierarchy[class_id_str].get("children", [])
            valid_children = [c for c in children if 0 <= c < num_classes]
            
            if valid_children:
                children_vec = updated[valid_children].mean(axis=0)
                temp[class_id] = (1 - alpha) * updated[class_id] + alpha * children_vec
        
        updated = temp
    
    # Normalize
    if normalize:
        norms = np.linalg.norm(updated, axis=1, keepdims=True)
        updated = updated / (norms + 1e-8)
    
    return updated

def convert(o):
    import numpy as np
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o

def generate_silver_labels_TFIDF(
    train_texts,
    train_ids,
    test_texts,
    test_ids,
    id2class,
    class2related,
    class_hierarchy,
    output_path_train="Silver/silver_train_tfidf.json",
    output_path_test="Silver/silver_test_tfidf.json"
):
    
    print("\n🔧 Preprocessing texts...")
    all_texts = train_texts + test_texts
    all_ids = train_ids + test_ids
    
    # Preprocess all texts
    all_texts_clean = [preprocess_text(text) for text in tqdm(all_texts, desc="Preprocessing")]
    
    # Enriched categories
    enriched_categories = [
        get_enriched_category_with_hierarchy(i, id2class, class2related, class_hierarchy)
        for i in tqdm(range(531), desc="Enriching categories")
    ]
    enriched_categories_clean = [preprocess_text(cat) for cat in enriched_categories]
    
    print(f"\nBuilding TF-IDF vectorizer on {len(all_texts_clean)} documents + {len(enriched_categories_clean)} categories...")
    
    # Fit TF-IDF on concatenated corpus (products + categories)
    vectorizer = TfidfVectorizer(max_features=5000)
    combined_corpus = all_texts_clean + enriched_categories_clean
    tfidf_matrix = vectorizer.fit_transform(combined_corpus)
    
    n_docs = len(all_texts_clean)
    doc_tfidf = tfidf_matrix[:n_docs]
    label_tfidf_base = tfidf_matrix[n_docs:]
    
    print(f"TF-IDF matrix: {tfidf_matrix.shape}")
    print(f"Documents: {doc_tfidf.shape}")
    print(f"Labels: {label_tfidf_base.shape}")
    
    # Build hierarchy structure for propagation
    hierarchy_int = {}
    for cid, rel in class_hierarchy.items():
        parents = rel.get("parents", []) if isinstance(rel, dict) else []
        children = rel.get("children", []) if isinstance(rel, dict) else rel if isinstance(rel, list) else []
        hierarchy_int[cid] = {"parents": parents, "children": children}
    
    print("\nPropagating hierarchy through TF-IDF vectors...")
    label_tfidf_hierarchical = propagate_hierarchy_tfidf(
        label_tfidf=label_tfidf_base,
        class_hierarchy=hierarchy_int,
        alpha=0.7,
        include_children=False,
        normalize=True
    )
    
    # Compute similarities
    print("\nComputing similarities...")
    all_similarities = cosine_similarity(doc_tfidf, label_tfidf_hierarchical)
    all_similarities2 = cosine_similarity(doc_tfidf, label_tfidf_base)
    
    print(f"Similarity matrices: {all_similarities.shape}")
    
    # Generate silver labels
    silver_train, silver_test = {}, {}
    silver_train_nohier, silver_test_nohier = {}, {}
    
    n_train = len(train_ids)
    
    for idx, rid in enumerate(tqdm(all_ids, desc="Assigning labels")):
        
        # With hierarchy
        sims = all_similarities[idx]
        top_idx = np.argmax(sims)
        top_score = sims[top_idx]
        
        # Expand with hierarchy
        expanded = expand_with_hierarchy([top_idx], class_hierarchy)
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
        
        # Sigmoid for probs (though cosine sim is already [0,1])
        final_probs = [1 / (1 + np.exp(-s)) for s in final_scores]
        
        record = {
            "labels": final_labels,
            "scores": final_scores,
            "probs": final_probs
        }
        
        if idx < n_train:
            silver_train[rid] = record
        else:
            silver_test[rid] = record
        
        # Without hierarchy
        sims2 = all_similarities2[idx]
        top_idx2 = np.argmax(sims2)
        
        expanded2 = expand_with_hierarchy([top_idx2], class_hierarchy)
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
            "probs": [1 / (1 + np.exp(-s)) for s in sorted_scores2]
        }
        
        if idx < n_train:
            silver_train_nohier[rid] = record_nohier
        else:
            silver_test_nohier[rid] = record_nohier
    
    # Save results
    os.makedirs("Silver", exist_ok=True)

    json.dump(
        silver_train,
        open(output_path_train, "w", encoding="utf-8"),
        indent=2,
        ensure_ascii=False,
        default=convert
    )

    json.dump(
        silver_test,
        open(output_path_test, "w", encoding="utf-8"),
        indent=2,
        ensure_ascii=False,
        default=convert
    )

    json.dump(
        silver_train_nohier,
        open("Silver/silver_train_tfidf_nohier.json", "w", encoding="utf-8"),
        indent=2,
        ensure_ascii=False,
        default=convert
    )

    json.dump(
        silver_test_nohier,
        open("Silver/silver_test_tfidf_nohier.json", "w", encoding="utf-8"),
        indent=2,
        ensure_ascii=False,
        default=convert
    )
    
    return silver_train, silver_test, silver_train_nohier, silver_test_nohier


# Execute
print("\n" + "="*50)
print("GENERATING SILVER LABELS WITH TF-IDF")
print("="*50)

silver_train_safe, silver_test_safe, silver_train_safe_nohier, silver_test_safe_nohier = generate_silver_labels_TFIDF(
    list(id2text_train.values()),
    list(id2text_train.keys()),
    list(id2text_test.values()),
    list(id2text_test.keys()),
    id2class,
    class2related,
    class2hierarchy,
    output_path_train="Silver/silver_train_tfidf.json",
    output_path_test="Silver/silver_test_tfidf.json"
)

# Stats
print()
label_stats("TF-IDF Train (with hierarchy)", silver_train_safe)

silver_train_labels_only = {
    pid: info["labels"]
    for pid, info in silver_train_safe.items()
}

label_stats("TF-IDF Train (no hierarchy)", silver_train_safe_nohier)

silver_train_labels_only_nohier = {
    pid: info["labels"]
    for pid, info in silver_train_safe_nohier.items()
}

consistency = hierarchy_consistency(silver_train_labels_only, class2hierarchy)
print(f"\nHierarchy Consistency (with): {consistency:.2%}")

consistency = hierarchy_consistency(silver_train_labels_only_nohier, class2hierarchy)
print(f"\nHierarchy Consistency (without): {consistency:.2%}")

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
print(f"\nCoverage (with hierarchy): {coverage:.2%}")
print(f"Covered classes: {len(classes)}/{531}")

coverage, classes = label_coverage(silver_train_labels_only_nohier)
print(f"Coverage (without hierarchy): {coverage:.2%}")
print(f"Covered classes: {len(classes)}/{531}")