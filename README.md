# Hierarchical Multi-Label Text Classification
2025/10/31 - 2025/12/20

## Overview

This project addresses a **Hierarchical Multi-Label Text Classification** task. The objective is to classify product reviews into multiple relevant categories, while **no ground-truth labels are provided**. Each document is associated with **two to three product categories**, which are organized in a **directed acyclic hierarchical taxonomy of 531 classes**.

We are given:
- 29,487 unlabeled text reviews
- 19,658 reviews for prediction
- A taxonomy file defining parent–child relationships between categories
- A list of keywords associated with each category

Since we do not have real labels for training, we create silver labels using the class-related keywords and the hierarchical structure of the taxonomy associated to sentance transformer model. These labels are then used to train a hierarchical multi-label classification model that can predict the relevant product categories for each review while respecting parent–child dependencies.

The project evaluates different learning strategies such as self-training, pseudo-labeling, and graph-based methods, and compares their impact on performance. Finally, examples of both correct and incorrect predictions are analyzed to better understand the model’s behavior (see Report/).

## Repository Structure

### Root
project_release/  

### Dataset
├── Amazon_products/  
├──├──test/  
├──├──├──test_corpus.txt -> Raw test corpus (product reviews)   
├──├──train/  
├──├──├──train_corpus.txt -> Raw training corpus  
├──├──classes.txt -> List of class IDs / categories  
├──├──class_related_keywords.txt -> keywords related at each class (used for silver labeling)  
├──├──class_hierarchy.txt -> Parent–child hierarchical taxonomy  

### Model Artifacts
├── Embeddings/ -> Stored sentence / label embeddings    
├── Silver/ -> Silver-labeled datasets generated from keywords  
├── Gold/ -> contain a csv with the 5 gold labels given by Kaggle   
├── Submission/ -> Final submission outputs (predictions, results)   
├── Models/ -> Saved Models     

### Additionnal documents
├── Pdf/  
├──├──final_project.pdf -> Assignment instructions   
├──├──main_paper.pdf -> Related research paper and main reference   
├──├──├──Report/  
├──├──├──├──report.pdf -> Final report for the project  

### Model Notebooks
├── classicGNN.ipynb -> Graph-based GNN baseline  
├── classicMLP.ipynb -> MLP baseline model  
├── innerproduct.ipynb -> Inner-product classifier        
├── selfGNN.ipynb -> Self-training + GNN version  
├── selftraining.ipynb -> Self-training pipeline (no GNN)   

### Silver Label Generation Notebook
├── silver_generation_miniML.ipynb -> MiniLM embeddings  
├── silver_generation_mpnet.ipynb -> MPNet embeddings    
├── silver_generation_roberta.ipynb -> RoBERTa embeddings  

### Core Files
├── .gitattributes  
├── .gitignore  
├── README.md  
├── requirements.txt  
└── utils.py (useless)

## Requirements

Note : A GPU is recommended for the use of this project

### 0. Cloning repo (with git LFS)
```
git lfs install
git clone https://github.com/Mushahime/20252R0136DATA30400
cd 20252R0136DATA30400
```

### 1. Install Python 3.9+  
Download from: https://www.python.org/downloads/

Check: ```python --version```

### 2. Create a virtual environment
```python -m venv venv```

### 3. Activate it

macOS / Linux  
```source venv/bin/activate```

Windows  
```venv\Scripts\activate```

### 4. Install dependencies  
```pip install -r requirements.txt```

## How to run the project and reproduce the results

### Generate Silver Labels
The silver labels are produced by running the silver-labeling notebooks.
Each notebook uses a different sentence-embedding model to match product reviews with their most relevant categories based on the class keywords and hierarchy.

You can generate the silver labels by simply running the notebooks:
```
silver_generation_miniLM.ipynb
silver_generation_mpnet.ipynb
silver_generation_roberta.ipynb
```

Model differences :
- MPNet (recommended)
    - Best performance and most accurate silver labels
    - Slowest to compute
    - Use this one if you want the highest-quality labels  
- MiniLM
    - Much faster
    - Slightly worse than MPNet
    - Good trade-off when you need speed
- RoBERTa
    - Not relevant for this task in our experiments
    - Lower performance compared to the two above

Output :  
Each notebook produces Silver-labeled samples, saved in:
```
Silver/
```

Embeddings for the training corpus and all label representations
(both hierarchical and non-hierarchical versions), saved in:

```
Embeddings/
```

These silver labels and embeddings are then used to train the hierarchical classifier.

Note1: 
- for embeddings of the corpus : X_train_test_{model_name}.pt
- for embeddings of the labels (with hierarchy) : labels_{model_name}.pt
- for embeddings of the corpus (without hierarchy): labels_base_{model_name}.pt
- for silver labels (without hierarchy): silver_train_new_{model_name}_nohier.json
- for silver labels (with hierarchy): silver_train_new_{model_name}.json  

Note2: If you don’t want to lose time, all the .pt and .json files are already provided, and I confirm that I did not cheat in the creation of these documents.  

### Use Model for Submission
To generate the official submission files, we provide **6 training notebooks**.  
Each notebook trains the hierarchical classifier using a different configuration
(MLP, InnerProduct, self training, GNN, etc.).

List of submission notebooks :
- `MLP_GNN.ipynb` -> GNN + MLP linear proj
- `classicGNN.ipynb` -> GNN
- `classicMLP.ipynb` -> MLP (2 linear layers)
- `innerproduct.ipynb` -> InnerProduct Classifier
- `classicMIX.ipynb` -> Mix of GNN and innerprodtuct (ensemble)
- `classicselftraining.ipynb` -> MLP + self training

Each notebook follows the same pipeline:
1. Load embeddings  
2. Load label embeddings  
3. Load silver labels  
4. Train the model (classifier, training with loss, etc.)  
5. Generate a submission file inside the `Submission/` folder

To switch between configurations, you only need to change three parameters:
```
X_ALL_PATH = EMB_DIR / "XXX.pt" # Document embeddings
LABEL_EMB_PATH = EMB_DIR / "XXX.pt" # Label embeddings
json file (in with open() function )= "Silver/XXX.json" # Silver labels
```
Once these three paths are updated, you just run the entire notebook, and it will train the model + produce the submission file.

Note: If you don’t want to lose time, all the .csv files are already provided, and I confirm that I did not cheat in the creation of these documents (format : submission_{model}.csv | if there is reg, it means "with regularization -> temporal ensemble, weight decay, more dropout, etc" and notreg means "without").  

## Reproducibility

To reproduct the same results, we use a seed : 

```py
import random, numpy as np, torch
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
```

## Submissions format

For each id in the test set, we predict a label for the target id(0~19657). The submission file contain a header and have the following format:

`id, label 0, 3,21 etc.`

## Author
Name: Noam CATHERINE
Student ID: 2025952809
Course: DATA304 Big Data Analysis

## Credits
This work was conducted as part of an academic assignment for the Big Data Analysis course at Korea University.  
It is provided for reference only and **should not be copied or reused** for other submissions.

## Useful Links
Task reference paper:  
https://aclanthology.org/2021.naacl-main.335.pdf

## GitHUB
https://github.com/Mushahime/20252R0136DATA30400.git
