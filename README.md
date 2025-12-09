# Hierarchical Multi-Label Text Classification
2025/10/31 - 2025/12/20
Last Time written : 2025/12/09

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
├── Gold/ -> contain a csv with the 5 gold labels given by Kaggle and others docs   
├── Submission/ -> Final submission outputs (predictions, results)   
├── Models/ -> Saved Models     

### Additionnal documents
├── Pdf/  
├──├──final_project.pdf -> Assignment instructions   
├──├──main_paper.pdf -> Related research paper and main reference   
├──├──├──Report/  
├──├──├──├──report.pdf -> Final report for the project  
├── FormerVersion -> old documents  

### Model Notebooks
├── 1_Linear.ipynb -> MLP baseline model (~MLP only 1 linear layer)       
├── 2_GNN_ST.ipynb -> Graph-based GNN baseline + Self training  
├── 3_Ensemble_SelfT.ipynb -> Self-training on innerproduct baseline + ensemble methods  

### Silver Label Generation Notebook + some Model Artifacts
├── SilverGeneration/  
├── ├── Silver/ -> Silver-labeled datasets generated from keywords (with the use of hier embeddings and without)  
├── ├── SilverCombo/ -> Silver-labeled datasets combo (ensemble)  
├── ├── Embeddings/  -> Stored sentence / label embeddings    
├── V2_silver_generation_classic.ipynb -> silver generation with sentence transf embeddings   
├── V2_silver_generation_BM25.ipynb -> silver generation with sentence transf embeddings + BM25  
├── V2_silver_generation_multi.ipynb -> silver gen classic but => do not just take the most similar one + expand with hierarchy -> silver labels with more than 3 labels  
├── Combo_silver_generation.ipynb -> file that allows us to do ensemble techniques on csv/json predictions/silvers labels and create new silvers labels (upgraded)

### Core Files
├── .gitattributes  
├── .gitignore  
├── README.md  
├── requirements.txt  

## Requirements

Note : A GPU is recommended for the use of this project (I used Nvidia GeForce RTX 3060)

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
Each notebook use sentence-embedding model to match product reviews with their most relevant categories based on the class keywords and hierarchy.

You can generate the silver labels by simply running the 3 notebooks (V2 versions):
```
silver_generation_classic.ipynb
silver_generation_BM25.ipynb
silver_generation_multi.ipynb
```
To change of Sentence Model, you simply need to change the model_name variable (middle of the file) => ```e.g : model_name = "sentence-transformers/all-MiniLM-L6-v2"```  

Model differences :
- MPNet (recommended)
    - Best performance and most accurate silver labels
    - Slowest to compute
    - Use this one if you want the highest-quality labels (but it will takes too much time ~25 min) 
- MiniLM
    - Much faster
    - Slightly worse than MPNet
    - Good trade-off when you need speed (3 min max)

N.B : I tried also RoBERTa and ParaMini but it's not relevant for this task in our experiments => Lower performance compared to the two above  

3 types of Generation :
- Classic (only use the chosen model and 1 silver label list follow the heuristic "1 core class and 1/2 ancestors" => max 3 labels / min 2 labels) -> silver_generation_classic.ipynb
- Combination Classic with BM25 -> silver_generation_BM25.ipynb
- MultiClassic (only use the chosen model and 1 silver label list can have 2 core classes and at least 1 + some ancestors) -> silver_generation_multi.ipynb

Create Better Silver labels :  
You can significantly improve the quality of the silver labels by combining multiple JSON/CSV files produced by different models.
Using majority voting across models leads to more stable and more accurate labels.  

N.B : For more informations, the most performants is the combination of all the models, all predicted submissions or the combination of the 2 mpnet ones   
(see SilverCombo/ (v1 and v2 versions) => mpnet : combination of mpnet and mpnetBM (paths1) ; multi : combination of multi (mpnet) and multi (miniLM) (paths3) ; pseudo_gold_all : combination of all classic ones + mpnetBM with total = True "it means full vote and not juste global majority to keep the labels" (paths2); pseudo_silver_all : combination of all classic ones + mpnetBM with total = False "it means global majority to keep the labels >0.5" (paths2); pseudo_gold_ensemble : ensemble (total = True) on all submissions files generated by 1_Linear and 2_GNN_ST (paths4))  

See the notebook silver_generation_combo for the implementation details of the combined-label procedure.

To use the system quickly:
Choose one of the predefined dictionaries (paths, paths2, paths3, …) or create your own.

Set it in the loader:
- Choose one of the predefined dictionaries (paths, paths2, paths3, …) or create your own
- Set it in the loader: ```silvers = load_silver_files(paths3)```
- Select an output name: ```OUT_PATH = "SilverCombo/silver_train_new_clean_all.json"```
- Run the notebook to generate the merged silver labels

General Output :  
Each notebook produces Silver-labeled samples, saved in:
```
SilverGeneration/Silver/ or SilverCombo/ or SilverRemake/ (v2)
```

Embeddings for the training corpus and all label representations
(both hierarchical and non-hierarchical versions), saved in:

```
SilverGeneration/Embeddings/ or SilverGeneration/EmbeddingsRemake/ (v2)
```

These silver labels and embeddings are then used to train the hierarchical classifier.

Note1: Convention save (approximately)
- for embeddings of the corpus : X_train_test_{model_name}.pt
- for embeddings of the labels (with hierarchy) : labels_{model_name}.pt
- for embeddings of the corpus (without hierarchy): labels_base_{model_name}.pt
- for silver labels (with hierarchy): silver_train_{model_name}.json

But attention you need to change manually the output => see variables in the notebook : output_path_train, save_corpus, save_label_hier, save_label_nohier (change also in the output statistics variable (last cell) if you want some stats)  

Note2: If you do not want to lose time, all the .pt and .json files are already provided, and I confirm that I did not cheat in the creation of these documents. I did not try to imagine the good gold labels, I only used the 5 gold labels provided by kaggle and ensemble techniques (full majority vote on 4 predictions or 4 silver labels).

### Use Model for Submission
To generate the official submission files, we provide **3 training notebooks**.  
Each notebook trains the hierarchical classifier using a different configuration
(MLP, InnerProduct, self training, GNN, etc.).

List of submission notebooks :  
├── 1_Linear.ipynb -> MLP baseline model (~MLP only 1 linear layer)       
├── 2_GNN_ST.ipynb -> Graph-based GNN baseline + Self training  
├── 3_Ensemble_SelfT.ipynb -> Self-training on innerproduct baseline + ensemble methods 

Each notebook follows the same pipeline (sometimes it does it twice or more in the same notebook):
1. Load corpus embeddings  
2. Load label embeddings (with hierarchy or not)  
3. Load silver labels  
4. Train the model (classifier, training with loss, etc.)  
5. Generate a submission file inside the `Submission/` folder

To switch between configurations, you only need to change thesse parameters:
```
X_ALL_PATH = EMB_DIR / "XXX.pt" # Document embeddings (use mpnet for more results)
LABEL_EMB_PATH = EMB_DIR / "XXX.pt" # Label embeddings (use mpnet for more results)

json file (in with open() function )= "Silver/XXX.json" # for Silver labels (modify only the first notebook; the others are automatically generated based on the first one and the pipeline)
json file (in with open() function )= "Silver/XXX.json" # for Gold labels (this appears immediately after the Silver labels variable) => e.g :

with open("SilverGeneration/SilverCombo/pseudo_silver_all.json", "r", encoding="utf-8") as f:
    raw = json.load(f)
silver_labels = {int(pid): data["labels"] for pid, data in raw.items()}
silver_scores = {int(pid): data["scores"] for pid, data in raw.items()}

with open("SilverGeneration/SilverCombo/pseudo_silver_all.json", "r", encoding="utf-8") as f:
    raw = json.load(f)
gold_labels = {int(pid): data["labels"] for pid, data in raw.items()}
gold_scores = {int(pid): data["scores"] for pid, data in raw.items()}
```
Once these three paths are updated, you just run the entire notebook, and it will train the model + produce the submission file.

Note Important : Gold labels is generated from ensemble techniques and full majority vote on different json/csv prediction. If you do not want to use it (code will be less performant) but you can just give the same name file for the reader of gold labels and silver labels and you will have no problem. You just have after to change test_size of the first train_test_split() of the notebook if you do this : `gold_train, gold_val = train_test_split(gold_ids, test_size=0.33, random_state=42)`. Put a test size in order to have a good ratio train/val => 0.2 recommended

Note2: If you don’t want to lose time, all the .csv files are already provided, and I confirm that I did not cheat in the creation of these documents (format : submission_{model}.csv | if there is reg, it means "with regularization ->  weight decay, more dropout, etc" and notreg means "without", "st" means self training).  

Note3 : GNN notebook have 3 differents trainings and it can be quite long (10 mins) but you can do all the training separately and the first one give the best kaggle score. Linear and Ensemble are more in the 5 mins of duration.  

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
This work was conducted as part of an academic assignment for the Big Data Analysis course at Korea University fall 2025.  
It is provided for reference only and **should not be copied or reused** for other submissions.

## Useful Links
Task reference paper:  
https://aclanthology.org/2021.naacl-main.335.pdf

## GitHUB
https://github.com/Mushahime/20252R0136DATA30400.git
