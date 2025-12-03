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
├── Gold/ -> contain a csv with the 5 gold labels given by Kaggle and others docs   
├── Submission/ -> Final submission outputs (predictions, results)   
├── Models/ -> Saved Models     

### Additionnal documents
├── Pdf/  
├──├──final_project.pdf -> Assignment instructions   
├──├──main_paper.pdf -> Related research paper and main reference   
├──├──├──Report/  
├──├──├──├──report.pdf -> Final report for the project  

### Model Notebooks
├── 1_Linear.ipynb -> MLP baseline model (~MLP only 1 linear layer)  
├── 2_Innerproduct.ipynb -> Inner-product classifier        
├── 3_GNN.ipynb -> Graph-based GNN baseline   
├── 4_Selftraining.ipynb -> Self-training on innerproduct baseline  
├── 5_MLP_GNN.ipynb -> Graph-based GNN baseline + MLP projection before enter in the GNN   
├── 6_MIX_ensemble.ipynb -> Ensemble baseline (GNN + Innerproduct baseline) 

### Silver Label Generation Notebook + some Model Artifacts
├── SilverGeneration/  
├── ├── Silver/ -> Silver-labeled datasets generated from keywords (with the use of hier embeddings and without)  
├── ├── SilverCombo/ -> Silver-labeled datasets combo (ensemble)
├── ├── Embeddings/  -> Stored sentence / label embeddings    
├── silver_generation_mini.ipynb -> MiniLM embeddings  
├── silver_generation_mpnet.ipynb -> MPNet embeddings    
├── silver_generation_miniBM25.ipynb -> MiniLM embeddings + BM25  
├── silver_generation_mpnetBM25.ipynb -> MPNet embeddings + BM25   
├── silver_generation_multiMini.ipynb -> MiniLM embeddings (do not just take the most similar one + expand with hierarchy) -> silver labels with more than 3 labels  
├── silver_generation_multiMpnet.ipynb -> MPNet embeddings (do not just take the most similar one + expand with hierarchy) -> silver labels with more than 3 labels  
├── silver_generation_combo.ipynb -> File that allows us to combine multiple json together (ensemble json)   

### Core Files
├── .gitattributes  
├── .gitignore  
├── README.md  
├── requirements.txt  

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

You can generate the silver labels by simply running the 6 notebooks:
```
silver_generation_miniLM.ipynb
silver_generation_mpnet.ipynb
silver_generation_miniBM25.ipynb
silver_generation_mpnetBM25.ipynb
silver_generation_multiMini.ipynb
silver_generation_multiMpnet.ipynb
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

N.B : I tried also RoBERTa but it's not relevant for this task in our experiments => Lower performance compared to the two above  

3 types of Generation :
- Classic (only use the chosen model and 1 silver label list follow the heuristic "1 core class and 1/2 ancestors" => max 3 labels / min 2 labels) -> e.g : silver_generation_miniLM.ipynb
- Combination Classic with BM25 -> e.g : silver_generation_miniBM25.ipynb
- MultiClassic (only use the chosen model and 1 silver label list can have 2 core classes and at least 1 + some ancestors) -> e.g : silver_generation_multiMini.ipynb

N.B : The most performants is the combination of all the classic model + BM25, the combination of the 2 mpnet ones.

Create Better Silver labels :  
You can significantly improve the quality of the silver labels by combining multiple JSON files produced by different models.
Using majority voting across models leads to more stable and more accurate labels.  

See the notebook silver_generation_combo for the implementation details of the combined-label procedure.

To use the system quickly:
Choose one of the predefined dictionaries (paths, paths2, paths3, …) or create your own.

Set it in the loader:
- Choose one of the predefined dictionaries (paths, paths2, paths3, …) or create your own
- Set it in the loader: ```silvers = load_silver_files(paths3)```
- Select an output name: ```OUT_PATH = "SilverCombo/silver_train_new_clean_all.json"```
- Run the notebook to generate the merged silver labels

N.B : The most performant is the combination of classic Mpnet + Mpnet with BM25.

General Output :  
Each notebook produces Silver-labeled samples, saved in:
```
SilverGeneration/Silver/ or SilverCombo/
```

Embeddings for the training corpus and all label representations
(both hierarchical and non-hierarchical versions), saved in:

```
SilverGeneration/Embeddings/
```

These silver labels and embeddings are then used to train the hierarchical classifier.

Note1: 
- for embeddings of the corpus : X_train_test_{model_name}.pt
- for embeddings of the labels (with hierarchy) : labels_{model_name}.pt
- for embeddings of the corpus (without hierarchy): labels_base_{model_name}.pt
- for silver labels (without hierarchy): silver_train_{model_name}_nohier.json
- for silver labels (with hierarchy): silver_train_{model_name}.json  

Note2: If you do not want to lose time, all the .pt and .json files are already provided, and I confirm that I did not cheat in the creation of these documents.  

### Use Model for Submission
To generate the official submission files, we provide **6 training notebooks**.  
Each notebook trains the hierarchical classifier using a different configuration
(MLP, InnerProduct, self training, GNN, etc.).

List of submission notebooks :  
├── 1_Linear.ipynb -> MLP baseline model (~MLP only 1 linear layer)    
├── 2_Innerproduct.ipynb -> Inner-product classifier        
├── 3_GNN.ipynb -> Graph-based GNN baseline   
├── 4_Selftraining.ipynb -> Self-training on innerproduct baseline  
├── 5_MLP_GNN.ipynb -> Graph-based GNN baseline + MLP projection before enter in the GNN    
├── 6_MIX_ensemble.ipynb -> Ensemble baseline (GNN + Innerproduct baseline) 

Each notebook follows the same pipeline:
1. Load corpus embeddings  
2. Load label embeddings (with hierarchy or not)  
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
