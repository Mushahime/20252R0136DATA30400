import json, torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import RobertaTokenizer, RobertaModel
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import random
import numpy as np


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ======== LOAD DATA ========
with open("Amazon_products/train/train_corpus.txt", "r", encoding="utf-8") as f:
    id2text_train = {
        int(line.split("\t")[0]): line.split("\t")[1].strip()
        for line in f if "\t" in line
    }

with open("Silver/hier.json", "r", encoding="utf-8") as f:
    silver_data = json.load(f)
silver_hierarchy = silver_data["silver_hierarchy"]

NUM_CLASSES = 531

# ======== DATASET ========
class SilverDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.samples = []
        for pid, label_ids in labels.items():
            if str(pid).isdigit() and int(pid) in texts:
                self.samples.append((texts[int(pid)], label_ids))
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label_ids = self.samples[idx]
        label_vec = torch.zeros(NUM_CLASSES)
        label_vec[label_ids] = 1.0
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": label_vec
        }

# ======== MODEL ========
class SimpleRobertaClassifier(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.dropout = torch.nn.Dropout(0.2)
        self.fc = torch.nn.Linear(self.roberta.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]  # CLS token
        x = self.dropout(cls)
        return self.fc(x)

# ======== SPLIT TRAIN / VAL ========
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
dataset = SilverDataset(id2text_train, silver_hierarchy, tokenizer)

train_size = int(0.85 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# ======== TRAIN LOOP ========
model = SimpleRobertaClassifier(NUM_CLASSES).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
EPOCHS = 2

def evaluate(loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            X = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)
            logits = model(X, mask)
            preds = (torch.sigmoid(logits) > 0.5).float()
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    f1 = f1_score(all_labels, all_preds, average="micro", zero_division=0)
    return f1

print(f"🚀 Training on {len(train_dataset)} samples...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
        X = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        y = batch["labels"].to(device)

        logits = model(X, mask)
        loss = F.binary_cross_entropy_with_logits(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    val_f1 = evaluate(val_loader)
    print(f"Epoch {epoch} | train_loss={total_loss/len(train_loader):.4f} | val_f1={val_f1:.4f}")

# ======== SAVE ========
torch.save(model.state_dict(), "Embeddings/roberta_student.pt")
print("Model saved: roberta_student.pt")
