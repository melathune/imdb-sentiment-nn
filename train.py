import os
import re
import json
import joblib
import numpy as np
import pandas as pd

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

import matplotlib.pyplot as plt

# =========================================================
# CONFIG
# =========================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 0.0005

MAX_FEATURES = 10000
NGRAM_RANGE = (1, 2)

HIDDEN_DIM = 256
DROPOUT = 0.3

SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

os.makedirs("model", exist_ok=True)

# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):
    text = str(text)

    # remove html
    text = re.sub(r"<br\s*/?>", " ", text)

    # lowercase
    text = text.lower()

    # keep letters/numbers only
    text = re.sub(r"[^a-z0-9\s]", "", text)

    # remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

# =========================================================
# DATASET
# =========================================================

class IMDBDataset(Dataset):

    def __init__(self, features, labels):

        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels).unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        return {
            "inputs": self.features[idx],
            "labels": self.labels[idx]
        }

# =========================================================
# MODEL
# =========================================================

class SentimentMLP(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Linear(HIDDEN_DIM, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Linear(128, 1)
        )

    def forward(self, x):

        return self.network(x)

# =========================================================
# TRAIN
# =========================================================

def train_epoch(model, loader, optimizer, criterion):

    model.train()

    epoch_loss = 0

    all_preds = []
    all_labels = []

    for batch in tqdm(loader, desc="Training"):

        inputs = batch["inputs"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

        preds = (torch.sigmoid(outputs) >= 0.5).float()

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)

    return epoch_loss / len(loader), accuracy

# =========================================================
# EVALUATE
# =========================================================

def evaluate(model, loader, criterion):

    model.eval()

    epoch_loss = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for batch in tqdm(loader, desc="Evaluating"):

            inputs = batch["inputs"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            epoch_loss += loss.item()

            preds = (torch.sigmoid(outputs) >= 0.5).float()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)

    precision = precision_score(all_labels, all_preds)

    recall = recall_score(all_labels, all_preds)

    f1 = f1_score(all_labels, all_preds)

    return (
        epoch_loss / len(loader),
        accuracy,
        precision,
        recall,
        f1,
        all_preds,
        all_labels
    )

# =========================================================
# PLOTS
# =========================================================

def plot_training(train_accs, val_accs):

    plt.figure(figsize=(8, 5))

    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(val_accs, label="Validation Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.title("Training Results")

    plt.legend()

    plt.grid(True)

    plt.savefig("model/training_results.png")

    plt.close()

def plot_conf_matrix(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 6))

    plt.imshow(cm)

    plt.title("Confusion Matrix")

    plt.colorbar()

    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.savefig("model/confusion_matrix.png")

    plt.close()

# =========================================================
# MAIN
# =========================================================

def main():

    print(f"Using device: {DEVICE}")

    # =====================================================
    # LOAD DATA
    # =====================================================

    print("\nLoading dataset...")

    df = pd.read_csv("data/imdb_balanced_10k.csv")

    texts = df["text"].astype(str)

    labels = df["label"].values

    print(f"Dataset size: {len(texts)}")

    # =====================================================
    # SPLIT
    # =====================================================

    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=SEED,
        stratify=labels
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=SEED,
        stratify=y_train
    )

    # =====================================================
    # TF-IDF
    # =====================================================

    print("\nBuilding TF-IDF features...")

    vectorizer = TfidfVectorizer(
        preprocessor=clean_text,
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        stop_words="english",
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )

    X_train_vec = vectorizer.fit_transform(X_train).astype(np.float32)

    X_val_vec = vectorizer.transform(X_val).astype(np.float32)

    X_test_vec = vectorizer.transform(X_test).astype(np.float32)

    print(f"Feature size: {X_train_vec.shape[1]}")

    # convert sparse -> dense
    X_train_vec = X_train_vec.toarray()
    X_val_vec = X_val_vec.toarray()
    X_test_vec = X_test_vec.toarray()

    # =====================================================
    # DATASETS
    # =====================================================

    train_dataset = IMDBDataset(X_train_vec, y_train)

    val_dataset = IMDBDataset(X_val_vec, y_val)

    test_dataset = IMDBDataset(X_test_vec, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE
    )

    # =====================================================
    # MODEL
    # =====================================================

    print("\nInitializing model...")

    model = SentimentMLP(X_train_vec.shape[1]).to(DEVICE)

    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    criterion = nn.BCEWithLogitsLoss()

    # =====================================================
    # TRAINING
    # =====================================================

    train_accs = []
    val_accs = []

    best_val_acc = 0

    print("\nStarting training...")

    for epoch in range(EPOCHS):

        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion
        )

        val_loss, val_acc, val_prec, val_rec, val_f1, _, _ = evaluate(
            model,
            val_loader,
            criterion
        )

        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Accuracy: {train_acc:.4f}")

        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}")

        print(f"Precision: {val_prec:.4f}")
        print(f"Recall: {val_rec:.4f}")
        print(f"F1 Score: {val_f1:.4f}")

        if val_acc > best_val_acc:

            best_val_acc = val_acc

            torch.save(model.state_dict(), "model/model.pt")

            print("Best model saved")

    # =====================================================
    # TEST
    # =====================================================

    print("\nTesting best model...")

    model.load_state_dict(torch.load("model/model.pt"))

    (
        test_loss,
        test_acc,
        test_prec,
        test_rec,
        test_f1,
        test_preds,
        test_labels
    ) = evaluate(
        model,
        test_loader,
        criterion
    )

    print("\n==============================")
    print("FINAL TEST RESULTS")
    print("==============================")

    print(f"Accuracy : {test_acc:.4f}")
    print(f"Precision: {test_prec:.4f}")
    print(f"Recall   : {test_rec:.4f}")
    print(f"F1 Score : {test_f1:.4f}")

    # =====================================================
    # SAVE METRICS
    # =====================================================

    metrics = {
        "accuracy": float(test_acc),
        "precision": float(test_prec),
        "recall": float(test_rec),
        "f1_score": float(test_f1)
    }

    with open("model/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # =====================================================
    # SAVE CONFIG
    # =====================================================

    config = {
        "max_features": MAX_FEATURES,
        "ngram_range": list(NGRAM_RANGE),
        "hidden_dim": HIDDEN_DIM,
        "dropout": DROPOUT,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE
    }

    with open("model/config.json", "w") as f:
        json.dump(config, f, indent=4)

    # =====================================================
    # SAVE VECTORIZER
    # =====================================================

    joblib.dump(vectorizer, "model/vectorizer.pkl")

    # =====================================================
    # PLOTS
    # =====================================================

    plot_training(train_accs, val_accs)

    plot_conf_matrix(test_labels, test_preds)

    print("\nArtifacts saved in model/")

    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()