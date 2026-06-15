import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from dataset import TextClassificationDataset
from utils import set_seed, compute_metrics, plot_training_curves, plot_confusion_matrix

# ── Configuration ──────────────────────────────────────────────
CONFIG = {
    "data_path": "data/inshort_news_data-1 2.csv",
    "model_name": "bert-base-uncased",
    "max_length": 128,
    "batch_size": 16,
    "epochs": 3,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "seed": 42,
    "num_classes": 7,
    "save_path": "best_model.pt",
}


def train_epoch(model, dataloader, optimizer, device):
    """
    Boucle d'entraînement pour une epoch.
    Args:
        model: Modèle BERT
        dataloader: DataLoader d'entraînement
        optimizer: Optimiseur AdamW
        device: CPU ou GPU
    Returns:
        avg_loss, accuracy, f1
    """
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    loop = tqdm(dataloader, desc="Training", leave=False)

    for batch in loop:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(outputs.logits, dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())

        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(all_preds, all_labels)

    return avg_loss, metrics["accuracy"], metrics["f1"]


def eval_epoch(model, dataloader, device):
    """
    Boucle d'évaluation pour une epoch.
    Args:
        model: Modèle BERT
        dataloader: DataLoader de validation
        device: CPU ou GPU
    Returns:
        avg_loss, accuracy, f1, all_preds, all_labels
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    loop = tqdm(dataloader, desc="Validation", leave=False)

    with torch.no_grad():
        for batch in loop:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            total_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

            loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(all_preds, all_labels)

    return avg_loss, metrics["accuracy"], metrics["f1"], all_preds, all_labels


def main():
    """Fonction principale : chargement, entraînement, évaluation."""

    # Seed pour reproductibilité
    set_seed(CONFIG["seed"])

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device utilisé : {device}")

    # Chargement du dataset
    print("Chargement du dataset...")
    df = pd.read_csv(CONFIG["data_path"])
    df = df.dropna(subset=["news_article", "news_category"])

    texts = df["news_article"].tolist()
    labels = df["news_category"].tolist()

    # Split 80/20 stratifié
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels,
        test_size=0.2,
        random_state=CONFIG["seed"],
        stratify=labels
    )

    print(f"Train : {len(train_texts)} exemples")
    print(f"Validation : {len(val_texts)} exemples")

    # Tokenizer
    print("Chargement du tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(CONFIG["model_name"])

    # Datasets et DataLoaders
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, CONFIG["max_length"])
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, CONFIG["max_length"])

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    # Modèle
    print("Chargement du modèle BERT...")
    model = BertForSequenceClassification.from_pretrained(
        CONFIG["model_name"],
        num_labels=CONFIG["num_classes"]
    )
    model.to(device)

    # Optimiseur
    optimizer = AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])

    # Historique
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_loss = float("inf")

    # Boucle d'entraînement
    print("\nDébut de l'entraînement...")
    for epoch in range(1, CONFIG["epochs"] + 1):
        print(f"\n── Epoch {epoch}/{CONFIG['epochs']} ──")

        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc, val_f1, val_preds, val_labels_list = eval_epoch(model, val_loader, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Train  → Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f} | F1: {train_f1:.4f}")
        print(f"Val    → Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f} | F1: {val_f1:.4f}")

        # Sauvegarde du meilleur modèle
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CONFIG["save_path"])
            print(f"Meilleur modèle sauvegardé (val_loss={val_loss:.4f})")

    # Courbes et matrice de confusion
    print("\nGénération des courbes...")
    plot_training_curves(train_losses, val_losses, train_accs, val_accs)

    class_names = list(TextClassificationDataset.LABEL2ID.keys())
    plot_confusion_matrix(val_labels_list, val_preds, class_names)

    print("\nEntraînement terminé !")
    print(f"Meilleur modèle sauvegardé dans : {CONFIG['save_path']}")


if __name__ == "__main__":
    main()