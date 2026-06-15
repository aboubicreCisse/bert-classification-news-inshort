import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix


def set_seed(seed=42):
    """
    Fixe la seed pour la reproductibilité.
    Args:
        seed (int): Valeur de la seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def compute_metrics(preds, labels):
    """
    Calcule accuracy et F1-score.
    Args:
        preds (list): Prédictions du modèle
        labels (list): Labels réels
    Returns:
        dict avec accuracy et f1
    """
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path="training_curves.png"):
    """
    Trace les courbes de loss et accuracy.
    Args:
        train_losses (list): Loss d'entraînement par epoch
        val_losses (list): Loss de validation par epoch
        train_accs (list): Accuracy d'entraînement par epoch
        val_accs (list): Accuracy de validation par epoch
        save_path (str): Chemin de sauvegarde du graphique
    """
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Courbe de loss
    ax1.plot(epochs, train_losses, "b-o", label="Train Loss")
    ax1.plot(epochs, val_losses, "r-o", label="Val Loss")
    ax1.set_title("Loss par epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    # Courbe d'accuracy
    ax2.plot(epochs, train_accs, "b-o", label="Train Accuracy")
    ax2.plot(epochs, val_accs, "r-o", label="Val Accuracy")
    ax2.set_title("Accuracy par epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Courbes sauvegardées dans {save_path}")


def plot_confusion_matrix(labels, preds, class_names, save_path="confusion_matrix.png"):
    """
    Trace la matrice de confusion.
    Args:
        labels (list): Labels réels
        preds (list): Prédictions
        class_names (list): Noms des classes
        save_path (str): Chemin de sauvegarde
    """
    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title("Matrice de Confusion")
    plt.ylabel("Réel")
    plt.xlabel("Prédit")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Matrice de confusion sauvegardée dans {save_path}")