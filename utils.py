import random
import numpy as np
import torch

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def set_seed(seed=42):
    """
    Fixe toutes les graines aléatoires
    pour assurer la reproductibilité.
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_metrics(y_true, y_pred):
    """
    Calcule Accuracy et F1-score.
    """

    acc = accuracy_score(y_true, y_pred)

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted"
    )

    return acc, f1


def save_loss_curve(
    train_losses,
    val_losses,
    save_path="figures/loss_curve.png"
):
    """
    Sauvegarde la courbe de loss.
    """

    plt.figure(figsize=(8, 5))

    plt.plot(
        train_losses,
        label="Train Loss"
    )

    plt.plot(
        val_losses,
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")

    plt.legend()

    plt.grid(True)

    plt.savefig(
        save_path,
        bbox_inches="tight"
    )

    plt.close()


def save_accuracy_curve(
    train_accs,
    val_accs,
    save_path="figures/accuracy_curve.png"
):
    """
    Sauvegarde la courbe Accuracy.
    """

    plt.figure(figsize=(8, 5))

    plt.plot(
        train_accs,
        label="Train Accuracy"
    )

    plt.plot(
        val_accs,
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")

    plt.legend()

    plt.grid(True)

    plt.savefig(
        save_path,
        bbox_inches="tight"
    )

    plt.close()


def save_confusion_matrix(
    y_true,
    y_pred,
    class_names,
    save_path="figures/confusion_matrix.png"
):
    """
    Sauvegarde la matrice de confusion.
    """

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.title("Confusion Matrix")

    plt.savefig(
        save_path,
        bbox_inches="tight"
    )

    plt.close()


def print_classification_results(
    y_true,
    y_pred,
    class_names
):
    """
    Affiche le classification report.
    """

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )

    print("\n")
    print("=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(report)