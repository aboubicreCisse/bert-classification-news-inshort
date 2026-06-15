import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

class TextClassificationDataset(Dataset):
    """
    Dataset PyTorch pour la classification de texte avec BERT.
    Tokenize les textes et retourne les tenseurs nécessaires.
    """

    # Mapping des catégories vers des indices numériques
    LABEL2ID = {
        "world": 0,
        "entertainment": 1,
        "sports": 2,
        "technology": 3,
        "politics": 4,
        "science": 5,
        "automobile": 6
    }

    ID2LABEL = {v: k for k, v in LABEL2ID.items()}

    def __init__(self, texts, labels, tokenizer, max_length=128):
        """
        Args:
            texts (list): Liste des textes (news_article)
            labels (list): Liste des catégories (news_category)
            tokenizer: Tokenizer BERT
            max_length (int): Longueur maximale des séquences
        """
        self.texts = texts
        self.labels = [self.LABEL2ID[label] for label in labels]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """Retourne le nombre total d'exemples."""
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Retourne un exemple tokenizé.
        Returns:
            dict avec input_ids, attention_mask, label
        """
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Tokenization avec padding et truncation
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }