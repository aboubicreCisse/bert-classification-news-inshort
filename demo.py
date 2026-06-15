import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import gradio as gr
from transformers import BertTokenizer, BertForSequenceClassification
from dataset import TextClassificationDataset

# ── Configuration ──────────────────────────────────────────────
MODEL_NAME = "bert-base-uncased"
MODEL_PATH = "best_model.pt"
MAX_LENGTH = 128
NUM_CLASSES = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Chargement du modèle ───────────────────────────────────────
print("Chargement du modèle...")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()
print("Modèle chargé !")

# ── Fonction de prédiction ─────────────────────────────────────
def predict(text):
    """
    Prédit la catégorie d'un texte de news.
    Args:
        text (str): Texte saisi par l'utilisateur
    Returns:
        dict: Probabilités par classe
    """
    encoding = tokenizer(
        text,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1).squeeze(0)

    id2label = TextClassificationDataset.ID2LABEL
    return {id2label[i]: float(probs[i]) for i in range(NUM_CLASSES)}


# ── Interface Gradio ───────────────────────────────────────────
examples = [
    ["DeepMind's AlphaFold AI system has solved the protein folding problem, predicting 3D shapes of proteins within the width of an atom."],
    ["Cristiano Ronaldo scored a hat-trick as his team secured a dramatic victory in the Champions League final last night."],
    ["The government announced new fiscal policies aimed at reducing the national deficit over the next five years."],
    ["NASA scientists have discovered traces of water ice on the surface of the Moon near the south pole."],
    ["The new Tesla Model S features an upgraded battery that offers a range of over 400 miles on a single charge."]
]

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(
        lines=5,
        placeholder="Entrez un texte de news ici...",
        label="Texte de news"
    ),
    outputs=gr.Label(
        num_top_classes=7,
        label="Catégories prédites"
    ),
    title="📰 Classification de News avec BERT",
    description="Ce modèle classifie des articles de news en 7 catégories : world, entertainment, sports, technology, politics, science, automobile. Fine-tuné sur le dataset Inshort News avec une accuracy de 93.78%.",
    examples=examples,
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())