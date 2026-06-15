from transformers import BertForSequenceClassification


def get_model(model_name, num_classes):
    """
    Charge un modèle BERT pré-entraîné pour la classification.
    """

    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_classes
    )

    return model