---

# Résultats

## Métriques par epoch

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Val F1 |
|-------|-----------|-----------|----------|---------|--------|
| 1     | 0.6585    | 81.94%    | 0.2246   | 93.46%  | 93.46% |
| 2     | 0.1838    | 94.89%    | 0.2150   | 92.95%  | 92.86% |
| 3     | 0.1180    | 96.37%    | 0.2054   | **93.78%** | **93.76%** |

## Meilleur modèle
- **Val Accuracy : 93.78%**
- **Val F1-score : 93.76%**
- Sauvegardé à l'epoch 3 (val_loss = 0.2054)

---

# Courbe de Loss
![Loss Curve](figures/loss_curve.png)

---

# Courbe Accuracy
![Accuracy Curve](figures/accuracy_curve.png)

---

# Matrice de Confusion
![Confusion Matrix](figures/confusion_matrix.png)

---

# Démo Gradio
![Gradio Demo](figures/gradio_demo.png)

---

# Installation
```bash
pip install -r requirements.txt
```

# Entraînement
```bash
python train.py
```

# Démo
```bash
python demo.py
```

---

# Difficultés Rencontrées
- Conflit de librairies OpenMP sur Windows (résolu avec KMP_DUPLICATE_LIB_OK=TRUE)
- Incompatibilité entre versions des fichiers lors de la collaboration Git
- Temps d'entraînement sur CPU (~15 min par epoch)
- Gestion du .gitignore pour éviter de pousser les fichiers lourds

---

# Répartition du Travail
### Aboubicre CISSE
- Analyse exploratoire du dataset
- Implémentation de `dataset.py`
- Boucle d'entraînement `train.py`
- `utils.py` (métriques, courbes)
- Interface Gradio `demo.py`

### Emmanuel Takor Iwuobi
- Implémentation de `model.py`
- Structure initiale du projet
- `README.md`
- Tests et validation

---

# Conclusion
Le fine-tuning de BERT sur le dataset Inshort News a permis d'atteindre une accuracy de **93.78%** et un F1-score de **93.76%** en seulement 3 epochs. Ces résultats démontrent l'efficacité du transfer learning en NLP — BERT pré-entraîné converge rapidement et généralise bien même sur un dataset de taille modeste.