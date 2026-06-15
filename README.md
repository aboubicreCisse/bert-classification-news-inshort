# bert-classification-news-inshort

# NLP avec PyTorch – Fine-Tuning de BERT

## Membres du binôme

- Emmanuel Takor Iwuobi
- Aboubicre CISSE

---

# Présentation du Dataset

Dataset : Inshort News Dataset

Nombre total d'exemples : 4817

Nombre de classes : 7

Classes :

- automobile
- entertainment
- politics
- science
- sports
- technology
- world

---

# Analyse Exploratoire

## Distribution des classes

| Classe | Nombre |
|----------|---------:|
| world | 1021 |
| entertainment | 998 |
| sports | 856 |
| technology | 751 |
| politics | 546 |
| science | 389 |
| automobile | 256 |

Le dataset présente un déséquilibre modéré (≈4:1).

---

## Longueur des textes

Minimum : 49 mots

Maximum : 78 mots

Moyenne : 69.77 mots

Nous avons choisi :

```python
MAX_LENGTH = 128
```

afin de couvrir l'ensemble des textes.

## Exemples

Ajouter 5 exemples du dataset ici.

---

# Modèle Utilisé

```python
bert-base-uncased
```

Pourquoi ?

- Dataset anglais
- Modèle robuste
- Pré-entraîné sur un très grand corpus

---

# Hyperparamètres

```python
Learning Rate = 2e-5

Batch Size = 16

Epochs = 4

Max Length = 128

Weight Decay = 0.01
```

---

# Architecture

BERT Encoder

↓

Classification Head

↓

Softmax

---

# Résultats

## Accuracy Finale

À compléter après entraînement.

## F1-Score Final

À compléter après entraînement.

---

# Courbe de Loss

Ajouter :

```
figures/loss_curve.png
```

---

# Courbe Accuracy

Ajouter :

```
figures/accuracy_curve.png
```

---

# Matrice de Confusion

Ajouter :

```
figures/confusion_matrix.png
```

---

# Démo Gradio

Capture d'écran :

```
gradio_demo.png
```

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

- Choix des hyperparamètres
- Gestion du fine-tuning BERT
- Temps d'entraînement

---

# Répartition du Travail

### Membre 1

- Analyse du dataset
- Implémentation Dataset
- Entraînement

### Membre 2

- Gradio
- README
- Tests

---

# Conclusion

Le fine-tuning de BERT a permis d'obtenir d'excellentes performances sur la classification multi-classes de news grâce au transfert d'apprentissage.