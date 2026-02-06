# Projet de Deep Learning – Classification et interprétabilité par CrossViT

Ce dépôt contient le code, les configurations et les résultats expérimentaux du projet de Deep Learning réalisé dans le cadre du cours **Deep Learning (CY Tech)** – année universitaire 2025–2026.

Le projet vise à étudier l’apport de l’architecture **CrossViT** pour la classification d’images de plantes, en combinant des informations issues d’images brutes (*raw*) et d’images segmentées (*seg*), ainsi qu’à analyser l’interprétabilité des modèles à l’aide de cartes d’attention.

---

## 1. Structure du projet

```
DL_projet/
├─ configs/
│  └─ base.yaml
├─ data/                        # (non inclus dans le dépôt) jeu de données local
├─ outputs/                     # résultats (courbes, métriques, figures) – sans poids
├─ rapport/                     # rapport final (6–10 pages)
├─ src/
│  ├─ data/
│  │  ├─ build_manifest.py
│  │  ├─ paired_dataset.py
│  │  └─ transforms.py
│  ├─ eval/
│  │  ├─ eval_ablation_summary.py
│  │  ├─ eval_confusion.py
│  │  └─ eval_iou_stats.py
│  ├─ interpretability/
│  │  └─ attention_rollout.py
│  ├─ models/
│  │  ├─ crossvit_like.py
│  │  ├─ losses.py
│  │  └─ patch_weighting.py
│  ├─ train/
│  │  ├─ train_o1.py
│  │  ├─ train_o2.py
│  │  ├─ train_o3.py
│  │  ├─ train_o4.py
│  │  └─ train_o5.py
│  ├─ eval.py
│  └─ utils.py
├─ .gitignore
└─ README.md
```

---

## 2. Données

Les données ne sont **pas incluses dans ce dépôt**, conformément aux consignes du projet.

Le jeu de données est composé de :
- **images brutes (raw)**,
- **images segmentées (seg)**.

Les données doivent être placées localement dans un dossier `data/` selon l’organisation attendue par le script `paired_dataset.py`.

> ⚠️ Les données et les fichiers volumineux (poids des modèles) sont volontairement exclus du dépôt GitHub.

---

## 3. Objectifs expérimentaux

Le projet est structuré en cinq objectifs successifs :

- **O1** – Étude de modèles de base et variantes mono- et bi-branches  
- **O2** – Fusion bimodale raw / seg  
- **O3** – Pondération des patches à partir de l’image segmentée  
- **O4** – Interprétabilité par attention rollout et évaluation IoU  
- **O5** – Ajout d’un terme de régularisation basé sur l’IoU  

Chaque objectif possède son script d’entraînement dédié dans `src/train/`.

---

## 4. Lancement des expériences

Exemple pour lancer l’objectif O1 :

```bash
python src/train/train_o1.py
```

Les paramètres principaux (batch size, learning rate, nombre d’epochs, seed, etc.) sont définis dans :

```
configs/base.yaml
```

Les résultats (courbes d’apprentissage, métriques, visualisations) sont automatiquement sauvegardés dans le dossier `outputs/`.

---

## 5. Résultats et évaluation

Les résultats incluent :
- courbes de **loss**, **accuracy** et **F1-score**,
- tableaux de métriques (CSV / JSON),
- matrices de confusion,
- cartes d’attention superposées aux images,
- statistiques **IoU (moyenne et écart-type)**.

Ces résultats sont utilisés pour la rédaction du rapport final (6–10 pages).

---

## 6. Reproductibilité

- Les seeds sont fixés dans les scripts d’entraînement afin d’assurer la reproductibilité.
- Les poids des modèles entraînés (`.pt`) ne sont pas fournis dans le dépôt pour des raisons de taille.
- Les figures et métriques fournies permettent néanmoins d’évaluer les performances et les contributions du projet.

---

## 7. Auteur

Projet réalisé par :  
**[Votre nom]**  
Étudiant en informatique – CY Tech