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
├─ outputs/                     # résultats 
├─ rapport/                     # rapport final 
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

## 4. Prérequis et installation

### Environnement
- **Python** ≥ 3.10
- Système recommandé : Linux / macOS (MPS pris en charge sur Mac Apple Silicon)
- GPU optionnel (CUDA si disponible). Le projet fonctionne aussi sur CPU/MPS.

### Dépendances
Les dépendances principales sont :
- torch
- torchvision (optionnel)
- timm
- numpy
- pandas
- scikit-learn
- matplotlib
- pyyaml
- pillow

Installation rapide avec pip :
```bash
pip install torch timm numpy pandas scikit-learn matplotlib pyyaml pillow
```

> Remarque : les backbones pré-entraînés (timm) peuvent nécessiter un accès Internet lors du premier lancement.

---

## 5. Lancement des expériences

Exemple pour lancer l’objectif O1 :

```bash
python src/train/train_o1.py
```

Les paramètres principaux (batch size, learning rate, nombre d’epochs, seed, etc.) sont définis dans :

```
configs/base.yaml
```

Les résultats (courbes d’apprentissage, métriques, visualisations) sont automatiquement sauvegardés dans le dossier `outputs/`.

### Ordre recommandé (reproductibilité)

1. Préparer les données localement dans `data/` (non incluses dans le dépôt).
2. Générer le manifest :
   ```bash
   python src/data/build_manifest.py
   ```
3. Entraîner les modèles de base (O1 – variantes A/B/C1/C2) :
   ```bash
   python src/train/train_o1.py
   ```
4. Entraîner le modèle bimodal à résolution identique (O2) :
   ```bash
   python src/train/train_o2.py
   ```
5. Entraîner le modèle avec pondération des patches (O3) :
   ```bash
   python src/train/train_o3.py
   ```
6. Générer les visualisations d’attention et statistiques IoU (O4) :
   ```bash
   python src/train/train_o4.py
   python src/eval/eval_iou_stats.py
   ```
7. Entraîner le modèle avec régularisation IoU (O5) :
   ```bash
   python src/train/train_o5.py
   ```
8. Évaluer la matrice de confusion (modèle final) :
   ```bash
   python src/eval/eval_confusion.py
   ```
9. Générer le tableau d’ablation récapitulatif :
   ```bash
   python src/eval/eval_ablation_summary.py
   ```

---

## 6. Résultats et évaluation

Les résultats incluent :
- courbes de **loss**, **accuracy** et **F1-score**,
- tableaux de métriques (CSV / JSON),
- matrices de confusion,
- cartes d’attention superposées aux images,
- statistiques **IoU (moyenne et écart-type)**.

Ces résultats sont utilisés pour la rédaction du rapport final (6–10 pages).

---

## 7. Reproductibilité

- Les seeds sont fixés dans les scripts d’entraînement afin d’assurer la reproductibilité.
- Les poids des modèles entraînés (`.pt`) ne sont pas fournis dans le dépôt pour des raisons de taille.
- Les figures et métriques fournies permettent néanmoins d’évaluer les performances et les contributions du projet.

---
