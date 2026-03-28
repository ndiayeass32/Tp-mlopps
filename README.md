# TP MLOps — Prédiction du prix de voitures BMW

## Description du projet

Le dataset BMW contient des informations techniques et commerciales sur différents
modèles de voitures BMW. Il est pertinent pour étudier l'impact des caractéristiques
mécaniques (puissance, cylindrée, consommation, transmission…) sur le prix final.

La tâche est un problème de régression supervisée, car la variable cible price est
continue. L'objectif est donc de prédire une valeur numérique.

Le dataset contient 10 781 observations et 9 variables. On y retrouve :
- des variables numériques (ex. year, mileage, tax, mpg, engineSize)
- des variables catégorielles (ex. model, transmission, fuelType)

Cela nécessite un prétraitement adapté, notamment l'encodage des variables
catégorielles et une gestion cohérente des features dans un pipeline.

Les défis anticipés sont :
- la présence de valeurs extrêmes pouvant influencer la régression,
- la corrélation entre certaines variables mécaniques,
- l'encodage correct des variables catégorielles,
- la reproductibilité dans une approche MLOps.

---

## Changement de dataset

Le dataset initial CarPrice_Assignment.csv (205 lignes) a été remplacé par le
dataset BMW (10 781 lignes) car le premier dataset était trop petit pour obtenir
un modèle de qualité suffisante. Le passage à RandomForest sur ce nouveau dataset
a permis d'atteindre un R²=0.953 contre 0.934 précédemment.

---

## Structure du projet
```
TP_MLOPPS/
│
├── data/
│   └── bmw.csv
│
├── src/
│   └── mlops_tp/
│       ├── __init__.py
│       ├── train.py
│       ├── inference.py
│       ├── api.py
│       ├── config.py
│       ├── schemas.py
│       └── artifacts/
│           ├── model.joblib
│           ├── metrics.json
│           ├── feature_schema.json
│           └── run_info.json
│
├── tests/
│   ├── test_training.py
│   ├── test_inference.py
│   └── test_api.py
│
├── .github/
│   └── workflows/
│       └── ci.yml
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Installation & lancement

### 1. Cloner le dépôt
```
git clone https://github.com/ndiayeass32/Tp-mlopps.git
cd Tp-mlopps
```

### 2. Créer et activer l'environnement virtuel
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Installer les dépendances
```
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Entraîner le modèle
```
python src/mlops_tp/train.py
```

### 5. Lancer l'API
```
uvicorn src.mlops_tp.api:app --host 0.0.0.0 --port 8000 --reload
```

Swagger disponible sur : http://localhost:8000/docs

### 6. Lancer les tests
```
pytest tests/ -v
```

---

## MLflow

La bibliothèque MLflow a été intégrée pour suivre les expériences d'entraînement,
comparer les runs et enregistrer les modèles.

### Lancer l'interface MLflow
```
mlflow ui --host 127.0.0.1 --port 5000
```

Interface disponible sur : http://127.0.0.1:5000

### Ce qui est enregistré

- Paramètres : model, alpha, random_state, test_size, numeric_imputer, categorical_imputer
- Métriques : val_MAE, val_RMSE, val_R2, test_MAE, test_RMSE, test_R2
- Artefacts : model.joblib, metrics.json, feature_schema.json, run_info.json

### Meilleur modèle retenu

RandomForest + MinMaxScaler + imputation median avec le dataset BMW :
- test_R2 = 0.953
- test_MAE = 1540
- test_RMSE = 2427

---

## Docker

La bibliothèque YData a été installée afin d'obtenir un aperçu exploratoire
des données et d'analyser leurs caractéristiques.

Un Dockerfile a été créé pour construire un conteneur Docker avec les commandes
de base nécessaires à l'exécution du projet.

### Lancer avec Docker Compose
```
docker compose up --build
```

- API disponible sur : http://localhost:8000/docs
- MLflow disponible sur : http://localhost:5000

### Build et run manuel
```
docker build -t carprice-api .
docker run -p 8000:8000 carprice-api
```

---

## CI/CD

### CI — GitHub Actions

A chaque push ou pull request sur main, le pipeline CI :
1. Installe Python 3.11
2. Installe les dépendances
3. Lance les tests avec pytest
4. Build l'image Docker

### CD — Render

Le service se redéploie automatiquement après validation de la CI
grâce à l'option "Auto-Deploy after CI checks pass" sur Render.

---

## Déploiement

L'ensemble du projet a été versionné et push sur le dépôt GitHub personnel : ndiayeass32.

- API FastAPI : https://tp-mlopps.onrender.com
- Swagger : https://tp-mlopps.onrender.com/docs
- Health check : https://tp-mlopps.onrender.com/health

---

## Outils utilisés

- scikit-learn : Pipeline de prétraitement + RandomForest
- MLflow : Tracking des expériences
- FastAPI : API REST pour l'inférence
- Docker : Conteneurisation
- GitHub Actions : CI automatisée
- Render : Déploiement cloud
- Streamlit : Interface utilisateur
- YData Profiling : Analyse exploratoire des données

---

## Dépôt GitHub

https://github.com/ndiayeass32/Tp-mlopps