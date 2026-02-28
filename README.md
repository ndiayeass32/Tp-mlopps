## 📊 Justification des choix (analyse des données)

Le dataset **CarPrice_Assignment.csv** contient des informations techniques et commerciales sur différents modèles de voitures. Il est intéressant car il permet d’étudier l’impact des caractéristiques mécaniques et structurelles (puissance, poids, cylindrée, consommation…) sur le prix final d’un véhicule.

La tâche est un problème de **régression supervisée**, car la variable cible `price` est continue. L’objectif est de prédire une valeur numérique.

Le dataset contient environ 205 observations et 26 variables explicatives, comprenant des variables numériques (horsepower, enginesize, curbweight, etc.) et catégorielles (fueltype, carbody, drivewheel, etc.). Cela nécessite un prétraitement adapté, notamment l’encodage des variables catégorielles.

Les principaux défis anticipés sont la présence potentielle de valeurs extrêmes (outliers) influençant la régression, la corrélation entre certaines variables mécaniques, et la nécessité d’un encodage correct des variables catégorielles pour garantir la robustesse du modèle.


TP_MLOPPS/
│
├── data/
│ └── CarPrice_Assignment.csv
│
├── src/
│ └── mlops_tp/
│ ├── train.py
│ ├── inference.py
│ ├── api.py
│ ├── config.py
│ ├── schemas.py
│ └── artifacts/
│ ├── model.joblib
│ ├── metrics.json
│ ├── feature_schema.json
│ └── run_info.json
│
├── tests/
│ ├── test_training.py
│ ├── test_inference.py
│ └── test_api.py
│
├── requirements.txt
└── README.md


  Reproduire le projet

```bash
python -m venv .venv
source .venv/bin/activate  # ou .venv\Scripts\activate sous Windows
pip install -r requirements.txt
python -m src.mlops_tp.train
uvicorn src.mlops_tp.api:app --reload

Salut Poto
