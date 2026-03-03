Le dataset CarPrice_Assignment.csv regroupe des informations techniques et commerciales sur différents modèles de voitures. Il est pertinent pour étudier l’impact des caractéristiques mécaniques et structurelles (puissance, poids, cylindrée, consommation, transmission…) sur le prix final d’un véhicule.

La tâche est un problème de régression supervisée, car la variable cible price est continue. L’objectif est donc de prédire une valeur numérique.

Le dataset contient 205 observations et 26 variables. On y retrouve :

des variables numériques (ex. horsepower, enginesize, curbweight, etc.)

des variables catégorielles (ex. fueltype, carbody, drivewheel, etc.)

Cela nécessite un prétraitement adapté, notamment l’encodage des variables catégorielles et une gestion cohérente des features dans un pipeline.

Les défis anticipés sont :

la présence de valeurs extrêmes (outliers) pouvant influencer la régression,

la corrélation entre certaines variables mécaniques (risque de redondance),

l’encodage correct des variables catégorielles pour garantir la robustesse du modèle,

la reproductibilité (mêmes features, même pipeline, mêmes métriques) dans une approche MLOps.


TP_MLOPPS/
│
├── data/
│   └── CarPrice_Assignment.csv
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
│           ├── run_info.json
│           └── eda_report.html
│
├── tests/
│   ├── test_training.py
│   ├── test_inference.py
│   └── test_api.py
│
├── Dockerfile
├── .dockerignore
├── requirements.txt
└── README.md

cd "C:\Users\ndiay\Desktop\Tp MLOpps"

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt

# Entraîner le modèle + générer les artifacts
python -m src.mlops_tp.train

# Lancer l'API
uvicorn src.mlops_tp.api:app --host 0.0.0.0 --port 8000 --reload

Swagger : http://localhost:8000/docs