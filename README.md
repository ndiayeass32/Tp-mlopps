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

J’ai installé la bibliothèque YData afin d’obtenir un aperçu exploratoire des données et d’analyser leurs caractéristiques.

J’ai ensuite créé un Dockerfile permettant de construire un conteneur Docker avec les commandes de base nécessaires à l’exécution du projet.

Enfin, l’ensemble du projet a été versionné et push sur mon dépôt GitHub personnel : ndiayeass.




TP 2 : Intégration de MLflow dans un projet ML

.\.venv\Scripts\Activate.ps1
mlflow ui --host 127.0.0.1 --port 5000

Q1 — Qu'appelle-t-on une expérience dans MLflow ?
Une expérience regroupe plusieurs runs liés au même objectif. Dans notre cas on a
appelé la nôtre "CarPrice_Ridge".

Q2 — Qu'appelle-t-on un run ?
C'est une exécution du script d'entraînement. Chaque fois qu'on lance train.py,
un nouveau run est créé avec ses paramètres et résultats.

Q3 — Différence entre paramètre, métrique et artefact ?
Un paramètre c'est ce qu'on fixe avant (ex: alpha=1.0), une métrique c'est ce qu'on
obtient après l'entraînement (ex: MAE), et un artefact c'est un fichier généré
pendant le run (ex: le model.joblib).

Q4 — Dans notre projet
Les paramètres qu'on enregistre sont alpha, random_state et test_size. Les métriques
pertinentes sont MAE, RMSE et R² puisque c'est un problème de régression. Comme
artefacts on a gardé model.joblib, metrics.json et feature_schema.json.

Q5 — Sur quelle adresse l'interface MLflow est-elle accessible ?
Sur http://127.0.0.1:5000 

Q6 — Que remarquez-vous dans l'interface avant d'exécuter votre premier entraînement ?
L'interface est quasi vide, il y a juste un experiment "Default" créé automatiquement
mais pas encore de runs ni de métriques.

Q13 — Quels paramètres avez-vous choisi d'enregistrer ?
On a enregistré model, alpha, random_state, test_size, numeric_imputer et categorical_imputer.

Q14 — Pourquoi ces paramètres sont-ils importants dans votre projet ?
model permet de savoir quel algo on a utilisé, alpha est le paramètre principal de Ridge,
random_state garantit la reproductibilité, test_size documente le découpage des données
et les deux imputers influencent directement le prétraitement.

Q15 — Quelles métriques avez-vous retenues ?
On a retenu MAE, RMSE et R², évaluées sur la validation et le test. Sur la validation
on obtient MAE=1904, RMSE=2346 et R²=0.886. Sur le test MAE=2074, RMSE=3460 et R²=0.866.

Q16 — Pourquoi ces métriques sont-elles adaptées à votre problème ?
C'est un problème de régression donc accuracy ou F1 ne s'appliquent pas. La MAE donne
l'erreur en euros ce qui est facile à comprendre, le RMSE est plus sensible aux grosses
erreurs et le R² indique si le modèle est globalement bon.

Q18 — Quel artefact avez-vous choisi d'enregistrer ?
On a logué 3 fichiers : metrics.json, feature_schema.json et run_info.json.

Q19 — Pourquoi cet artefact est-il utile ?
metrics.json garde une trace des perfs avec la date, feature_schema.json est utile
pour l'inférence car on sait exactement quelles features attendre, et run_info.json
récapitule tout le run en un seul fichier.

Q20 — À quel moment du pipeline est-il produit ?
Après l'évaluation du modèle, à la fin du bloc with mlflow.start_run().

Q21 — Vérification dans l'interface MLflow
On a bien retrouvé dans l'interface les 6 paramètres, les 6 métriques sur val et test,
les artefacts json et le modèle avec status Ready.

Q22 — Ce qui a été modifié pour chaque run
Run 1 Ridge_alpha1_split70 : Ridge, alpha=1.0, test_size=0.30, median, StandardScaler.
Run 2 Ridge_alpha10_split80 : Ridge, alpha=10.0, test_size=0.20, mean, StandardScaler.
Run 3 RandomForest_split70_MinMax : RandomForest, test_size=0.30, median, MinMaxScaler.

Q23 — Pourquoi ces variations ?
On voulait tester si une régularisation plus forte aidait Ridge, si un split différent
changeait les résultats, et surtout comparer avec un modèle non-linéaire comme RandomForest.

Q24 — Quel run semble être le meilleur ?
RandomForest_split70_MinMax, clairement meilleur sur toutes les métriques avec
test_R2=0.934, test_MAE=1633 et test_RMSE=2418.

Q25 — Selon quelle métrique ?
On s'est basé surtout sur le R² et le MAE sur les données de test.

Q26 — Compromis observé entre métriques ?
Ridge alpha=10.0 semblait meilleur sur la validation (val_R2=0.908) mais était
en fait moins bon sur le test (test_R2=0.853), ce qui montre qu'il ne faut pas
se fier qu'à une seule valeur ou qu'à la validation.

Q27 — Une seule métrique suffit-elle ?
Non, l'exemple de Ridge alpha=10.0 le montre bien : meilleur val_R2 mais moins bon
sur le test. Il faut regarder plusieurs métriques sur les données de test.

Q28 — Configuration retenue à ce stade ?
On retient RandomForest avec MinMaxScaler et imputation median car c'est la config
qui donne les meilleurs résultats (R²=0.934, MAE=1633€) sans signe de surapprentissage.


## Changement de Dataset BMW

Remplacement du dataset CarPrice_Assignment.csv (205 lignes) par le dataset BMW
(10 000+ lignes) pour améliorer la qualité du modèle. Le passage à RandomForest
sur ce nouveau dataset a permis d'atteindre un R²=0.953 contre 0.934 précédemment.