# ============================================
# 0) Imports
# ============================================
import os
import json
import joblib
from datetime import datetime

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor  


# ============================================
# 1) Constantes & chemins
# ============================================
RANDOM_STATE = 42
TARGET = "price"

ARTIFACTS_DIR = os.path.join("src", "mlops_tp", "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

DATA_PATH = os.path.join("data", "bmw.csv")


# ============================================
# 2) Chargement des données
# ============================================
df = pd.read_csv(DATA_PATH)

X = df.drop(TARGET, axis=1)
y = df[TARGET]


# ============================================
# 3) Split 70/15/15
# ============================================
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE
)

print("\n Taille des splits :")
print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)


# ============================================
# 4) Préprocessing + Pipeline
# ============================================
categorical_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
numeric_cols = X.select_dtypes(exclude=["object", "string"]).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE))
])


# ============================================
# 8. Créer / sélectionner l'expérience MLflow
# ============================================
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
mlflow.set_experiment("CarPrice_Ridge")

# ============================================
# 9. Démarrer le run MLflow
# ============================================
with mlflow.start_run():

    # ============================================
    # 5) Entraînement
    # ============================================
    model.fit(X_train, y_train)

    # ============================================
    # 6) Évaluation (Validation + Test)
    # ============================================
    def eval_split(name, X_split, y_split):
        preds = model.predict(X_split)
        mae  = mean_absolute_error(y_split, preds)
        rmse = np.sqrt(mean_squared_error(y_split, preds))
        r2   = r2_score(y_split, preds)

        print(f"\n {name}")
        print("MAE :", mae)
        print("RMSE:", rmse)
        print("R2  :", r2)

        return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}

    metrics_val  = eval_split("Validation", X_val,  y_val)
    metrics_test = eval_split("Test",       X_test, y_test)

    # ============================================
    # 10. Enregistrer les paramètres
    # ============================================
    mlflow.log_param("model",        "Ridge")
    mlflow.log_param("alpha",        1.0)
    mlflow.log_param("random_state", RANDOM_STATE)
    mlflow.log_param("test_size",    0.30)
    mlflow.log_param("numeric_imputer",     "median")
    mlflow.log_param("categorical_imputer", "most_frequent")

    # ============================================
    # 11. Enregistrer les métriques
    # ============================================
    mlflow.log_metric("val_MAE",   metrics_val["MAE"])
    mlflow.log_metric("val_RMSE",  metrics_val["RMSE"])
    mlflow.log_metric("val_R2",    metrics_val["R2"])
    mlflow.log_metric("test_MAE",  metrics_test["MAE"])
    mlflow.log_metric("test_RMSE", metrics_test["RMSE"])
    mlflow.log_metric("test_R2",   metrics_test["R2"])

    # ============================================
    # 7) Traçabilité (run_info.json)
    # ============================================
    run_info = {
        "dataset": "CarPrice_Assignment.csv",
        "shape": list(df.shape),
        "target": TARGET,
        "split": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "random_state": RANDOM_STATE,
        "model": "Ridge(alpha=1.0)",
        "preprocessing": {
            "numeric": ["median imputer", "standard scaler"],
            "categorical": ["most_frequent imputer", "onehot (handle_unknown=ignore)"]
        },
        "metrics": {
            "validation": metrics_val,
            "test": metrics_test
        }
    }

    with open("run_info.json", "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=4, ensure_ascii=False)
    print("\n run_info.json mis à jour")

    # Log du fichier JSON comme artefact MLflow
    mlflow.log_artifact("run_info.json")

    # ============================================
    # 8) Sauvegarde des artefacts
    # ============================================

    # 8.1 model.joblib
    model_path = os.path.join(ARTIFACTS_DIR, "model.joblib")
    joblib.dump(model, model_path)
    print(" Pipeline sauvegardé :", model_path)
    mlflow.log_artifact(model_path)

    # 8.2 metrics.json
    metrics_payload = {
        "timestamp": datetime.now().isoformat(),
        "task": "regression",
        "dataset": "CarPrice_Assignment.csv",
        "model": "Ridge(alpha=1.0)",
        "hyperparameters": {"alpha": 1.0, "random_state": RANDOM_STATE},
        "validation": metrics_val,
        "test": metrics_test
    }

    metrics_path = os.path.join(ARTIFACTS_DIR, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=4, ensure_ascii=False)
    print(" metrics.json généré :", metrics_path)
    mlflow.log_artifact(metrics_path)

    # 8.3 feature_schema.json
    feature_schema = {
        "dataset": "CarPrice_Assignment.csv",
        "target": TARGET,
        "n_features": int(len(X.columns)),
        "features": [
            {"name": col, "dtype": str(df[col].dtype)}
            for col in X.columns
        ]
    }

    schema_path = os.path.join(ARTIFACTS_DIR, "feature_schema.json")
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(feature_schema, f, indent=4, ensure_ascii=False)
    print(" feature_schema.json généré :", schema_path)
    mlflow.log_artifact(schema_path)

    # ============================================
    # 12. Enregistrer le modèle dans MLflow
    # ============================================
    mlflow.sklearn.log_model(model, "model")
    print("\n Run MLflow terminé ")