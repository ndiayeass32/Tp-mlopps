import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Données ──────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join("data", "CarPrice_Assignment.csv")
TARGET = "price"
RANDOM_STATE = 42

df = pd.read_csv(DATA_PATH)
X = df.drop(TARGET, axis=1)
y = df[TARGET]

categorical_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
numeric_cols = X.select_dtypes(exclude=["object", "string"]).columns.tolist()

# cration des  3 runs ────────────────────────────────────────────────────────
experiments = [
    {
        "run_name":          "Ridge_alpha1_split70",
        "model":             Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "model_name":        "Ridge",
        "alpha":             1.0,
        "test_size":         0.30,
        "numeric_imputer":   "median",
        "scaler":            StandardScaler(),
        "scaler_name":       "StandardScaler",
    },
    {
        "run_name":          "Ridge_alpha10_split80",
        "model":             Ridge(alpha=10.0, random_state=RANDOM_STATE),
        "model_name":        "Ridge",
        "alpha":             10.0,
        "test_size":         0.20,          
        "numeric_imputer":   "mean",         
        "scaler":            StandardScaler(),
        "scaler_name":       "StandardScaler",
    },
    {
        "run_name":          "RandomForest_split70_MinMax",
        "model":             RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
        "model_name":        "RandomForest",
        "alpha":             None,
        "test_size":         0.30,
        "numeric_imputer":   "median",
        "scaler":            MinMaxScaler(),  # normalisation différente
        "scaler_name":       "MinMaxScaler",
    },
]

# ── MLflow ───────────────────────────────────────────────────────────────────
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("CarPrice_Ridge")

for cfg in experiments:
    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=cfg["test_size"], random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE
    )

    # Pipeline
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy=cfg["numeric_imputer"])),
        ("scaler",  cfg["scaler"])
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ])
    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("regressor",  cfg["model"])
    ])

    with mlflow.start_run(run_name=cfg["run_name"]):
        pipeline.fit(X_train, y_train)

        # Métriques
        for split_name, X_s, y_s in [("val", X_val, y_val), ("test", X_test, y_test)]:
            preds = pipeline.predict(X_s)
            mlflow.log_metric(f"{split_name}_MAE",  float(mean_absolute_error(y_s, preds)))
            mlflow.log_metric(f"{split_name}_RMSE", float(np.sqrt(mean_squared_error(y_s, preds))))
            mlflow.log_metric(f"{split_name}_R2",   float(r2_score(y_s, preds)))

        # Paramètres
        mlflow.log_param("model",            cfg["model_name"])
        mlflow.log_param("alpha",            cfg["alpha"])
        mlflow.log_param("test_size",        cfg["test_size"])
        mlflow.log_param("numeric_imputer",  cfg["numeric_imputer"])
        mlflow.log_param("scaler",           cfg["scaler_name"])
        mlflow.log_param("random_state",     RANDOM_STATE)

        # Modèle
        mlflow.sklearn.log_model(pipeline, "model")
        print(f"Run terminé : {cfg['run_name']}")