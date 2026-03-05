import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt


# =========================
# Paths
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "CarPrice_Assignment.csv"

ARTIFACTS_DIR = PROJECT_ROOT / "src" / "mlops_tp" / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
SCHEMA_PATH = ARTIFACTS_DIR / "feature_schema.json"


# =========================
# Helpers
# =========================
@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def load_json(path: Path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_resource
def load_model(path: Path):
    if path.exists():
        return joblib.load(path)
    return None


def is_numeric_dtype_str(dtype_str: str) -> bool:
    """dtype_str comes from str(df[col].dtype) like 'int64', 'float64', 'object'."""
    dtype_str = (dtype_str or "").lower()
    return ("int" in dtype_str) or ("float" in dtype_str) or ("bool" in dtype_str)


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def get_feature_list(schema: dict) -> list[dict]:
    """Return features as list of dicts: [{'name':..., 'dtype':...}, ...]"""
    if not schema:
        return []
    feats = schema.get("features")
    if isinstance(feats, list) and len(feats) > 0 and isinstance(feats[0], dict):
        return feats
    return []


# =========================
# UI
# =========================
st.set_page_config(page_title="TP MLOps - Car Price Prediction", layout="wide")
st.title("TP MLOps - Car Price Prediction 🚗")
st.caption("Dashboard EDA + métriques + prédiction (Streamlit)")

df = load_data(DATA_PATH) if DATA_PATH.exists() else None
metrics = load_json(METRICS_PATH)
schema = load_json(SCHEMA_PATH)
model = load_model(MODEL_PATH)

st.sidebar.header("Navigation")
page = st.sidebar.radio("Aller à :", ["Dataset & EDA", " Modèle & Métriques", " Prédiction"])


# =========================
# PAGE 1: EDA
# =========================
if page == "Dataset & EDA":
    st.subheader(" Dataset")

    if df is None:
        st.error(f"Dataset introuvable : {DATA_PATH}")
        st.stop()

    col1, col2, col3 = st.columns(3)
    col1.metric("Lignes", df.shape[0])
    col2.metric("Colonnes", df.shape[1])
    col3.metric("Valeurs manquantes", int(df.isna().sum().sum()))

    st.write("Aperçu :")
    st.dataframe(df.head(20), use_container_width=True)

    st.divider()
    st.subheader(" Statistiques rapides")
    st.dataframe(df.describe(include="all").transpose(), use_container_width=True)

    st.divider()
    st.subheader(" Graphiques (choix)")

    target_col = "price" if "price" in df.columns else None
    if not target_col:
        st.warning("Colonne `price` non trouvée dans le dataset.")
        st.stop()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # Distribution target
    st.markdown("### Distribution de `price`")
    fig = plt.figure()
    plt.hist(df[target_col].dropna(), bins=30)
    plt.xlabel("price")
    plt.ylabel("count")
    st.pyplot(fig, clear_figure=True)

    # Scatter configurable
    st.markdown("### Scatter (numérique vs price)")
    xcol = st.selectbox("Choisir une variable numérique :", [c for c in numeric_cols if c != target_col])
    fig = plt.figure()
    plt.scatter(df[xcol], df[target_col], s=12)
    plt.xlabel(xcol)
    plt.ylabel("price")
    st.pyplot(fig, clear_figure=True)

    # Boxplot configurable
    if len(cat_cols) > 0:
        st.markdown("### Boxplot (catégorielle)")
        cat = st.selectbox("Choisir une variable catégorielle :", cat_cols)
        fig = plt.figure(figsize=(10, 4))
        df.boxplot(column=target_col, by=cat)
        plt.title("")
        plt.suptitle("")
        plt.xticks(rotation=30, ha="right")
        st.pyplot(fig, clear_figure=True)

    # Correlation (numériques)
    st.markdown("### Corrélation (variables numériques)")
    corr = df[numeric_cols].corr(numeric_only=True)
    fig = plt.figure(figsize=(10, 6))
    plt.imshow(corr, aspect="auto")
    plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=90)
    plt.yticks(range(len(numeric_cols)), numeric_cols)
    plt.colorbar()
    st.pyplot(fig, clear_figure=True)


# =========================
# PAGE 2: METRICS / MODEL
# =========================
elif page == "Modèle & Métriques":
    st.subheader("Modèle & Artefacts")

    c1, c2, c3 = st.columns(3)

    c1.metric("model.joblib", "OK" if model is not None else "Missing")
    c2.metric("metrics.json", "OK" if metrics is not None else "Missing")
    c3.metric("feature_schema.json", "OK" if schema is not None else "Missing")

    st.divider()
    st.subheader(" Métriques (selon ton train.py)")

    if not metrics:
        st.warning(f"metrics.json introuvable : {METRICS_PATH}")
    else:
        val = metrics.get("validation", {})
        test = metrics.get("test", {})

        col1, col2, col3 = st.columns(3)
        col1.metric("Validation MAE", f"{val.get('MAE', '—')}")
        col2.metric("Validation RMSE", f"{val.get('RMSE', '—')}")
        col3.metric("Validation R²", f"{val.get('R2', '—')}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Test MAE", f"{test.get('MAE', '—')}")
        col2.metric("Test RMSE", f"{test.get('RMSE', '—')}")
        col3.metric("Test R²", f"{test.get('R2', '—')}")

        st.write("Contenu complet :")
        st.json(metrics)

    st.divider()
    st.subheader(" Feature schema (features + dtypes)")

    feats = get_feature_list(schema)
    if not feats:
        st.warning(f"feature_schema.json invalide/introuvable : {SCHEMA_PATH}")
    else:
        st.write(f"Nombre de features : **{len(feats)}**")
        st.dataframe(pd.DataFrame(feats), use_container_width=True)


# =========================
# PAGE 3: PREDICTION
# =========================
else:
    st.subheader("🤖 Prédiction")

    if df is None:
        st.error("Dataset introuvable. Il est nécessaire pour construire les menus de catégories.")
        st.stop()

    if model is None:
        st.error("model.joblib introuvable. Lance d'abord l'entraînement pour générer les artefacts.")
        st.code("python -m src.mlops_tp.train", language="bash")
        st.stop()

    feats = get_feature_list(schema)
    if not feats:
        st.error("feature_schema.json manquant / invalide. Impossible de générer le formulaire complet.")
        st.stop()

    # target removed
    target = schema.get("target", "price")
    feature_names = [f["name"] for f in feats if f.get("name") != target]

    st.info("Formulaire auto généré à partir de feature_schema.json + valeurs possibles depuis le dataset.")

    # Build inputs
    with st.form("predict_form"):
        # Option: split in 2 columns
        left, right = st.columns(2)
        inputs = {}

        for i, f in enumerate(feats):
            name = f.get("name")
            dtype = f.get("dtype", "")

            if not name or name == target:
                continue

            col = left if (i % 2 == 0) else right

            # If categorical: use selectbox with dataset uniques
            if not is_numeric_dtype_str(dtype):
                options = df[name].dropna().astype(str).unique().tolist()
                options = sorted(options)
                default_index = 0 if options else None

                with col:
                    if options:
                        inputs[name] = st.selectbox(name, options, index=default_index)
                    else:
                        inputs[name] = st.text_input(name, value="")
            else:
                # Numeric: choose min/max from dataset if possible
                series = pd.to_numeric(df[name], errors="coerce")
                minv = float(np.nanmin(series)) if np.isfinite(np.nanmin(series)) else 0.0
                maxv = float(np.nanmax(series)) if np.isfinite(np.nanmax(series)) else 100.0
                meanv = float(np.nanmean(series)) if np.isfinite(np.nanmean(series)) else 0.0

                # if int-like
                is_int = "int" in (dtype or "").lower()

                with col:
                    if is_int:
                        inputs[name] = st.number_input(
                            name,
                            min_value=int(np.floor(minv)),
                            max_value=int(np.ceil(maxv)),
                            value=int(np.round(meanv)),
                            step=1,
                        )
                    else:
                        inputs[name] = st.number_input(
                            name,
                            min_value=float(minv),
                            max_value=float(maxv),
                            value=float(meanv),
                        )

        submitted = st.form_submit_button("Predict ")

    if submitted:
        X = pd.DataFrame([inputs])

        # Ensure numeric conversion where needed
        for f in feats:
            name = f.get("name")
            dtype = f.get("dtype", "")
            if name in X.columns and is_numeric_dtype_str(dtype):
                X[name] = X[name].apply(safe_float)

        try:
            pred = model.predict(X)
            st.success(f" Prix prédit : **{float(pred[0]):.2f}**")
            st.caption("Le modèle est ton Pipeline (preprocess + Ridge) → donc pas besoin d’encoder à la main.")
        except Exception as e:
            st.error("Erreur pendant la prédiction.")
            st.code(str(e))

        st.write("Entrée envoyée au modèle :")
        st.dataframe(X, use_container_width=True)