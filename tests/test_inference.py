import os
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "CarPrice_Assignment.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "src", "mlops_tp", "artifacts", "model.joblib")


def test_predict_returns_scalar_for_regression():
    assert os.path.exists(MODEL_PATH), (
        "model.joblib not found. Run training first or run pytest which includes training test."
    )

    model = joblib.load(MODEL_PATH)

    df = pd.read_csv(DATA_PATH)
    X = df.drop("price", axis=1)

    # On prend 1 seule ligne
    x_one = X.iloc[[0]]
    pred = model.predict(x_one)

    # predict retourne un array de taille 1
    assert isinstance(pred, (list, np.ndarray)), "predict should return array-like"
    assert len(pred) == 1, "predict should return exactly one prediction for one row"

    # la valeur doit être un scalaire numérique
    assert np.issubdtype(type(pred[0]), np.number) or np.issubdtype(np.array(pred).dtype, np.number)