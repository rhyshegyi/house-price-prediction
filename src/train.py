"""Train baseline and tree models; save the best full sklearn Pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.preprocess import CAT_COLS, NUM_COLS, SuburbGrouper, select_features

DATA_PATH = ROOT / "data" / "melb_data.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "price_model.joblib"
METRICS_PATH = MODEL_DIR / "training_metrics.json"


def rmse_dollars(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(np.expm1(y_true_log), np.expm1(y_pred_log))))


def build_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_transformer, NUM_COLS),
            ("cat", categorical_transformer, CAT_COLS),
        ],
        remainder="drop",
    )


def build_full_pipeline(estimator) -> Pipeline:
    return Pipeline(
        [
            ("suburb", SuburbGrouper(column="Suburb", min_count=20, other_label="Other")),
            ("prep", build_preprocessor()),
            ("model", estimator),
        ]
    )


def evaluate(
    name: str,
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test_log: np.ndarray,
) -> dict:
    y_pred_log = pipeline.predict(X_test)
    return {
        "model": name,
        "rmse_dollars": rmse_dollars(y_test_log, y_pred_log),
        "r2_log": float(r2_score(y_test_log, y_pred_log)),
    }


def top_feature_importances(
    pipeline: Pipeline, n: int = 10
) -> list[dict[str, float]]:
    """RandomForest feature importances on transformed feature names."""
    model = pipeline.named_steps["model"]
    prep = pipeline.named_steps["prep"]
    if not hasattr(model, "feature_importances_"):
        return []
    names = prep.get_feature_names_out()
    scores = model.feature_importances_
    order = np.argsort(scores)[::-1][:n]
    return [{"feature": str(names[i]), "importance": float(scores[i])} for i in order]


def main():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["Price"])
    y_log = np.log1p(df["Price"].astype(float))
    X = select_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )

    candidates = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
            max_depth=None,
        ),
    }

    results = []
    fitted = {}
    for name, est in candidates.items():
        pipe = build_full_pipeline(est)
        pipe.fit(X_train, y_train)
        metrics = evaluate(name, pipe, X_test, y_test.values)
        results.append(metrics)
        fitted[name] = pipe

    best_name = min(results, key=lambda m: m["rmse_dollars"])["model"]
    best_pipe = fitted[best_name]

    importances = top_feature_importances(best_pipe, n=10)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "pipeline": best_pipe,
        "model_name": best_name,
        "metrics": {m["model"]: m for m in results},
        "feature_importances": importances,
    }
    joblib.dump(bundle, MODEL_PATH)

    METRICS_PATH.write_text(json.dumps(bundle["metrics"], indent=2), encoding="utf-8")

    print(f"Saved best model ({best_name}) to {MODEL_PATH}")
    for m in results:
        print(m)


if __name__ == "__main__":
    main()
