"""Streamlit UI for Melbourne house price predictions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.schemas import PropertyInput  # noqa: E402

MODEL_PATH = ROOT / "models" / "price_model.joblib"
METRICS_PATH = ROOT / "models" / "training_metrics.json"
DATA_PATH = ROOT / "data" / "melb_data.csv"


@st.cache_resource
def load_bundle():
    if not MODEL_PATH.is_file():
        st.error(
            f"Trained model not found at `{MODEL_PATH}`. "
            "Run `python -m src.train` from the project root first."
        )
        st.stop()
    return joblib.load(MODEL_PATH)


@st.cache_data
def reference_data():
    df = pd.read_csv(DATA_PATH)
    med = df.median(numeric_only=True)
    return {
        "suburbs": sorted(df["Suburb"].dropna().unique()),
        "types": sorted(df["Type"].dropna().unique()),
        "regions": sorted(df["Regionname"].dropna().unique()),
        "median": med,
    }


def top_feature_importances_from_pipeline(pipeline, n: int = 10) -> list[dict]:
    """Match training script: importances on preprocessed feature names."""
    model = pipeline.named_steps["model"]
    prep = pipeline.named_steps["prep"]
    if not hasattr(model, "feature_importances_"):
        return []
    names = prep.get_feature_names_out()
    scores = model.feature_importances_
    order = np.argsort(scores)[::-1][:n]
    return [{"feature": str(names[i]), "importance": float(scores[i])} for i in order]


def main():
    st.set_page_config(page_title="Melbourne House Price Predictor", layout="wide")
    st.title("Melbourne house price predictor")
    st.caption(
        "Regression model trained on the Melbourne Housing dataset (HistGradientBoosting + preprocessing pipeline)."
    )

    bundle = load_bundle()
    pipeline = bundle["pipeline"]
    ref = reference_data()
    med = ref["median"]

    with st.sidebar:
        st.header("Property features")
        suburb = st.selectbox("Suburb", ref["suburbs"], index=0)
        property_type = st.selectbox("Type", ref["types"], index=0)
        regionname = st.selectbox("Region", ref["regions"], index=0)
        rooms = st.slider("Rooms", 1, 12, int(med.get("Rooms", 3)))
        distance = st.slider(
            "Distance to CBD (km)",
            0.0,
            50.0,
            float(med.get("Distance", 10.0)),
            0.5,
        )
        postcode = st.slider(
            "Postcode",
            3000.0,
            4000.0,
            float(med.get("Postcode", 3100)),
            1.0,
        )
        bedroom2 = st.slider("Bedrooms (Bedroom2)", 0, 10, int(med.get("Bedroom2", 3)))
        bathroom = st.slider("Bathrooms", 0, 8, int(med.get("Bathroom", 2)))
        car = st.slider("Car spaces", 0, 10, int(med.get("Car", 2)))
        landsize = st.number_input(
            "Land size (m²)",
            min_value=0.0,
            value=float(med.get("Landsize", 500.0)),
            step=10.0,
        )
        building_area = st.number_input(
            "Building area (m²)",
            min_value=0.0,
            value=float(med.get("BuildingArea", 120.0)) if pd.notna(med.get("BuildingArea")) else 120.0,
            step=5.0,
        )
        year_built = st.number_input(
            "Year built",
            min_value=1800,
            max_value=2030,
            value=int(med.get("YearBuilt", 1980)) if pd.notna(med.get("YearBuilt")) else 1980,
            step=1,
        )
        lat = st.slider(
            "Latitude",
            -38.5,
            -37.4,
            float(med.get("Lattitude", -37.8)),
            0.001,
        )
        lon = st.slider(
            "Longitude",
            144.5,
            145.5,
            float(med.get("Longtitude", 145.0)),
            0.001,
        )
        propertycount = st.number_input(
            "Property count (suburb)",
            min_value=0.0,
            value=float(med.get("Propertycount", 2000.0)),
            step=50.0,
        )

    try:
        prop = PropertyInput(
            suburb=suburb,
            property_type=property_type,
            regionname=regionname,
            rooms=rooms,
            distance=distance,
            postcode=postcode,
            bedroom2=float(bedroom2),
            bathroom=float(bathroom),
            car=float(car),
            landsize=landsize,
            building_area=building_area if building_area > 0 else None,
            year_built=float(year_built),
            lattitude=lat,
            longtitude=lon,
            propertycount=propertycount,
        )
    except Exception as e:
        st.error(f"Validation error: {e}")
        st.stop()

    X = prop.to_feature_frame()
    pred_log = pipeline.predict(X)[0]
    price = float(np.expm1(pred_log))

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Estimated price (AUD)", f"${price:,.0f}")
    with col2:
        st.write("Adjust inputs in the sidebar and use validated features only.")

    st.subheader("Top feature importances")
    imp = bundle.get("feature_importances") or []
    if not imp:
        imp = top_feature_importances_from_pipeline(pipeline)
    if imp:
        chart_data = pd.DataFrame(imp).set_index("feature")["importance"].sort_values()
        st.bar_chart(chart_data)
    else:
        st.info(
            "Feature importances are not exposed for the fitted estimator (e.g. linear regression)."
        )

    with st.expander("Model metrics (holdout test set)"):
        metrics = bundle.get("metrics", {})
        st.json(metrics)
        if METRICS_PATH.is_file():
            st.caption(f"Also saved to `{METRICS_PATH.name}`")
            st.code(json.dumps(json.loads(METRICS_PATH.read_text(encoding="utf-8")), indent=2))


if __name__ == "__main__":
    main()
