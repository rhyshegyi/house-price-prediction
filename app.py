"""Streamlit UI: Melbourne buyer's guide with a model-based price estimate."""

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

# Session-state keys for sidebar inputs (used by reset)
SS_SUBURB = "inp_suburb"
SS_TYPE = "inp_property_type"
SS_REGION = "inp_regionname"
SS_POSTCODE = "inp_postcode"
SS_DISTANCE = "inp_distance"
SS_ROOMS = "inp_rooms"
SS_BEDROOM2 = "inp_bedroom2"
SS_BATHROOM = "inp_bathroom"
SS_CAR = "inp_car"
SS_LANDSIZE = "inp_landsize"
SS_BUILDING = "inp_building_area"
SS_YEAR = "inp_year_built"
SS_LAT = "inp_lat"
SS_LON = "inp_lon"
SS_PROPCOUNT = "inp_propertycount"

PROPERTY_TYPE_LABELS = {
    "h": "House",
    "t": "Townhouse",
    "u": "Unit / apartment",
}


def _format_property_type(code: str) -> str:
    label = PROPERTY_TYPE_LABELS.get(code, code)
    return f"{label} ({code})"


def _format_postcode(pc: float) -> str:
    v = float(pc)
    if v == int(v):
        return str(int(v))
    return str(v)


def _postcode_for_suburb(ref: dict, suburb: str, med: pd.Series) -> float:
    """Modal training-data postcode for this suburb (aligns with suburb choice)."""
    m = ref["suburb_mode_postcode"]
    if suburb in m and pd.notna(m[suburb]):
        return float(m[suburb])
    return float(med.get("Postcode", 3100))


def _lat_for_suburb(ref: dict, suburb: str, med: pd.Series) -> float:
    m = ref["suburb_median_lat"]
    if suburb in m and pd.notna(m[suburb]):
        v = float(m[suburb])
    else:
        v = float(med.get("Lattitude", -37.8))
    lo, hi = ref["lat_bounds"]
    return max(lo, min(hi, v))


def _lon_for_suburb(ref: dict, suburb: str, med: pd.Series) -> float:
    m = ref["suburb_median_lon"]
    if suburb in m and pd.notna(m[suburb]):
        v = float(m[suburb])
    else:
        v = float(med.get("Longtitude", 145.0))
    lo, hi = ref["lon_bounds"]
    return max(lo, min(hi, v))


def _propertycount_for_suburb(ref: dict, suburb: str, med: pd.Series) -> float:
    """`Propertycount` is constant per suburb in the training CSV."""
    m = ref["suburb_propertycount"]
    if suburb in m and pd.notna(m[suburb]):
        return float(m[suburb])
    return float(med.get("Propertycount", 2000.0))


def _distance_for_suburb(ref: dict, suburb: str, med: pd.Series) -> float:
    m = ref["suburb_median_distance"]
    if suburb in m and pd.notna(m[suburb]):
        return float(m[suburb])
    return float(med.get("Distance", 10.0))


def _region_for_suburb(ref: dict, suburb: str) -> str:
    """Each suburb has exactly one `Regionname` in the training CSV."""
    m = ref["suburb_region"]
    if suburb in m and pd.notna(m.get(suburb)):
        return str(m[suburb])
    return ref["regions"][0]


def _int_median_suburb_type(
    ref: dict,
    suburb: str,
    ptype: str,
    med: pd.Series,
    st_key: str,
    su_key: str,
    ty_key: str,
    med_col: str,
    default_med: float,
    lo: int,
    hi: int,
) -> int:
    """Median for (suburb, type), else type-only, else suburb-only, else global."""
    k = (suburb, ptype)
    v = ref[st_key].get(k)
    if v is not None and pd.notna(v):
        x = int(round(float(v)))
        return max(lo, min(hi, x))
    v = ref[ty_key].get(ptype)
    if v is not None and pd.notna(v):
        x = int(round(float(v)))
        return max(lo, min(hi, x))
    v = ref[su_key].get(suburb)
    if v is not None and pd.notna(v):
        x = int(round(float(v)))
        return max(lo, min(hi, x))
    fb = med.get(med_col, default_med)
    if pd.notna(fb):
        return max(lo, min(hi, int(round(float(fb)))))
    return lo


def _landsize_suburb_type(ref: dict, suburb: str, ptype: str, med: pd.Series) -> float:
    k = (suburb, ptype)
    v = ref["suburb_type_median_landsize"].get(k)
    if v is not None and pd.notna(v) and float(v) >= 0:
        return float(v)
    v = ref["type_median_landsize"].get(ptype)
    if v is not None and pd.notna(v) and float(v) >= 0:
        return float(v)
    v = ref["suburb_median_landsize"].get(suburb)
    if v is not None and pd.notna(v) and float(v) >= 0:
        return float(v)
    fb = med.get("Landsize", 500.0)
    return float(fb) if pd.notna(fb) else 0.0


def _building_suburb_type(ref: dict, suburb: str, ptype: str, med: pd.Series) -> float:
    k = (suburb, ptype)
    v = ref["suburb_type_median_building"].get(k)
    if v is not None and pd.notna(v) and float(v) >= 0:
        return float(v)
    v = ref["type_median_building"].get(ptype)
    if v is not None and pd.notna(v) and float(v) >= 0:
        return float(v)
    v = ref["suburb_median_building"].get(suburb)
    if v is not None and pd.notna(v) and float(v) >= 0:
        return float(v)
    fb = med.get("BuildingArea", 120.0)
    return float(fb) if pd.notna(fb) else 120.0


def _year_suburb_type(ref: dict, suburb: str, ptype: str, med: pd.Series) -> int:
    k = (suburb, ptype)
    v = ref["suburb_type_median_year"].get(k)
    if v is not None and pd.notna(v):
        y = int(round(float(v)))
        return max(1800, min(2030, y))
    v = ref["type_median_year"].get(ptype)
    if v is not None and pd.notna(v):
        y = int(round(float(v)))
        return max(1800, min(2030, y))
    v = ref["suburb_median_year"].get(suburb)
    if v is not None and pd.notna(v):
        y = int(round(float(v)))
        return max(1800, min(2030, y))
    fb = med.get("YearBuilt", 1980)
    return int(fb) if pd.notna(fb) else 1980


def _dwelling_values_from_profile(
    ref: dict, med: pd.Series, suburb: str, ptype: str
) -> dict:
    """Median defaults for suburb × property type — exactly these seven inputs:

    rooms, bedrooms, bathrooms, car spaces, land size, building area, year built.
    """
    return {
        SS_ROOMS: _int_median_suburb_type(
            ref,
            suburb,
            ptype,
            med,
            "suburb_type_median_rooms",
            "suburb_median_rooms",
            "type_median_rooms",
            "Rooms",
            3,
            1,
            10,
        ),
        SS_BEDROOM2: _int_median_suburb_type(
            ref,
            suburb,
            ptype,
            med,
            "suburb_type_median_bedroom2",
            "suburb_median_bedroom2",
            "type_median_bedroom2",
            "Bedroom2",
            3,
            0,
            20,
        ),
        SS_BATHROOM: _int_median_suburb_type(
            ref,
            suburb,
            ptype,
            med,
            "suburb_type_median_bathroom",
            "suburb_median_bathroom",
            "type_median_bathroom",
            "Bathroom",
            2,
            0,
            8,
        ),
        SS_CAR: _int_median_suburb_type(
            ref,
            suburb,
            ptype,
            med,
            "suburb_type_median_car",
            "suburb_median_car",
            "type_median_car",
            "Car",
            2,
            0,
            10,
        ),
        SS_LANDSIZE: _landsize_suburb_type(ref, suburb, ptype, med),
        SS_BUILDING: _building_suburb_type(ref, suburb, ptype, med),
        SS_YEAR: _year_suburb_type(ref, suburb, ptype, med),
    }


def _input_defaults(ref: dict, med: pd.Series) -> dict:
    """Dataset-aligned defaults: first suburb + first property type (sorted lists)."""
    first_suburb = ref["suburbs"][0]
    first_type = ref["types"][0]
    out = {
        SS_SUBURB: first_suburb,
        SS_TYPE: first_type,
        SS_REGION: _region_for_suburb(ref, first_suburb),
        SS_POSTCODE: _postcode_for_suburb(ref, first_suburb, med),
        SS_DISTANCE: _distance_for_suburb(ref, first_suburb, med),
        SS_LAT: _lat_for_suburb(ref, first_suburb, med),
        SS_LON: _lon_for_suburb(ref, first_suburb, med),
        SS_PROPCOUNT: _propertycount_for_suburb(ref, first_suburb, med),
    }
    out.update(_dwelling_values_from_profile(ref, med, first_suburb, first_type))
    return out


def _apply_defaults(defaults: dict) -> None:
    for key, val in defaults.items():
        st.session_state[key] = val


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

    def _modal_postcode(s: pd.Series) -> float:
        s = s.dropna()
        if s.empty:
            return float("nan")
        modes = s.mode()
        return float(modes.iloc[0]) if len(modes) else float(s.iloc[0])

    suburb_mode_postcode = (
        df.groupby("Suburb")["Postcode"].agg(_modal_postcode).to_dict()
    )
    suburb_listing_count = df.groupby("Suburb", observed=False).size().to_dict()
    postcodes = sorted(df["Postcode"].dropna().astype(float).unique().tolist())
    lat_bounds = (
        float(df["Lattitude"].min()),
        float(df["Lattitude"].max()),
    )
    lon_bounds = (
        float(df["Longtitude"].min()),
        float(df["Longtitude"].max()),
    )

    g = df.groupby("Suburb", observed=False)
    suburb_median_lat = g["Lattitude"].median().to_dict()
    suburb_median_lon = g["Longtitude"].median().to_dict()
    suburb_median_distance = g["Distance"].median().to_dict()
    # Constant per suburb in this file; .first() matches any row
    suburb_propertycount = g["Propertycount"].first().to_dict()
    suburb_region = g["Regionname"].first().to_dict()
    suburb_median_rooms = g["Rooms"].median().to_dict()
    suburb_median_bedroom2 = g["Bedroom2"].median().to_dict()
    suburb_median_bathroom = g["Bathroom"].median().to_dict()
    suburb_median_car = g["Car"].median().to_dict()
    suburb_median_landsize = g["Landsize"].median().to_dict()
    suburb_median_building = g["BuildingArea"].median().to_dict()
    suburb_median_year = g["YearBuilt"].median().to_dict()

    g_st = df.groupby(["Suburb", "Type"], observed=False)
    suburb_type_median_rooms = g_st["Rooms"].median().dropna().to_dict()
    suburb_type_median_bedroom2 = g_st["Bedroom2"].median().dropna().to_dict()
    suburb_type_median_bathroom = g_st["Bathroom"].median().dropna().to_dict()
    suburb_type_median_car = g_st["Car"].median().dropna().to_dict()
    suburb_type_median_landsize = g_st["Landsize"].median().dropna().to_dict()
    suburb_type_median_building = g_st["BuildingArea"].median().dropna().to_dict()
    suburb_type_median_year = g_st["YearBuilt"].median().dropna().to_dict()

    g_ty = df.groupby("Type", observed=False)
    type_median_rooms = g_ty["Rooms"].median().to_dict()
    type_median_bedroom2 = g_ty["Bedroom2"].median().to_dict()
    type_median_bathroom = g_ty["Bathroom"].median().to_dict()
    type_median_car = g_ty["Car"].median().to_dict()
    type_median_landsize = g_ty["Landsize"].median().to_dict()
    type_median_building = g_ty["BuildingArea"].median().to_dict()
    type_median_year = g_ty["YearBuilt"].median().to_dict()

    return {
        "suburbs": sorted(df["Suburb"].dropna().unique()),
        "types": sorted(df["Type"].dropna().unique()),
        "regions": sorted(df["Regionname"].dropna().unique()),
        "postcodes": postcodes,
        "suburb_mode_postcode": suburb_mode_postcode,
        "suburb_listing_count": suburb_listing_count,
        "suburb_median_lat": suburb_median_lat,
        "suburb_median_lon": suburb_median_lon,
        "suburb_median_distance": suburb_median_distance,
        "suburb_region": suburb_region,
        "suburb_propertycount": suburb_propertycount,
        "suburb_median_rooms": suburb_median_rooms,
        "suburb_median_bedroom2": suburb_median_bedroom2,
        "suburb_median_bathroom": suburb_median_bathroom,
        "suburb_median_car": suburb_median_car,
        "suburb_median_landsize": suburb_median_landsize,
        "suburb_median_building": suburb_median_building,
        "suburb_median_year": suburb_median_year,
        "suburb_type_median_rooms": suburb_type_median_rooms,
        "suburb_type_median_bedroom2": suburb_type_median_bedroom2,
        "suburb_type_median_bathroom": suburb_type_median_bathroom,
        "suburb_type_median_car": suburb_type_median_car,
        "suburb_type_median_landsize": suburb_type_median_landsize,
        "suburb_type_median_building": suburb_type_median_building,
        "suburb_type_median_year": suburb_type_median_year,
        "type_median_rooms": type_median_rooms,
        "type_median_bedroom2": type_median_bedroom2,
        "type_median_bathroom": type_median_bathroom,
        "type_median_car": type_median_car,
        "type_median_landsize": type_median_landsize,
        "type_median_building": type_median_building,
        "type_median_year": type_median_year,
        "lat_bounds": lat_bounds,
        "lon_bounds": lon_bounds,
        "median": med,
    }


def _buyer_guide_map_point(ref: dict, suburb: str, med: pd.Series) -> tuple[float, float]:
    """Median listing coordinates for the suburb (informational only; not tied to Advanced sliders)."""
    return (
        _lat_for_suburb(ref, suburb, med),
        _lon_for_suburb(ref, suburb, med),
    )


def _dollar_range_from_uncertainty(bundle: dict, pred_log: float) -> tuple[float, float] | None:
    """Approximate dollar range using holdout log-residual percentiles, else ±RMSE fallback."""
    unc = bundle.get("uncertainty") or {}
    p5 = unc.get("log_residual_p05")
    p95 = unc.get("log_residual_p95")
    if p5 is not None and p95 is not None:
        lo = float(np.expm1(pred_log + float(p5)))
        hi = float(np.expm1(pred_log + float(p95)))
        return max(0.0, lo), hi
    metrics = bundle.get("metrics") or {}
    name = bundle.get("model_name") or "hist_gradient_boosting"
    row = metrics.get(name) or (next(iter(metrics.values())) if metrics else None)
    if row and row.get("rmse_dollars") is not None:
        rmse = float(row["rmse_dollars"])
        mid = float(np.expm1(pred_log))
        return max(0.0, mid - rmse), mid + rmse
    return None


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


@st.cache_data
def _latest_sales_in_suburb(suburb: str, n: int = 3) -> pd.DataFrame:
    """Most recent dated sales rows for a suburb in the training CSV (day/month/year dates)."""
    df = pd.read_csv(
        DATA_PATH,
        usecols=["Suburb", "Address", "Date", "Price", "Type", "Rooms"],
    )
    df = df[df["Suburb"] == suburb].copy()
    if df.empty:
        return df
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date", "Price"])
    df = df.sort_values("Date", ascending=False).head(n)
    return df


def main():
    st.set_page_config(
        page_title="Melbourne buyer's guide",
        page_icon="🏠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    bundle = load_bundle()
    pipeline = bundle["pipeline"]
    ref = reference_data()
    med = ref["median"]
    defaults = _input_defaults(ref, med)

    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val
    pc_set = set(ref["postcodes"])
    if st.session_state.get(SS_POSTCODE) not in pc_set:
        st.session_state[SS_POSTCODE] = defaults[SS_POSTCODE]
    if st.session_state.get(SS_REGION) not in ref["regions"]:
        st.session_state[SS_REGION] = defaults[SS_REGION]

    lo_lat, hi_lat = ref["lat_bounds"]
    v_lat = float(st.session_state[SS_LAT])
    if v_lat < lo_lat or v_lat > hi_lat:
        st.session_state[SS_LAT] = max(lo_lat, min(hi_lat, v_lat))
    lo_lon, hi_lon = ref["lon_bounds"]
    v_lon = float(st.session_state[SS_LON])
    if v_lon < lo_lon or v_lon > hi_lon:
        st.session_state[SS_LON] = max(lo_lon, min(hi_lon, v_lon))

    st.markdown("## Melbourne buyer's guide")
    st.caption(
        "Explore a suburb and property scenario, see a **model-based price estimate**, "
        "and skim **local context** from the historical sales sample (not a valuation — "
        "the map and notes below are qualitative only)."
    )

    with st.sidebar:
        st.markdown("### Your scenario")
        st.caption(
            "Adjust inputs — the estimate updates when you change a control. "
            "Use **Advanced** only for fine-tuning what goes into the model."
        )
        st.button(
            "↺ Reset to defaults",
            on_click=_apply_defaults,
            args=(defaults,),
            width="stretch",
            help="Restore all fields to defaults: first-listed suburb and property type. "
            "Rooms, bedrooms, bathrooms, car spaces, land size, building area, and year built "
            "use training medians for that suburb × type (with type / suburb / global fallbacks). "
            "Location fields follow the suburb.",
        )

        st.markdown("**Location**")
        suburb = st.selectbox(
            "Suburb",
            ref["suburbs"],
            key=SS_SUBURB,
            help="Rare suburbs are grouped as “Other” in the model.",
        )
        if st.session_state.get("_suburb_prev") != suburb:
            st.session_state[SS_POSTCODE] = _postcode_for_suburb(ref, suburb, med)
            st.session_state[SS_REGION] = _region_for_suburb(ref, suburb)
            st.session_state[SS_DISTANCE] = _distance_for_suburb(ref, suburb, med)
            st.session_state[SS_LAT] = _lat_for_suburb(ref, suburb, med)
            st.session_state[SS_LON] = _lon_for_suburb(ref, suburb, med)
            st.session_state[SS_PROPCOUNT] = _propertycount_for_suburb(ref, suburb, med)
            p_cur = st.session_state.get(SS_TYPE, ref["types"][0])
            for k, v in _dwelling_values_from_profile(
                ref, med, suburb, p_cur
            ).items():
                st.session_state[k] = v
            st.session_state["_suburb_prev"] = suburb

        property_type = st.selectbox(
            "Property type",
            ref["types"],
            key=SS_TYPE,
            format_func=_format_property_type,
            help="**House (h)**, **townhouse (t)**, or **unit / apartment (u)**. When you change this, "
            "**rooms through year built** reset to training **medians for this suburb × type** "
            "(then type-only → suburb-only → global median if a cell is missing).",
        )
        if st.session_state.get("_ptype_prev") != property_type:
            for k, v in _dwelling_values_from_profile(
                ref, med, suburb, property_type
            ).items():
                st.session_state[k] = v
            st.session_state["_ptype_prev"] = property_type

        st.divider()
        st.markdown("**Dwelling**")
        rooms = st.slider(
            "Rooms",
            1,
            10,
            key=SS_ROOMS,
            help="Medians for suburb × property type when Suburb or Property type changes. "
            "Training data max is 10.",
        )
        bedroom2 = st.slider(
            "Bedrooms",
            0,
            20,
            key=SS_BEDROOM2,
            help="Medians for suburb × property type when Suburb or Property type changes. "
            "Matches `Bedroom2` in the data (max 20 in the training set).",
        )
        bathroom = st.slider(
            "Bathrooms",
            0,
            8,
            key=SS_BATHROOM,
            help="Medians for suburb × property type when Suburb or Property type changes. "
            "Training data max is 8.",
        )
        car = st.slider(
            "Car spaces",
            0,
            10,
            key=SS_CAR,
            help="Medians for suburb × property type when Suburb or Property type changes. "
            "Training data max is 10.",
        )

        st.divider()
        st.markdown("**Land & building**")
        landsize = st.number_input(
            "Land size (m²)",
            min_value=0.0,
            step=10.0,
            key=SS_LANDSIZE,
            help="Medians for suburb × property type when Suburb or Property type changes.",
        )
        building_area = st.number_input(
            "Building area (m²)",
            min_value=0.0,
            step=5.0,
            key=SS_BUILDING,
            help="Medians for suburb × property type when Suburb or Property type changes.",
        )
        year_built = st.number_input(
            "Year built",
            min_value=1800,
            max_value=2030,
            step=1,
            key=SS_YEAR,
            help="Medians for suburb × property type when Suburb or Property type changes.",
        )

        st.divider()
        with st.expander("Advanced · model location inputs", expanded=False):
            st.selectbox(
                "Postcode",
                ref["postcodes"],
                key=SS_POSTCODE,
                format_func=_format_postcode,
                help=f"{len(ref['postcodes'])} postcodes in the training data. "
                "Usually matches **Suburb** unless you choose another.",
            )
            st.selectbox(
                "Region",
                ref["regions"],
                key=SS_REGION,
                help="Each suburb maps to exactly one region in this dataset. "
                "Follows **Suburb** unless you override.",
            )
            st.divider()
            st.slider(
                "Distance to CBD (km)",
                0.0,
                50.0,
                key=SS_DISTANCE,
                step=0.5,
                help="Median distance for sales in the selected suburb in training. "
                "Updates when **Suburb** changes.",
            )
            st.divider()
            lat = st.slider(
                "Latitude",
                ref["lat_bounds"][0],
                ref["lat_bounds"][1],
                key=SS_LAT,
                step=0.001,
                help="Suburb median from training (within min–max in the dataset). "
                "Updates with **Suburb**; adjust for a custom pin.",
            )
            lon = st.slider(
                "Longitude",
                ref["lon_bounds"][0],
                ref["lon_bounds"][1],
                key=SS_LON,
                step=0.001,
                help="Suburb median from training (within min–max in the dataset). "
                "Updates with **Suburb**; adjust for a custom pin.",
            )
            propertycount = st.number_input(
                "Sales in suburb (count)",
                min_value=0.0,
                step=50.0,
                key=SS_PROPCOUNT,
                help="`Propertycount` in the dataset — one value per suburb. Override if needed.",
            )

    postcode = float(st.session_state[SS_POSTCODE])
    distance = float(st.session_state[SS_DISTANCE])
    regionname = str(st.session_state[SS_REGION])

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
    unc_range = _dollar_range_from_uncertainty(bundle, pred_log)

    left, right = st.columns([1, 1.15], gap="large")
    with left:
        with st.container(border=True):
            st.markdown("**Model estimate (sale price)**")
            # Escape $ as \$ — bare $ starts LaTeX/math mode in Streamlit markdown (monospace font).
            st.markdown(f"# \\${price:,.0f}")
            if unc_range is not None:
                lo_u, hi_u = unc_range
                st.markdown(
                    f"**Approx. range (uncertainty):** \\${lo_u:,.0f} – \\${hi_u:,.0f} AUD"
                )
            uinf = bundle.get("uncertainty") or {}
            if unc_range is not None and uinf.get("log_residual_p05") is not None:
                st.caption(
                    "AUD · Range uses **5th–95th percentile** of holdout **log-price** errors "
                    "(same split as training metrics). Heuristic band, not a formal prediction "
                    "interval — illustrative only, not a valuation."
                )
            elif unc_range is not None:
                st.caption(
                    "AUD · Range uses **± holdout RMSE** (older bundle). Illustrative only, "
                    "not a valuation."
                )
            else:
                st.caption(
                    "AUD · Re-train with `python -m src.train` to bundle uncertainty for a range. "
                    "Illustrative only, not a valuation."
                )

        recent = _latest_sales_in_suburb(suburb, 3)
        if len(recent) > 0:
            st.markdown("**Latest sales in this suburb**")
            st.caption(
                "Three most recent **dated** rows for **"
                + suburb
                + "** in the training file — historical sample only, not live or complete."
            )
            disp = recent.assign(
                When=recent["Date"].dt.strftime("%d %b %Y"),
                Price_aud=recent["Price"].apply(lambda x: f"${float(x):,.0f}"),
                Dwelling=recent["Type"].map(lambda c: PROPERTY_TYPE_LABELS.get(str(c), str(c))),
            )[["When", "Address", "Price_aud", "Dwelling", "Rooms"]]
            disp = disp.rename(
                columns={
                    "When": "Sale date",
                    "Price_aud": "Price",
                    "Dwelling": "Type",
                }
            )
            st.dataframe(disp, width="stretch", hide_index=True)
        else:
            st.caption("No dated sales rows for this suburb in the training file.")

    with right:
        st.markdown("**What drives the model estimate**")
        imp = bundle.get("feature_importances") or []
        if not imp:
            imp = top_feature_importances_from_pipeline(pipeline)
        if imp:
            st.caption(
                "Permutation importance on raw inputs (higher = stronger effect when shuffled)."
            )
            chart_data = (
                pd.DataFrame(imp)
                .set_index("feature")["importance"]
                .sort_values(ascending=True)
            )
            st.bar_chart(chart_data, height=320)
        else:
            st.info(
                "No importance data in the bundle. Re-run `python -m src.train` "
                "and redeploy with the new `models/price_model.joblib`."
            )

    st.divider()
    st.markdown(f"### Suburb snapshot · **{suburb}**")
    st.caption(
        "Quick orientation from the historical sales file — **not** used to compute the estimate "
        "above."
    )
    g_lat, g_lon = _buyer_guide_map_point(ref, suburb, med)
    snap_left, snap_right = st.columns([1.1, 1], gap="large")
    with snap_left:
        st.map(pd.DataFrame({"lat": [g_lat], "lon": [g_lon]}), width="stretch")
        st.caption(
            "Pin: **median lat/lon** of listings in this suburb in the training sample "
            "(not a suburb boundary; ignores Advanced location sliders)."
        )
    with snap_right:
        n_rows = int(ref["suburb_listing_count"].get(suburb, 0))
        med_km = _distance_for_suburb(ref, suburb, med)
        pc_g = _postcode_for_suburb(ref, suburb, med)
        reg_g = _region_for_suburb(ref, suburb)
        prop_ct = _propertycount_for_suburb(ref, suburb, med)
        st.markdown(
            f"- **Broad region:** {reg_g}\n"
            f"- **Typical postcode:** {_format_postcode(pc_g)}\n"
            f"- **Median distance to CBD** (in-sample sales): **{med_km:.1f} km**\n"
            f"- **Sales rows in this dataset** for {suburb}: **{n_rows:,}**\n"
            f"- **`Propertycount` in file** (suburb statistic): **{prop_ct:,.0f}**\n"
        )
        st.info(
            "This block is a **qualitative** buyer’s-aid only. The **price** comes solely from "
            "the model inputs in the sidebar."
        )

    with st.expander("Model details (holdout metrics)"):
        metrics = bundle.get("metrics", {})
        st.json(metrics)
        if bundle.get("uncertainty"):
            st.caption("Used for the **approx. dollar range** (log-residual percentiles on holdout).")
            st.json(bundle["uncertainty"])
        if METRICS_PATH.is_file():
            st.caption(f"Mirrored in `{METRICS_PATH.name}` for reference.")
            st.code(json.dumps(json.loads(METRICS_PATH.read_text(encoding="utf-8")), indent=2))


if __name__ == "__main__":
    main()
