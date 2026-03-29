# House price prediction (Melbourne)

Predicts residential sale prices from listing features using a **scikit-learn** pipeline (rare-suburb grouping, median / mode imputation, one-hot encoding, **RandomForestRegressor**). The target is `log1p(Price)`; predictions are returned on the dollar scale with `expm1`. A **Streamlit** app loads the trained artifact and validates inputs with **Pydantic**.

## Setup

Python 3.10+ recommended.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

Place the Kaggle Melbourne Housing file as `data/melb_data.csv` (see [Melbourne Housing Market](https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot) or your existing copy).

## Train the model

From the repository root:

```bash
python -m src.train
```

This fits a **LinearRegression** baseline and a **RandomForestRegressor**, compares RMSE (AUD) on a holdout set, and saves the better model to `models/price_model.joblib` plus `models/training_metrics.json`.

## Run the app

```bash
streamlit run app.py
```

Open the local URL shown in the terminal. Use the sidebar to set property features; the main panel shows the estimated price, a feature-importance bar chart (for the tree model), and test-set metrics in an expander.

## Deploy (Streamlit Community Cloud)

1. Push this repo (include `models/price_model.joblib` if you want the app to run without a training step on the server, or run training in a build step).
2. Set the main file to `app.py` and Python version to match your environment.
3. Add the live URL to your resume or project page when ready.

## Project layout

| Path | Purpose |
|------|---------|
| `data/melb_data.csv` | Raw Melbourne Housing CSV |
| `notebooks/01_eda.ipynb` | EDA: nulls, distributions, correlation heatmap |
| `src/schemas.py` | `PropertyInput` (Pydantic) |
| `src/preprocess.py` | Feature columns, `SuburbGrouper` |
| `src/train.py` | Training, metrics, `joblib` export |
| `models/price_model.joblib` | Saved pipeline + metrics bundle (generated) |
| `app.py` | Streamlit UI |

## Screenshot

After `streamlit run app.py`, capture the browser window and save as `docs/screenshot.png` (optional).
