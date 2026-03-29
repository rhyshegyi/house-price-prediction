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

The hosted app **does not run training**; it expects these files **in the GitHub repo**:

1. `models/price_model.joblib` and `models/training_metrics.json` — run `python -m src.train` locally, then commit and push them.
2. `data/melb_data.csv` — required for suburb/type/region dropdowns and default medians (same file as local training).

Then on [share.streamlit.io](https://share.streamlit.io): deploy from your repo, main file `app.py`, branch `main` (or your default). Redeploy after each push.

If `git push` rejects a large `price_model.joblib`, use [Git LFS](https://git-lfs.com) for that file or trim the model in `src/train.py` (fewer trees).

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
