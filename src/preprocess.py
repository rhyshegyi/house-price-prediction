"""Feature column definitions and suburb grouping for the modelling pipeline.

This module provides:
- The canonical feature columns the sklearn pipeline expects, and
- A small transformer (`SuburbGrouper`) that collapses rare suburbs into an
  `"Other"` bucket to keep one-hot encoding stable.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Columns used as model inputs (order matters for feature-name diagnostics).
CAT_COLS = ["Suburb", "Type", "Regionname"]
NUM_COLS = [
    "Rooms",
    "Distance",
    "Postcode",
    "Bedroom2",
    "Bathroom",
    "Car",
    "Landsize",
    "BuildingArea",
    "YearBuilt",
    "Lattitude",
    "Longtitude",
    "Propertycount",
]
FEATURE_COLS = CAT_COLS + NUM_COLS


class SuburbGrouper(BaseEstimator, TransformerMixin):
    """Collapse rare suburb labels into a single 'Other' bucket."""

    def __init__(self, column: str = "Suburb", min_count: int = 20, other_label: str = "Other"):
        # `min_count` controls how many samples a suburb must have to be kept
        # as its own category during training.
        self.column = column
        self.min_count = min_count
        self.other_label = other_label

    def fit(self, X, y=None):
        # Learn which suburb labels survive (vs get mapped to `"Other"`).
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        counts = X[self.column].value_counts()
        self.keep_: set = set(counts[counts >= self.min_count].index)
        return self

    def transform(self, X):
        # Apply the learned keep-vs-Other mapping to new data.
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=FEATURE_COLS)
        out = X.copy()
        out[self.column] = out[self.column].apply(
            lambda s: s if s in self.keep_ else self.other_label
        )
        return out


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return only model feature columns in the canonical order."""
    # The model training script and the Streamlit app both rely on this
    # canonical column order.
    missing = set(FEATURE_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df[FEATURE_COLS].copy()


def group_suburb_series(series: pd.Series, keep: Iterable[str], other: str = "Other") -> pd.Series:
    """Apply rare grouping using a fixed set of kept labels (for inspection)."""
    keep_set = set(keep)
    return series.apply(lambda s: s if s in keep_set else other)
