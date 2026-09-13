"""
preprocessing.py
-----------------
Builds the sklearn ColumnTransformer used to turn the 12 permitted raw
features into a numeric matrix a tree model can consume.

Design choices (see README for full justification):
  * Numeric weather / geographic features (X, Y, FFMC, DMC, DC, ISI,
    temp, RH, wind, rain) are passed through untouched. Tree-based
    models are invariant to monotonic rescaling, so StandardScaler
    would add complexity with zero benefit here.
  * `month` and `day` are nominal categories with no natural numeric
    order (encoding them as 1..12 / 0..6 would invent a false ordinal
    relationship, e.g. implying "dec" is "closer" to "jan" than to
    "jun" in a linear sense that a tree could split on incorrectly
    via a single threshold). We use OneHotEncoder(handle_unknown="ignore")
    inside a ColumnTransformer so unseen categories at evaluation time
    (e.g. a month absent from the training fold) do not crash the
    pipeline -- they are silently encoded as an all-zero row instead.
  * The transformer is only ever `.fit()` on a training partition
    (inside a Pipeline together with the model), never on validation
    or evaluation data, so no statistics leak across the train/eval
    boundary.
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# The only features the problem statement permits as model inputs.
NUMERIC_FEATURES = ["X", "Y", "FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", "rain"]
CATEGORICAL_FEATURES = ["month", "day"]
PERMITTED_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Return an unfitted ColumnTransformer.

    Must be fit exclusively on a training fold / training partition.
    """
    return ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",  # any extra columns (e.g. 'area') are dropped, never leaked in
    )


def validate_columns(df):
    """Raise a clear error if a required permitted feature is missing."""
    missing = [c for c in PERMITTED_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Input dataframe is missing required feature columns: {missing}")
