"""
model.py
--------
Lightweight tree-based regressors, trained to predict y = log1p(area).

Only three model families are considered, per the hackathon time budget:
Random Forest, Extra Trees, and Gradient Boosting. All are CPU-only,
train in well under a second on ~500 rows, and are far less prone to
overfitting a 500-row dataset than anything more flexible (e.g. deep
GBM stacks, neural nets).

Hyperparameters are deliberately conservative (shallow depth, min
leaf size >= 2) because with only ~500 rows and 80/20 CV folds of
~100 validation rows, an overly flexible model will fit training
noise and produce an excellent mean fold score but a terrible worst
fold score -- exactly the failure mode the "worst-split NDCG@25"
scoring component (20 pts) is designed to punish.
"""

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
)
from sklearn.pipeline import Pipeline

from preprocessing import build_preprocessor

RANDOM_STATE = 42


def make_random_forest():
    return RandomForestRegressor(
        n_estimators=500,
        max_depth=6,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def make_extra_trees():
    # max_depth=4 / min_samples_leaf=5 (shallower than the original starting
    # point of max_depth=6 / min_samples_leaf=2) was chosen after an explicit
    # depth-vs-ranking-quality investigation (see investigate.py section B
    # and INVESTIGATION.md): across 5 different KFold seeds, this shallower
    # configuration beat the deeper default on mean NDCG@25 in 5/5 seeds and
    # on worst-fold NDCG@25 in 4/5 seeds, with comparable recall and
    # Spearman. A 500-row dataset gives a depth-6 tree room to fit noise;
    # capping depth at 4 with a larger minimum leaf size regularizes that
    # away without giving up ranking quality.
    return ExtraTreesRegressor(
        n_estimators=500,
        max_depth=4,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def make_gradient_boosting():
    return GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.03,
        max_depth=2,
        random_state=RANDOM_STATE,
    )


MODEL_FACTORIES = {
    "random_forest": make_random_forest,
    "extra_trees": make_extra_trees,
    "gradient_boosting": make_gradient_boosting,
}


def build_pipeline(model_name: str) -> Pipeline:
    """Return an unfitted Pipeline(preprocessor -> regressor).

    Wrapping the preprocessor and the regressor in a single sklearn
    Pipeline guarantees that when we call `.fit(X_train, y_train)`
    the OneHotEncoder statistics are learned ONLY on X_train, and that
    `.predict(X_eval)` reuses those exact fitted statistics -- this is
    what makes "fit preprocessing only on the training fold" automatic
    and mistake-proof rather than a manual discipline we could forget.
    """
    if model_name not in MODEL_FACTORIES:
        raise ValueError(f"Unknown model '{model_name}'. Options: {list(MODEL_FACTORIES)}")
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("regressor", MODEL_FACTORIES[model_name]()),
        ]
    )


def ensemble_predict(predictions: dict, weights: dict):
    """Combine per-model prediction arrays into a single weighted average.

    predictions: {"random_forest": np.array, "extra_trees": np.array, ...}
    weights:     {"random_forest": 0.4, "extra_trees": 0.4, "gradient_boosting": 0.2}
    Weights are validated to sum to 1.0 (within floating point tolerance).
    """
    total_weight = sum(weights.values())
    if abs(total_weight - 1.0) > 1e-6:
        raise ValueError(f"Ensemble weights must sum to 1.0, got {total_weight}")
    combined = None
    for name, w in weights.items():
        contribution = w * predictions[name]
        combined = contribution if combined is None else combined + contribution
    return combined
