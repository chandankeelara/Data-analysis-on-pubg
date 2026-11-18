"""Models for predicting a team's finish percentile (winPlacePerc)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def split_by_match(match_ids: pd.Series, test_frac: float = 0.2, seed: int = 42) -> np.ndarray:
    """Boolean test mask that keeps every match wholly in train or in test.

    Teams in the same match share rank features, so a row-level split would
    leak information across the split.
    """
    codes = match_ids.cat.codes.to_numpy() if hasattr(match_ids, "cat") else pd.factorize(match_ids)[0]
    rng = np.random.default_rng(seed)
    is_test_match = rng.random(codes.max() + 1) < test_frac
    return is_test_match[codes]


def weighted_mae(y: np.ndarray, pred: np.ndarray, weight: np.ndarray) -> float:
    """MAE over players: a team's error counts once per member, as in the Kaggle metric."""
    return float(np.average(np.abs(y - pred), weights=weight))


def fit_ridge(X: pd.DataFrame, y: np.ndarray, w: np.ndarray):
    model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    model.fit(X, y, ridge__sample_weight=w)
    return model


def fit_lgbm(X_tr, y_tr, w_tr, X_va, y_va, w_va, seed: int = 42):
    import lightgbm as lgb

    params = dict(
        objective="regression_l1", learning_rate=0.05, num_leaves=127,
        min_child_samples=50, feature_fraction=0.7, bagging_fraction=0.8,
        bagging_freq=1, lambda_l2=1.0, n_estimators=3000, random_state=seed,
        n_jobs=4, verbose=-1,
    )
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_tr, y_tr, sample_weight=w_tr,
        eval_set=[(X_va, y_va)], eval_sample_weight=[w_va], eval_metric="l1",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return model


def rank_adjust(table: pd.DataFrame, pred: np.ndarray) -> np.ndarray:
    """Turn raw scores into valid placements.

    Within a match the true targets are evenly spaced: with maxPlace slots the
    possible values are k / (maxPlace - 1). This orders the teams by predicted
    score, spreads them evenly over [0, 1] and snaps each to that grid.
    """
    pred = np.clip(pred, 0.0, 1.0)
    df = pd.DataFrame({"matchId": table["matchId"].to_numpy(), "pred": pred})
    g = df.groupby("matchId", observed=True)["pred"]
    order = g.rank(method="first").to_numpy() - 1
    n = g.transform("size").to_numpy()
    max_place = table["maxPlace"].to_numpy()
    num_groups = table["numGroups"].to_numpy()

    gap = 1.0 / np.maximum(max_place - 1, 1)
    snapped = np.round(pred / gap) * gap
    # When every slot is filled and every team is present, the targets are
    # exactly 0, 1/(n-1), ..., 1, so the predicted order fixes the values.
    # Otherwise the empty slots' positions are unknown: only snap to the grid.
    full = (n == num_groups) & (num_groups == max_place) & (n > 1)
    spread = order / np.maximum(n - 1, 1)
    return np.clip(np.where(full, spread, snapped), 0.0, 1.0)
