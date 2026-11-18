"""End-to-end run: load -> clean -> features -> train -> evaluate -> report.

    python -m pubg.train --data data/train_V2.csv --out reports
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data import TARGET, anomaly_mask, clean, load_raw
from .features import build_group_table, feature_columns
from .model import fit_lgbm, fit_ridge, rank_adjust, split_by_match, weighted_mae


def log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train_V2.csv")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--match-frac", type=float, default=1.0, help="keep this share of whole matches (quick runs)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    raw = load_raw(args.data)
    if args.match_frac < 1.0:
        keep = split_by_match(raw["matchId"], test_frac=args.match_frac, seed=args.seed + 7)
        raw = raw[keep].copy()
    log(f"loaded {len(raw):,} rows x {raw.shape[1]} cols, {raw.memory_usage(deep=True).sum() / 1e6:.0f} MB", t0)

    df, rep = clean(raw)
    del raw
    gc.collect()
    log(f"cleaned: {rep.as_dict()}", t0)

    # Split by match before any filtering so the test set stays untouched.
    test_mask = split_by_match(df["matchId"], test_frac=0.2, seed=args.seed)
    train_players = df[~test_mask]
    test_players = df[test_mask]
    del df
    gc.collect()

    anom = anomaly_mask(train_players)
    n_anom = int(anom.sum())
    train_players = train_players[~anom]
    log(f"train players {len(train_players):,} (removed {n_anom:,} anomalous), test players {len(test_players):,}", t0)

    train = build_group_table(train_players)
    del train_players
    gc.collect()
    test = build_group_table(test_players)
    n_test_players = len(test_players)
    del test_players
    gc.collect()
    feats = feature_columns(train)
    log(f"group tables: train {len(train):,} teams, test {len(test):,} teams, {len(feats)} features", t0)

    # Hold out 10% of training matches for early stopping.
    va_mask = split_by_match(train["matchId"], test_frac=0.1, seed=args.seed + 1)
    tr, va = train[~va_mask], train[va_mask]

    y_te = test[TARGET].to_numpy()
    w_te = test["groupSize"].to_numpy()
    results = {}

    # Baselines.
    mean_pred = np.full(len(test), float(np.average(tr[TARGET], weights=tr["groupSize"])))
    results["baseline_mean"] = weighted_mae(y_te, mean_pred, w_te)
    kp = 1 - test["killPlace_mean_rank"].to_numpy()  # better (lower) killPlace -> higher placement
    results["baseline_killPlace_rank"] = weighted_mae(y_te, rank_adjust(test, kp), w_te)

    # Linear (statistical) model.
    ridge = fit_ridge(tr[feats], tr[TARGET].to_numpy(), tr["groupSize"].to_numpy())
    p = np.clip(ridge.predict(test[feats]), 0, 1)
    results["ridge"] = weighted_mae(y_te, p, w_te)
    results["ridge_rank_adjusted"] = weighted_mae(y_te, rank_adjust(test, p), w_te)
    from sklearn.metrics import r2_score
    r2_ridge = float(r2_score(y_te, p, sample_weight=w_te))
    log(f"ridge MAE {results['ridge']:.4f}", t0)

    # Gradient-boosted trees.
    lgbm = fit_lgbm(tr[feats], tr[TARGET].to_numpy(), tr["groupSize"].to_numpy(),
                    va[feats], va[TARGET].to_numpy(), va["groupSize"].to_numpy(), seed=args.seed)
    p = np.clip(lgbm.predict(test[feats]), 0, 1)
    results["lightgbm"] = weighted_mae(y_te, p, w_te)
    results["lightgbm_rank_adjusted"] = weighted_mae(y_te, rank_adjust(test, p), w_te)
    r2_lgbm = float(r2_score(y_te, p, sample_weight=w_te))
    log(f"lightgbm MAE {results['lightgbm']:.4f}, rank-adjusted {results['lightgbm_rank_adjusted']:.4f}", t0)

    # Per match-type error for the best model.
    best = rank_adjust(test, p)
    by_type = {}
    codes = dict(enumerate(["duo", "event", "solo", "squad"]))  # category codes are alphabetical
    for code, name in codes.items():
        m = test["matchTypeGroup"].to_numpy() == code
        if m.any():
            by_type[name] = round(weighted_mae(y_te[m], best[m], w_te[m]), 4)

    imp = pd.Series(lgbm.booster_.feature_importance("gain"), index=feats)
    imp = (imp / imp.sum()).sort_values(ascending=False)
    imp.round(4).to_csv(out / "feature_importance.csv", header=["gain_share"])

    corr = train[[f for f in feats if f.endswith("_rank")] + [TARGET]].corr(method="spearman")[TARGET]
    corr = corr.drop(TARGET).sort_values(key=np.abs, ascending=False)

    report = {
        "data": {
            "cleaning": rep.as_dict(),
            "train_anomalies_removed": n_anom,
            "test_players": n_test_players,
            "train_teams": int(len(train)),
            "test_teams": int(len(test)),
            "features": len(feats),
        },
        "test_mae_player_weighted": {k: round(v, 4) for k, v in results.items()},
        "test_r2": {"ridge": round(r2_ridge, 4), "lightgbm": round(r2_lgbm, 4)},
        "lightgbm_best_iteration": int(lgbm.best_iteration_ or lgbm.n_estimators),
        "best_model_mae_by_match_type": by_type,
        "top_features_by_gain": imp.head(10).round(4).to_dict(),
        "top_spearman_with_target": corr.head(10).round(4).to_dict(),
        "runtime_seconds": round(time.time() - t0, 1),
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    log("wrote " + str(out / "metrics.json"), t0)
    print(json.dumps(report["test_mae_player_weighted"], indent=2))


if __name__ == "__main__":
    main()
