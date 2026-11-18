"""Loading and cleaning for the PUBG Finish Placement dataset (train_V2.csv).

The raw file has 4.4M player rows and 29 columns. Memory matters at that size,
so the loader reads with explicit compact dtypes and stores the id columns as
categoricals instead of Python strings.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ID_COLS = ["Id", "groupId", "matchId"]
TARGET = "winPlacePerc"

INT_COLS = [
    "assists", "boosts", "DBNOs", "headshotKills", "heals", "killPlace",
    "killPoints", "kills", "killStreaks", "matchDuration", "maxPlace",
    "numGroups", "rankPoints", "revives", "roadKills", "teamKills",
    "vehicleDestroys", "weaponsAcquired", "winPoints",
]
FLOAT_COLS = [
    "damageDealt", "longestKill", "rideDistance", "swimDistance",
    "walkDistance", TARGET,
]

# Kaggle's 16 match types collapse to three team sizes plus event modes.
MATCH_TYPE_GROUP = {
    "solo": "solo", "solo-fpp": "solo", "normal-solo": "solo", "normal-solo-fpp": "solo",
    "duo": "duo", "duo-fpp": "duo", "normal-duo": "duo", "normal-duo-fpp": "duo",
    "squad": "squad", "squad-fpp": "squad", "normal-squad": "squad", "normal-squad-fpp": "squad",
    "crashfpp": "event", "crashtpp": "event", "flarefpp": "event", "flaretpp": "event",
}


def load_raw(path: str, nrows: int | None = None) -> pd.DataFrame:
    """Read train_V2.csv with compact dtypes (about 0.5 GB in memory for the full file)."""
    dtypes = {c: "category" for c in ID_COLS}
    dtypes.update({c: "int32" for c in INT_COLS})
    dtypes.update({c: "float32" for c in FLOAT_COLS})
    dtypes["matchType"] = "category"
    df = pd.read_csv(path, dtype=dtypes, nrows=nrows)
    # Narrow further where the value range allows it.
    for c in INT_COLS:
        lo, hi = df[c].min(), df[c].max()
        if lo >= np.iinfo(np.int16).min and hi <= np.iinfo(np.int16).max:
            df[c] = df[c].astype("int16")
    return df


@dataclass
class CleaningReport:
    rows_in: int = 0
    rows_out: int = 0
    steps: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"rows_in": self.rows_in, "rows_out": self.rows_out, "steps": self.steps}


def validate_schema(df: pd.DataFrame) -> None:
    expected = set(ID_COLS + INT_COLS + FLOAT_COLS + ["matchType"])
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    bad = df[TARGET].dropna()
    if ((bad < 0) | (bad > 1)).any():
        raise ValueError("winPlacePerc outside [0, 1]")


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Rows that cannot be scored at all. Safe to apply to train and evaluation data alike."""
    validate_schema(df)
    rep = CleaningReport(rows_in=len(df))

    mask = df[TARGET].isna()
    rep.steps["null_target"] = int(mask.sum())
    df = df[~mask]

    # With maxPlace <= 1 the placement percentile is undefined (always 0).
    mask = df["maxPlace"] <= 1
    rep.steps["single_place_match"] = int(mask.sum())
    df = df[~mask]

    df = df.copy()
    df["matchTypeGroup"] = df["matchType"].map(MATCH_TYPE_GROUP).astype("category")
    rep.steps["unknown_match_type"] = int(df["matchTypeGroup"].isna().sum())
    df = df[df["matchTypeGroup"].notna()]

    rep.rows_out = len(df)
    return df, rep


def anomaly_mask(df: pd.DataFrame) -> pd.Series:
    """Rows whose stats are physically implausible (likely cheaters or logging bugs).

    Used to filter the training split only, so the reported test error is on
    untouched data.
    """
    total_dist = df["walkDistance"] + df["rideDistance"] + df["swimDistance"]
    kills_without_moving = (df["kills"] > 0) & (total_dist == 0)
    too_many_road_kills = df["roadKills"] > 10
    impossible_kills = df["kills"] > 30
    long_snipe = df["longestKill"] >= 1000
    many_weapons_no_walk = (df["weaponsAcquired"] > 50) & (df["walkDistance"] < 100)
    return (kills_without_moving | too_many_road_kills | impossible_kills
            | long_snipe | many_weapons_no_walk)
