"""Feature engineering.

Every player in a team (groupId) shares the same winPlacePerc, so the model
works at the team level: player stats are engineered, then aggregated per
team, then ranked against the other teams in the same match. A team's
placement depends on how it compares to its opponents, which the per-match
percentile ranks capture directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import TARGET

# Player-level columns that get aggregated per team.
AGG_COLS = [
    "assists", "boosts", "damageDealt", "DBNOs", "headshotKills", "heals",
    "killPlace", "kills", "killStreaks", "longestKill", "revives",
    "rideDistance", "swimDistance", "walkDistance", "weaponsAcquired",
    "totalDistance", "healsAndBoosts", "headshotRate", "killsPerDistance",
    "damageNorm", "killsNorm",
]
# Columns whose per-match rank is added (mean and max of the team).
RANK_COLS = [
    "walkDistance", "killPlace", "boosts", "heals", "weaponsAcquired",
    "damageDealt", "kills", "totalDistance", "healsAndBoosts", "killStreaks",
    "longestKill", "DBNOs", "assists", "revives", "rideDistance",
]


def add_player_features(df: pd.DataFrame) -> pd.DataFrame:
    df["playersJoined"] = df.groupby("matchId", observed=True)["matchId"].transform("size").astype("int16")
    norm = ((100 - df["playersJoined"]) / 100 + 1).astype("float32")
    df["totalDistance"] = (df["walkDistance"] + df["rideDistance"] + df["swimDistance"]).astype("float32")
    df["healsAndBoosts"] = (df["heals"] + df["boosts"]).astype("int16")
    df["headshotRate"] = (df["headshotKills"] / df["kills"].replace(0, np.nan)).fillna(0).astype("float32")
    df["killsPerDistance"] = (df["kills"] / (df["totalDistance"] + 1)).astype("float32")
    # Scale by lobby size: a kill in a 50-player match is worth more than in a 100-player one.
    df["killsNorm"] = (df["kills"] * norm).astype("float32")
    df["damageNorm"] = (df["damageDealt"] * norm).astype("float32")
    return df


def build_group_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per team with aggregated, ranked and match-context features."""
    df = add_player_features(df)
    keys = ["matchId", "groupId"]
    g = df.groupby(keys, observed=True, sort=False)

    mean = g[AGG_COLS].mean().astype("float32").add_suffix("_mean")
    mx = g[AGG_COLS].max().astype("float32").add_suffix("_max")
    mn = g[["killPlace", "walkDistance", "boosts", "heals"]].min().astype("float32").add_suffix("_min")
    size = g.size().rename("groupSize").astype("int16")
    first = g[["matchDuration", "maxPlace", "numGroups", "playersJoined", "matchTypeGroup"]].first()
    out = pd.concat([mean, mx, mn, size, first], axis=1)
    if TARGET in df.columns:
        out[TARGET] = g[TARGET].first().astype("float32")
    del mean, mx, mn, size, first

    out = out.reset_index()
    mg = out.groupby("matchId", observed=True)
    for c in RANK_COLS:
        for agg in ("mean", "max"):
            col = f"{c}_{agg}"
            out[f"{col}_rank"] = mg[col].rank(pct=True).astype("float32")
    out["groupsInMatch"] = mg["groupId"].transform("size").astype("int16")
    out["matchTypeGroup"] = out["matchTypeGroup"].cat.codes.astype("int8")
    return out


def feature_columns(table: pd.DataFrame) -> list[str]:
    drop = {"matchId", "groupId", TARGET}
    return [c for c in table.columns if c not in drop]
