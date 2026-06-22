"""
Data loading and train/val/test splitting for the NBA prediction pipeline.
"""
import logging
from pathlib import Path
from typing import Literal, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Update this constant whenever the feature dataset is regenerated with a different season range.
DATASET_VERSION = "2000-01_2025-26"
FEATURES_PATH = PROJECT_ROOT / "data" / "3_features" / f"features_nba_data_{DATASET_VERSION}.csv"

# Columns that carry no predictive signal and must be removed before training
_META_COLS = [
    "game_id",
    "game_date",
    "season",
    "home_team_id",
    "away_team_id",
    "home_team_abbreviation",
    "away_team_abbreviation",
    "home_wl",
    "away_wl",
    "home_pts",
    "away_pts",
    "split",
]

Task = Literal["classification", "regression"]

SplitData = Tuple[
    Tuple[pd.DataFrame, pd.Series],  # train
    Tuple[pd.DataFrame, pd.Series],  # val
    Tuple[pd.DataFrame, pd.Series],  # test
]


def load_splits(
    task: Task = "classification",
    path: Path = FEATURES_PATH,
) -> SplitData:
    """Load the features CSV and return (X_train, y_train), (X_val, y_val), (X_test, y_test).

    Parameters
    ----------
    task:
        ``"classification"``  → target is ``home_win`` (binary int).
        ``"regression"``      → target is ``point_differential`` (float).
    path:
        Path to the features CSV; defaults to the canonical project path.

    Returns
    -------
    Three ``(X, y)`` tuples in train / val / test order.
    """
    df = pd.read_csv(path)
    logger.info("Loaded %d rows, %d columns from %s", len(df), df.shape[1], path)

    if task == "classification":
        target_col = "home_win"
        drop_targets = ["point_differential"]
    else:
        target_col = "point_differential"
        drop_targets = ["home_win"]

    drop_cols = _META_COLS + drop_targets
    # Keep only columns that actually exist in this version of the CSV
    drop_cols = [c for c in drop_cols if c in df.columns]

    for split_name, split_df in df.groupby("split", sort=False):
        n = len(split_df)
        if task == "classification":
            # For binary targets value_counts gives a compact, readable distribution.
            target_dist = split_df[target_col].value_counts(normalize=True).to_dict()
            logger.info(
                "split=%-5s  rows=%d  target_dist=%s",
                split_name,
                n,
                {k: round(v, 3) for k, v in target_dist.items()},
            )
        else:
            # For continuous targets value_counts would produce hundreds of keys; log summary stats instead.
            logger.info(
                "split=%-5s  rows=%d  target_mean=%.4f  target_std=%.4f",
                split_name,
                n,
                split_df[target_col].mean(),
                split_df[target_col].std(),
            )

    def _extract(mask: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        sub = df[mask].copy()
        y = sub[target_col].copy()
        X = sub.drop(columns=drop_cols + [target_col], errors="ignore")
        return X, y

    train_mask = df["split"] == "train"
    val_mask = df["split"] == "val"
    test_mask = df["split"] == "test"

    train = _extract(train_mask)
    val = _extract(val_mask)
    test = _extract(test_mask)

    for name, (X, y) in zip(("train", "val", "test"), (train, val, test)):
        logger.info("split=%-5s  X=%s  y=%s", name, X.shape, y.shape)

    return train, val, test
