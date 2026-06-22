"""
Preprocessing pipeline: null imputation, bool casting, optional scaling.

Usage
-----
    preprocessor = fit_preprocessor(X_train)
    X_train_t = transform(X_train, preprocessor)
    X_val_t   = transform(X_val,   preprocessor)
    X_test_t  = transform(X_test,  preprocessor)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical feature set
# ---------------------------------------------------------------------------

# Differential features (home − away)
DIFF_FEATURES: list[str] = [
    "net_rating_rolling_10_diff",
    "avg_pts_last_10_diff",
    "avg_ts_pct_last_10_diff",
    "avg_pace_last_10_diff",
    "winrate_last_10_diff",
    "streak_diff",
    "rest_days_diff",
    "fg_pct_diff",
    "fg3_pct_diff",
    "ft_pct_diff",
    "ts_pct_zscore_diff",
    "pace_zscore_diff",
    "win_rate_prev_season_diff",
    "is_back_to_back_diff",
]

# Per-team rolling / seasonal stats
HOME_FEATURES: list[str] = [
    "home_net_rating_rolling_10",
    "home_avg_pts_last_10",
    "home_avg_ts_pct_last_10",
    "home_avg_pace_last_10",
    "home_winrate_last_10",
    "home_streak",
    "home_rest_days",
    "home_is_back_to_back",
    "home_fg_pct",
    "home_fg3_pct",
    "home_ft_pct",
    "home_ts_pct_zscore",
    "home_pace_zscore",
    "home_win_rate_prev_season",
]

AWAY_FEATURES: list[str] = [
    "away_net_rating_rolling_10",
    "away_avg_pts_last_10",
    "away_avg_ts_pct_last_10",
    "away_avg_pace_last_10",
    "away_winrate_last_10",
    "away_streak",
    "away_rest_days",
    "away_is_back_to_back",
    "away_fg_pct",
    "away_fg3_pct",
    "away_ft_pct",
    "away_ts_pct_zscore",
    "away_pace_zscore",
    "away_win_rate_prev_season",
]

# Contextual / game-level features
CONTEXTUAL_FEATURES: list[str] = [
    "season_zscore",
    "is_covid_bubble",
]

# Full declared feature set
FEATURE_COLS: list[str] = HOME_FEATURES + AWAY_FEATURES + DIFF_FEATURES + CONTEXTUAL_FEATURES

# Columns that require median imputation (nulls present in first-season rows)
_IMPUTE_COLS: list[str] = [
    "home_win_rate_prev_season",
    "away_win_rate_prev_season",
    "win_rate_prev_season_diff",
    "ft_pct_diff",
    "away_ft_pct",
]

# Boolean columns that must be cast to int before scikit-learn estimators
_BOOL_COLS: list[str] = [
    "home_is_back_to_back",
    "away_is_back_to_back",
]


# ---------------------------------------------------------------------------
# Preprocessor state
# ---------------------------------------------------------------------------

@dataclass
class Preprocessor:
    """Holds all fitted statistics so train statistics are applied to val/test."""
    medians: dict[str, float] = field(default_factory=dict)
    scaler: Optional[StandardScaler] = None
    feature_cols: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_preprocessor(
    X_train: pd.DataFrame,
    *,
    scale: bool = False,
) -> Preprocessor:
    """Fit imputation medians (and optional scaler) on *train* data only.

    Parameters
    ----------
    X_train:
        Raw training feature matrix (bool columns still as bool).
    scale:
        If ``True``, fit a :class:`~sklearn.preprocessing.StandardScaler`.
        Tree-based models do not need scaling, but it can be useful for
        future linear / distance-based comparisons.

    Returns
    -------
    A :class:`Preprocessor` instance ready to be passed to :func:`transform`.
    """
    prep = Preprocessor()

    X = _cast_bools(X_train)

    for col in _IMPUTE_COLS:
        if col in X.columns:
            median_val = float(X[col].median())
            prep.medians[col] = median_val
            logger.info("Imputation median  %-40s = %.4f", col, median_val)

    feature_cols = [c for c in FEATURE_COLS if c in X.columns]
    prep.feature_cols = feature_cols

    if scale:
        scaler = StandardScaler()
        scaler.fit(X[feature_cols].fillna(0))
        prep.scaler = scaler
        logger.info("StandardScaler fitted on %d features", len(feature_cols))

    logger.info("Preprocessor fitted — %d feature columns", len(feature_cols))
    return prep


def transform(X: pd.DataFrame, preprocessor: Preprocessor) -> pd.DataFrame:
    """Apply the fitted preprocessor to *any* split.

    Parameters
    ----------
    X:
        Raw feature matrix (same schema as the training data).
    preprocessor:
        A :class:`Preprocessor` returned by :func:`fit_preprocessor`.

    Returns
    -------
    Processed :class:`~pandas.DataFrame` with exactly the columns in
    ``preprocessor.feature_cols``.
    """
    X = _cast_bools(X).copy()

    for col, median_val in preprocessor.medians.items():
        if col in X.columns:
            n_missing = int(X[col].isna().sum())
            if n_missing:
                X[col] = X[col].fillna(median_val)
                logger.debug("Imputed %d nulls in '%s' with %.4f", n_missing, col, median_val)

    X = X[preprocessor.feature_cols]

    if preprocessor.scaler is not None:
        scaled = preprocessor.scaler.transform(X.fillna(0))
        X = pd.DataFrame(scaled, index=X.index, columns=X.columns)

    return X


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cast_bools(X: pd.DataFrame) -> pd.DataFrame:
    """Convert boolean columns to int (0/1) in-place."""
    X = X.copy()
    for col in _BOOL_COLS:
        if col in X.columns and X[col].dtype == bool:
            X[col] = X[col].astype(np.int8)
    return X
