"""
Scikit-learn Gradient Boosting builder for classification and regression.

Default hyper-parameters
------------------------
n_estimators  : 300   – balanced depth vs. speed; no early-stopping in sklearn GBM
learning_rate : 0.05  – same conservative rate used across all boosting models
max_depth     : 4     – shallower than XGBoost (sklearn GBM is more sensitive to depth)
subsample     : 0.8   – stochastic gradient boosting; reduces variance
"""
from typing import Any, Dict, Literal, Optional

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

Task = Literal["classification", "regression"]

_CLASSIFIER_DEFAULTS: Dict[str, Any] = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 4,
    "subsample": 0.8,
    "random_state": 42,
}

_REGRESSOR_DEFAULTS: Dict[str, Any] = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 4,
    "subsample": 0.8,
    "random_state": 42,
}


def build_model(
    task: Task,
    params: Optional[Dict[str, Any]] = None,
) -> GradientBoostingClassifier | GradientBoostingRegressor:
    """Return an unfitted Gradient Boosting estimator.

    Parameters
    ----------
    task:
        ``"classification"`` → :class:`~sklearn.ensemble.GradientBoostingClassifier`.
        ``"regression"``     → :class:`~sklearn.ensemble.GradientBoostingRegressor`.
    params:
        Optional dict that overrides the defaults documented above.

    Returns
    -------
    Unfitted scikit-learn estimator.
    """
    if task == "classification":
        hp = {**_CLASSIFIER_DEFAULTS, **(params or {})}
        return GradientBoostingClassifier(**hp)
    elif task == "regression":
        hp = {**_REGRESSOR_DEFAULTS, **(params or {})}
        return GradientBoostingRegressor(**hp)
    else:
        raise ValueError(f"Unknown task '{task}'. Expected 'classification' or 'regression'.")
