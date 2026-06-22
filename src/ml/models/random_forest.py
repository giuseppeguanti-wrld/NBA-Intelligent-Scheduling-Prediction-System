"""
Random Forest builder for classification and regression tasks.

Default hyper-parameters
------------------------
n_estimators    : 300   – enough trees for stable OOB estimates without
                          excessive memory usage
max_depth       : None  – fully grown trees; regularised via min_samples_leaf
min_samples_leaf: 5     – light regularisation that reduces variance on NBA data
class_weight    : "balanced"  (classifier only) – corrects for the ~60 % home-
                  win prior so the model does not trivially predict the majority
"""
from typing import Any, Dict, Literal, Optional

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

Task = Literal["classification", "regression"]

_CLASSIFIER_DEFAULTS: Dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_leaf": 5,
    "class_weight": "balanced",
    "n_jobs": -1,
    "random_state": 42,
}

_REGRESSOR_DEFAULTS: Dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_leaf": 5,
    "n_jobs": -1,
    "random_state": 42,
}


def build_model(
    task: Task,
    params: Optional[Dict[str, Any]] = None,
) -> RandomForestClassifier | RandomForestRegressor:
    """Return an unfitted Random Forest estimator.

    Parameters
    ----------
    task:
        ``"classification"`` → :class:`~sklearn.ensemble.RandomForestClassifier`.
        ``"regression"``     → :class:`~sklearn.ensemble.RandomForestRegressor`.
    params:
        Optional dict of hyper-parameters that *override* the defaults.
        Unspecified keys fall back to the defaults documented above.

    Returns
    -------
    Unfitted scikit-learn estimator.
    """
    if task == "classification":
        hp = {**_CLASSIFIER_DEFAULTS, **(params or {})}
        return RandomForestClassifier(**hp)
    elif task == "regression":
        hp = {**_REGRESSOR_DEFAULTS, **(params or {})}
        return RandomForestRegressor(**hp)
    else:
        raise ValueError(f"Unknown task '{task}'. Expected 'classification' or 'regression'.")
