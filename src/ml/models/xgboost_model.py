"""
XGBoost builder for classification and regression tasks.

Default hyper-parameters
------------------------
n_estimators          : 500   – upper bound; early stopping will trim this down
learning_rate         : 0.05  – conservative step size for better generalisation
max_depth             : 6     – standard depth for tabular data
subsample             : 0.8   – row sub-sampling per tree (reduces overfitting)
colsample_bytree      : 0.8   – column sub-sampling per tree
early_stopping_rounds : 50    – stop when val metric does not improve for 50 rounds

Objectives
----------
classification : binary:logistic, eval_metric=auc
regression     : reg:squarederror, eval_metric=rmse
"""
from typing import Any, Dict, Literal, Optional

from xgboost import XGBClassifier, XGBRegressor

Task = Literal["classification", "regression"]

_CLASSIFIER_DEFAULTS: Dict[str, Any] = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "early_stopping_rounds": 50,
    "n_jobs": -1,
    "random_state": 42,
    "verbosity": 0,
}

_REGRESSOR_DEFAULTS: Dict[str, Any] = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "early_stopping_rounds": 50,
    "n_jobs": -1,
    "random_state": 42,
    "verbosity": 0,
}


def build_model(
    task: Task,
    params: Optional[Dict[str, Any]] = None,
) -> XGBClassifier | XGBRegressor:
    """Return an unfitted XGBoost estimator.

    Parameters
    ----------
    task:
        ``"classification"`` → :class:`~xgboost.XGBClassifier`.
        ``"regression"``     → :class:`~xgboost.XGBRegressor`.
    params:
        Optional dict that overrides the defaults documented above.

    Notes
    -----
    Early stopping requires an ``eval_set`` to be passed at ``fit()`` time.
    :func:`~src.trainer.train_all` handles this automatically by forwarding
    the validation set as ``eval_set=[(X_val, y_val)]``.

    Returns
    -------
    Unfitted XGBoost estimator.
    """
    if task == "classification":
        hp = {**_CLASSIFIER_DEFAULTS, **(params or {})}
        return XGBClassifier(**hp)
    elif task == "regression":
        hp = {**_REGRESSOR_DEFAULTS, **(params or {})}
        return XGBRegressor(**hp)
    else:
        raise ValueError(f"Unknown task '{task}'. Expected 'classification' or 'regression'.")
