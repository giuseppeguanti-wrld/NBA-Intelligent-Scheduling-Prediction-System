"""
Training orchestrator: fits all three models and serialises them to disk.

Saved artefacts
---------------
    models/{model_name}_{task}_v1.joblib

No hyper-parameter search is performed here.  If tuning is required,
run the dedicated optimisation notebook and pass the resulting ``params``
dicts via the ``custom_params`` argument.

Usage
-----
    from src.trainer import train_all
    results = train_all("classification")
"""
import logging
import time
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

import joblib
import pandas as pd

from src.data_loader import load_splits
from src.preprocessing import fit_preprocessor, transform
from src.models import gradient_boosting, random_forest, xgboost_model

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

Task = Literal["classification", "regression"]

# Registry: (name, module)
_MODEL_REGISTRY = [
    ("random_forest", random_forest),
    ("xgboost", xgboost_model),
    ("gradient_boosting", gradient_boosting),
]


def train_all(
    task: Task,
    custom_params: Optional[Dict[str, Dict[str, Any]]] = None,
    scale: bool = False,
) -> pd.DataFrame:
    """Fit all three models on the training split and save to ``models/``.

    Parameters
    ----------
    task:
        ``"classification"`` or ``"regression"``.
    custom_params:
        Optional mapping ``{model_name: {param: value}}`` that overrides
        default hyper-parameters for specific models.  Model names are
        ``"random_forest"``, ``"xgboost"``, and ``"gradient_boosting"``.
    scale:
        Whether to apply :class:`~sklearn.preprocessing.StandardScaler`
        (forwarded to :func:`~src.preprocessing.fit_preprocessor`).

    Returns
    -------
    :class:`~pandas.DataFrame` summarising model names and elapsed training
    times (seconds).
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    custom_params = custom_params or {}

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    logger.info("Loading splits for task='%s'", task)
    (X_train_raw, y_train), (X_val_raw, y_val), _ = load_splits(task=task)

    # ------------------------------------------------------------------
    # 2. Preprocess
    # ------------------------------------------------------------------
    preprocessor = fit_preprocessor(X_train_raw, scale=scale)
    X_train = transform(X_train_raw, preprocessor)
    X_val = transform(X_val_raw, preprocessor)

    # Save preprocessor alongside models so evaluator can reuse it
    prep_path = MODELS_DIR / f"preprocessor_{task}_v1.joblib"
    joblib.dump(preprocessor, prep_path)
    logger.info("Preprocessor saved → %s", prep_path)

    # ------------------------------------------------------------------
    # 3. Train each model
    # ------------------------------------------------------------------
    records = []
    for model_name, module in _MODEL_REGISTRY:
        params = custom_params.get(model_name)
        model = module.build_model(task, params=params)
        logger.info("Training %-20s  task=%s ...", model_name, task)

        t0 = time.perf_counter()
        _fit(model, model_name, X_train, y_train, X_val, y_val)
        elapsed = time.perf_counter() - t0

        save_path = MODELS_DIR / f"{model_name}_{task}_v1.joblib"
        joblib.dump(model, save_path)

        logger.info(
            "%-20s  trained in %.1fs  → %s",
            model_name,
            elapsed,
            save_path,
        )
        records.append({"model": model_name, "task": task, "train_time_s": round(elapsed, 2)})

    summary = pd.DataFrame(records)
    logger.info("\n%s", summary.to_string(index=False))
    return summary


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fit(
    model: Any,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> None:
    """Dispatch to the correct fit call depending on model type."""
    if model_name == "xgboost":
        # XGBoost uses eval_set for early-stopping monitoring
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
    else:
        model.fit(X_train, y_train)
