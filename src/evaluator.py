"""
Evaluation utilities: metrics, comparison tables, and diagnostic plots.

Usage
-----
    # Full comparative evaluation on the test set
    from src.evaluator import evaluate_all
    metrics_df = evaluate_all("classification")

    # Individual plots
    from src.evaluator import plot_feature_importance, plot_roc_curves, plot_residuals
    plot_feature_importance(model, feature_names)
    plot_roc_curves("classification")
    plot_residuals("regression")
"""
import logging
from pathlib import Path
from typing import Any, Literal

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.data_loader import load_splits
from src.preprocessing import transform
from src.loader import load_config

config = load_config()

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / config["models"]["models_dir"]

Task = Literal["classification", "regression"]

_MODEL_NAMES = ["random_forest", "xgboost", "gradient_boosting"]


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def evaluate_all(task: Task) -> pd.DataFrame:
    """Load every saved model, evaluate on the test set, and return a summary.

    Parameters
    ----------
    task:
        ``"classification"`` or ``"regression"``.

    Returns
    -------
    :class:`~pandas.DataFrame` with one row per model and one column per
    metric.  Classification metrics: accuracy, precision, recall, F1, AUC-ROC.
    Regression metrics: MAE, RMSE, R².
    """
    _, _, (X_test_raw, y_test) = load_splits(task=task)

    prep_path = MODELS_DIR / f"preprocessor_{task}_v1.joblib"
    preprocessor = joblib.load(prep_path)
    X_test = transform(X_test_raw, preprocessor)

    records = []
    for name in _MODEL_NAMES:
        model_path = MODELS_DIR / f"{name}_{task}_v1.joblib"
        if not model_path.exists():
            logger.warning("Model not found, skipping: %s", model_path)
            continue

        model = joblib.load(model_path)
        logger.info("Evaluating %-20s on test set (%d samples)", name, len(y_test))

        if task == "classification":
            row = _classification_metrics(model, X_test, y_test, name)
        else:
            row = _regression_metrics(model, X_test, y_test, name)

        records.append(row)

    df = pd.DataFrame(records).set_index("model")
    logger.info("\n%s", df.to_string())
    return df


def plot_feature_importance(
    model: Any,
    feature_names: list[str],
    top_n: int = 20,
    title: str = "Feature Importance",
) -> plt.Figure:
    """Horizontal bar chart of the top-N most important features.

    Works with any scikit-learn / XGBoost estimator that exposes
    ``feature_importances_``.

    Parameters
    ----------
    model:
        Fitted estimator.
    feature_names:
        Ordered list of feature names matching the training columns.
    top_n:
        Number of features to display (default 20).
    title:
        Plot title.

    Returns
    -------
    :class:`~matplotlib.figure.Figure`
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    top_names = [feature_names[i] for i in indices]
    top_vals = importances[indices]

    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.35)))
    ax.barh(range(top_n), top_vals[::-1], color="steelblue")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Importance")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_roc_curves(task: Task = "classification") -> plt.Figure:
    """Plot all three ROC curves on the same axes for visual comparison.

    Parameters
    ----------
    task:
        Must be ``"classification"``; raises ``ValueError`` otherwise.

    Returns
    -------
    :class:`~matplotlib.figure.Figure`
    """
    if task != "classification":
        raise ValueError("ROC curves are only available for classification tasks.")

    _, _, (X_test_raw, y_test) = load_splits(task=task)
    prep_path = MODELS_DIR / f"preprocessor_{task}_v1.joblib"
    preprocessor = joblib.load(prep_path)
    X_test = transform(X_test_raw, preprocessor)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")

    colors = ["steelblue", "darkorange", "forestgreen"]
    for name, color in zip(_MODEL_NAMES, colors):
        model_path = MODELS_DIR / f"{name}_{task}_v1.joblib"
        if not model_path.exists():
            continue
        model = joblib.load(model_path)
        y_prob = predict_proba(model, X_test)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc:.3f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models")
    ax.legend(loc="lower right")
    plt.tight_layout()
    return fig


def plot_residuals(task: Task = "regression") -> plt.Figure:
    """Scatter plot of residuals (predicted − actual) for regression models.

    Parameters
    ----------
    task:
        Must be ``"regression"``; raises ``ValueError`` otherwise.

    Returns
    -------
    :class:`~matplotlib.figure.Figure`
    """
    if task != "regression":
        raise ValueError("Residual plots are only available for regression tasks.")

    _, _, (X_test_raw, y_test) = load_splits(task=task)
    prep_path = MODELS_DIR / f"preprocessor_{task}_v1.joblib"
    preprocessor = joblib.load(prep_path)
    X_test = transform(X_test_raw, preprocessor)

    n_models = sum(
        1 for n in _MODEL_NAMES
        if (MODELS_DIR / f"{n}_{task}_v1.joblib").exists()
    )
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), squeeze=False)

    col = 0
    for name in _MODEL_NAMES:
        model_path = MODELS_DIR / f"{name}_{task}_v1.joblib"
        if not model_path.exists():
            continue
        model = joblib.load(model_path)
        y_pred = model.predict(X_test)
        residuals = y_pred - y_test.values

        ax = axes[0][col]
        ax.scatter(y_pred, residuals, alpha=0.3, s=10, color="steelblue")
        ax.axhline(0, color="red", lw=1.5, linestyle="--")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Residual (pred − actual)")
        ax.set_title(name)
        col += 1

    fig.suptitle("Residuals — All Models", fontsize=13, y=1.02)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def predict_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Return positive-class probability for classifiers."""
    return model.predict_proba(X)[:, 1]


def _classification_metrics(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    name: str,
) -> dict:
    y_pred = model.predict(X_test)
    y_prob = predict_proba(model, X_test)

    cm = confusion_matrix(y_test, y_pred)
    logger.info("Confusion matrix for %s:\n%s", name, cm)

    return {
        "model": name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "auc_roc": round(roc_auc_score(y_test, y_prob), 4),
    }


def _regression_metrics(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    name: str,
) -> dict:
    y_pred = model.predict(X_test)
    residuals = y_pred - y_test.values

    logger.info(
        "%s  residuals — mean=%.2f  std=%.2f  p5=%.1f  p95=%.1f",
        name,
        residuals.mean(),
        residuals.std(),
        np.percentile(residuals, 5),
        np.percentile(residuals, 95),
    )

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    return {
        "model": name,
        "mae": round(mean_absolute_error(y_test, y_pred), 4),
        "rmse": round(rmse, 4),
        "r2": round(r2_score(y_test, y_pred), 4),
    }
