"""
ensemble_models.py
==================
Build and evaluate all ensemble (and baseline) classifiers used in the
HAR comparison study.

All models are returned as a flat dict so the experiment runner can iterate
over them uniformly.  Hyperparameters are loaded from a YAML config file so
that experiments remain reproducible and easily configurable without code
changes.

Typical usage::

    models = build_all_models("config/experiment_config.yaml")
    for name, model in models.items():
        result = fit_and_evaluate(model, X_tr, y_tr, X_val, y_val)
        print(name, result)
"""

import time
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
    StackingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

# ---------------------------------------------------------------------------
# Defaults – used when config file is missing or a key is absent
# ---------------------------------------------------------------------------
_DEFAULTS: Dict[str, Any] = {
    "logistic_regression": {
        "max_iter": 1000,
        "C": 1.0,
        "solver": "lbfgs",
    },
    "decision_tree": {
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "criterion": "gini",
    },
    "random_forest": {
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_split": 2,
        "max_features": "sqrt",
    },
    "hist_gradient_boosting": {
        "max_iter": 200,
        "max_depth": None,
        "learning_rate": 0.1,
        "min_samples_leaf": 20,
    },
    "stacking": {
        "cv": 5,
    },
}

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Config loading helper
# ---------------------------------------------------------------------------

def _load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config from *config_path*; fall back to defaults on any error.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.

    Returns
    -------
    config : dict
        The ``models`` subsection of the YAML or an empty dict.
    """
    path = Path(config_path)
    if not path.exists():
        warnings.warn(
            f"Config file '{config_path}' not found. Using default hyperparameters.",
            UserWarning,
            stacklevel=3,
        )
        return {}
    try:
        with open(path, "r") as fh:
            full_cfg = yaml.safe_load(fh) or {}
        return full_cfg.get("ensemble") or full_cfg.get("models", {})
    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"Could not parse config '{config_path}': {exc}. Using defaults.",
            UserWarning,
            stacklevel=3,
        )
        return {}


def _get(cfg: dict, section: str, key: str) -> Any:
    """Return cfg[section][key] or the built-in default."""
    return cfg.get(section, {}).get(key, _DEFAULTS[section][key])


# ---------------------------------------------------------------------------
# Main factory
# ---------------------------------------------------------------------------

def build_all_models(config_path: str = "config/experiment_config.yaml") -> Dict[str, Any]:
    """Build all classifier models used in the comparison study.

    Hyperparameters are read from *config_path* (YAML).  Any missing key falls
    back to the hard-coded ``_DEFAULTS`` dictionary.

    Parameters
    ----------
    config_path : str
        Path to the YAML config file (relative to project root or absolute).

    Returns
    -------
    models : dict
        Mapping ``{name: sklearn_estimator}`` for all models:

        - ``"logistic_regression"``     – multinomial logistic regression
        - ``"decision_tree"``           – CART decision tree
        - ``"random_forest"``           – random forest
        - ``"hist_gradient_boosting"``  – histogram gradient boosting
        - ``"hard_voting"``             – hard VotingClassifier (LR + RF + HGB)
        - ``"soft_voting"``             – soft VotingClassifier (LR + RF + HGB)
        - ``"stacking"``                – StackingClassifier (LR+DT+RF base, LR meta)
    """
    cfg = _load_config(config_path)

    # ------------------------------------------------------------------
    # 1. Base models
    # ------------------------------------------------------------------
    lr = LogisticRegression(
        max_iter=_get(cfg, "logistic_regression", "max_iter"),
        C=_get(cfg, "logistic_regression", "C"),
        solver=_get(cfg, "logistic_regression", "solver"),
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    dt = DecisionTreeClassifier(
        max_depth=_get(cfg, "decision_tree", "max_depth"),
        min_samples_split=_get(cfg, "decision_tree", "min_samples_split"),
        min_samples_leaf=_get(cfg, "decision_tree", "min_samples_leaf"),
        criterion=_get(cfg, "decision_tree", "criterion"),
        random_state=RANDOM_STATE,
    )

    rf = RandomForestClassifier(
        n_estimators=_get(cfg, "random_forest", "n_estimators"),
        max_depth=_get(cfg, "random_forest", "max_depth"),
        min_samples_split=_get(cfg, "random_forest", "min_samples_split"),
        max_features=_get(cfg, "random_forest", "max_features"),
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    hgb = HistGradientBoostingClassifier(
        max_iter=_get(cfg, "hist_gradient_boosting", "max_iter"),
        max_depth=_get(cfg, "hist_gradient_boosting", "max_depth"),
        learning_rate=_get(cfg, "hist_gradient_boosting", "learning_rate"),
        min_samples_leaf=_get(cfg, "hist_gradient_boosting", "min_samples_leaf"),
        random_state=RANDOM_STATE,
    )

    # ------------------------------------------------------------------
    # 2. Voting ensembles
    # ------------------------------------------------------------------
    hard_voting = VotingClassifier(
        estimators=[
            ("lr", _clone_lr(cfg)),
            ("rf", _clone_rf(cfg)),
            ("hgb", _clone_hgb(cfg)),
        ],
        voting="hard",
        n_jobs=-1,
    )

    soft_voting = VotingClassifier(
        estimators=[
            ("lr", _clone_lr(cfg)),
            ("rf", _clone_rf(cfg)),
            ("hgb", _clone_hgb(cfg)),
        ],
        voting="soft",
        n_jobs=-1,
    )

    # ------------------------------------------------------------------
    # 3. Stacking classifier
    # ------------------------------------------------------------------
    stacking_cv = _get(cfg, "stacking", "cv")

    stacking = StackingClassifier(
        estimators=[
            ("lr", _clone_lr(cfg)),
            ("dt", _clone_dt(cfg)),
            ("rf", _clone_rf(cfg)),
        ],
        final_estimator=LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        cv=stacking_cv,
        n_jobs=-1,
        passthrough=False,
    )

    return {
        "logistic_regression": lr,
        "decision_tree": dt,
        "random_forest": rf,
        "hist_gradient_boosting": hgb,
        "hard_voting": hard_voting,
        "soft_voting": soft_voting,
        "stacking": stacking,
    }


# ---------------------------------------------------------------------------
# Fit + evaluate helper
# ---------------------------------------------------------------------------

def fit_and_evaluate(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> dict:
    """Fit *model*, then evaluate on the validation set.

    Parameters
    ----------
    model : sklearn estimator
        An unfitted (or already fitted) sklearn-compatible estimator.
    X_train : np.ndarray, shape (N_train, n_features)
    y_train : np.ndarray, shape (N_train,)
    X_val : np.ndarray, shape (N_val, n_features)
    y_val : np.ndarray, shape (N_val,)

    Returns
    -------
    result : dict with keys
        - ``"fit_time_s"``       – wall-clock seconds to fit
        - ``"predict_time_s"``   – wall-clock seconds to predict on val
        - ``"accuracy"``         – val set accuracy (0–1)
        - ``"macro_f1"``         – val set macro-averaged F1 (0–1)
        - ``"latency_ms_per_sample"`` – predict_time_s / N_val * 1000
    """
    # --- Training -----------------------------------------------------------
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    # --- Prediction ---------------------------------------------------------
    t1 = time.perf_counter()
    y_pred = model.predict(X_val)
    predict_time = time.perf_counter() - t1

    # --- Metrics ------------------------------------------------------------
    acc = accuracy_score(y_val, y_pred)
    mf1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
    n_val = len(y_val)
    latency_ms = (predict_time / n_val * 1000) if n_val > 0 else 0.0

    return {
        "fit_time_s": round(fit_time, 4),
        "predict_time_s": round(predict_time, 6),
        "accuracy": round(float(acc), 6),
        "macro_f1": round(float(mf1), 6),
        "latency_ms_per_sample": round(latency_ms, 6),
    }


# ---------------------------------------------------------------------------
# Private clone helpers (avoid sharing fitted state between voters)
# ---------------------------------------------------------------------------

def _clone_lr(cfg: dict) -> LogisticRegression:
    return LogisticRegression(
        max_iter=_get(cfg, "logistic_regression", "max_iter"),
        C=_get(cfg, "logistic_regression", "C"),
        solver=_get(cfg, "logistic_regression", "solver"),
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def _clone_dt(cfg: dict) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(
        max_depth=_get(cfg, "decision_tree", "max_depth"),
        min_samples_split=_get(cfg, "decision_tree", "min_samples_split"),
        min_samples_leaf=_get(cfg, "decision_tree", "min_samples_leaf"),
        criterion=_get(cfg, "decision_tree", "criterion"),
        random_state=RANDOM_STATE,
    )


def _clone_rf(cfg: dict) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=_get(cfg, "random_forest", "n_estimators"),
        max_depth=_get(cfg, "random_forest", "max_depth"),
        min_samples_split=_get(cfg, "random_forest", "min_samples_split"),
        max_features=_get(cfg, "random_forest", "max_features"),
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def _clone_hgb(cfg: dict) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=_get(cfg, "hist_gradient_boosting", "max_iter"),
        max_depth=_get(cfg, "hist_gradient_boosting", "max_depth"),
        learning_rate=_get(cfg, "hist_gradient_boosting", "learning_rate"),
        min_samples_leaf=_get(cfg, "hist_gradient_boosting", "min_samples_leaf"),
        random_state=RANDOM_STATE,
    )
