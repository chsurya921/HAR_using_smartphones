"""
evaluation.py
=============
Comprehensive evaluation utilities for the HAR HMM + Ensemble comparison study.

Functions
---------
evaluate_flat           – compute accuracy, precision, recall, F1, log-loss,
                          per-class F1, and confusion matrix for flat predictions.
evaluate_sequential     – sequential-specific metrics (false switch rate,
                          transition consistency).
save_metrics            – append a metrics row to results/metrics.csv.
print_metrics_table     – pretty-print a metric summary to stdout.
"""

import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)


# ---------------------------------------------------------------------------
# Flat (frame-level) evaluation
# ---------------------------------------------------------------------------

def evaluate_flat(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    label_names: Optional[List[str]] = None,
) -> Dict:
    """Compute all metrics for flat (non-sequential) predictions.

    Produces a comprehensive metrics dictionary suitable for tabular reporting
    and CSV export.

    Parameters
    ----------
    y_true : array-like of int, shape (N,)
        Ground-truth labels (0-indexed or 1-indexed — consistency required).
    y_pred : array-like of int, shape (N,)
        Predicted labels.
    y_proba : array-like of float, shape (N, n_classes) or None
        Predicted class probabilities.  If provided, log-loss is included.
    label_names : list of str or None
        Human-readable class names used as keys in ``per_class_f1``.
        If None, integer class ids are used.

    Returns
    -------
    metrics : dict
        Keys:

        - ``"accuracy"``          – float in [0, 1]
        - ``"macro_precision"``   – float in [0, 1]
        - ``"macro_recall"``      – float in [0, 1]
        - ``"macro_f1"``          – float in [0, 1]
        - ``"weighted_f1"``       – float in [0, 1]
        - ``"log_loss"``          – float ≥ 0 (only if y_proba is given)
        - ``"per_class_f1"``      – dict {label_name: float}
        - ``"confusion_matrix"``  – np.ndarray of shape (n_classes, n_classes)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc      = float(accuracy_score(y_true, y_pred))
    mac_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    mac_rec  = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    mac_f1   = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    wt_f1    = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    classes = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    per_class_f1_vals = f1_score(
        y_true, y_pred, labels=classes, average=None, zero_division=0
    )

    if label_names is not None and len(label_names) == len(classes):
        keys = label_names
    else:
        keys = [str(c) for c in classes]

    per_class_f1 = {k: float(v) for k, v in zip(keys, per_class_f1_vals)}

    cm = confusion_matrix(y_true, y_pred, labels=classes)

    metrics: Dict = {
        "accuracy": round(acc, 6),
        "macro_precision": round(mac_prec, 6),
        "macro_recall": round(mac_rec, 6),
        "macro_f1": round(mac_f1, 6),
        "weighted_f1": round(wt_f1, 6),
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm,
    }

    if y_proba is not None:
        y_proba = np.asarray(y_proba, dtype=float)
        try:
            ll = float(log_loss(y_true, y_proba, labels=classes))
            metrics["log_loss"] = round(ll, 6)
        except Exception:  # noqa: BLE001
            metrics["log_loss"] = None

    return metrics


# ---------------------------------------------------------------------------
# Sequential evaluation
# ---------------------------------------------------------------------------

def evaluate_sequential(
    y_true_sequences: List[np.ndarray],
    y_pred_sequences: List[np.ndarray],
) -> Dict:
    """Compute sequential-specific evaluation metrics.

    Metrics capture how well the model preserves the *temporal structure* of
    the activity stream, beyond raw frame accuracy.

    Parameters
    ----------
    y_true_sequences : list of np.ndarray, each shape (T_i,)
        Ground-truth label sequences (one per subject / segment).
    y_pred_sequences : list of np.ndarray, each shape (T_i,)
        Predicted label sequences (must align with y_true_sequences).

    Returns
    -------
    metrics : dict
        Keys:

        - ``"false_activity_switch_rate"``
              Fraction of consecutive (t, t+1) pairs where the prediction
              switches activity but the ground truth does not.
              Low is good.

        - ``"transition_consistency"``
              Fraction of true activity transitions (ground truth switches)
              that are also captured by the predicted sequence at the same
              time step.  High is good.

        - ``"n_consecutive_pairs_total"`` – total (t, t+1) pairs examined
        - ``"n_false_switches"``           – raw count of false switches
        - ``"n_true_transitions"``         – raw count of ground-truth transitions
        - ``"n_captured_transitions"``     – true transitions captured by pred
    """
    n_pairs_total = 0
    n_false_switches = 0
    n_true_transitions = 0
    n_captured_transitions = 0

    for y_true, y_pred in zip(y_true_sequences, y_pred_sequences):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        T = min(len(y_true), len(y_pred))
        if T < 2:
            continue

        # Boolean arrays of changes at each consecutive pair
        gt_switch   = y_true[1:T] != y_true[:T-1]   # ground-truth transition
        pred_switch = y_pred[1:T] != y_pred[:T-1]    # predicted transition

        pairs = T - 1
        n_pairs_total += pairs

        # False switch: pred switches but gt does not
        n_false_switches += int(np.sum(pred_switch & ~gt_switch))

        # True transition: gt switches
        n_true_trans = int(gt_switch.sum())
        n_true_transitions += n_true_trans

        # Captured: both switch at the same step
        n_captured_transitions += int(np.sum(gt_switch & pred_switch))

    false_switch_rate = (
        n_false_switches / n_pairs_total if n_pairs_total > 0 else 0.0
    )
    transition_consistency = (
        n_captured_transitions / n_true_transitions
        if n_true_transitions > 0
        else 0.0
    )

    return {
        "false_activity_switch_rate": round(false_switch_rate, 6),
        "transition_consistency": round(transition_consistency, 6),
        "n_consecutive_pairs_total": n_pairs_total,
        "n_false_switches": n_false_switches,
        "n_true_transitions": n_true_transitions,
        "n_captured_transitions": n_captured_transitions,
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_metrics(metrics_dict: dict, path: str = "results/metrics.csv") -> None:
    """Append a metrics row to a CSV file safely handling dynamic columns."""
    import pandas as pd
    flat: Dict[str, object] = {}
    for k, v in metrics_dict.items():
        if isinstance(v, np.ndarray):
            continue
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                flat[f"{k}__{sub_k}"] = sub_v
        else:
            flat[k] = v

    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    new_df = pd.DataFrame([flat])
    if save_path.exists() and save_path.stat().st_size > 0:
        existing_df = pd.read_csv(save_path)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df
    combined_df.to_csv(save_path, index=False)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_metrics_table(metrics_dict: dict, model_name: str) -> None:
    """Pretty-print a metric summary table to stdout.

    Parameters
    ----------
    metrics_dict : dict
        As returned by ``evaluate_flat`` or ``evaluate_sequential``.
    model_name : str
        Human-readable name shown in the table header.
    """
    width = 60
    sep = "─" * width

    print(f"\n{sep}")
    print(f"  Model: {model_name}")
    print(sep)

    # Scalar metrics first
    scalar_keys = [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "log_loss",
        "false_activity_switch_rate",
        "transition_consistency",
        "fit_time_s",
        "predict_time_s",
        "latency_ms_per_sample",
    ]
    for key in scalar_keys:
        if key in metrics_dict and metrics_dict[key] is not None:
            val = metrics_dict[key]
            if isinstance(val, float):
                print(f"  {key:<35s}  {val:.4f}")
            else:
                print(f"  {key:<35s}  {val}")

    # Per-class F1
    if "per_class_f1" in metrics_dict and metrics_dict["per_class_f1"]:
        print(f"\n  {'Per-class F1':}")
        for cls, f1v in metrics_dict["per_class_f1"].items():
            print(f"    {str(cls):<30s}  {float(f1v):.4f}")

    # Confusion matrix (compact)
    if "confusion_matrix" in metrics_dict:
        cm = metrics_dict["confusion_matrix"]
        print(f"\n  Confusion Matrix (shape {cm.shape}):")
        for row in cm:
            print("    " + "  ".join(f"{v:5d}" for v in row))

    print(sep + "\n")
