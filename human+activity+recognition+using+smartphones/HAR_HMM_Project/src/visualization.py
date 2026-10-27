"""
visualization.py
================
All visualisation functions for the HAR HMM + Ensemble comparison study.

Each function:
 - Accepts an optional ``save_path`` parameter.  When provided, the figure is
   saved to that path at 150 dpi with tight bounding box.
 - Sets clear titles, axis labels, and a descriptive subtitle.
 - Uses the seaborn style.
 - Closes the figure after saving (to free memory in long experiment runs).
 - Returns the ``matplotlib.figure.Figure`` object for downstream use or
   interactive inspection.

Dependencies: matplotlib, seaborn, numpy, pandas.
"""

from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Global style settings
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
_PALETTE = "tab10"
_DPI = 150
_BBOX = "tight"


def _save_and_close(fig: plt.Figure, save_path: Optional[str]) -> plt.Figure:
    """Save *fig* to *save_path* (creating parent dirs) and close it."""
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=_DPI, bbox_inches=_BBOX)
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# 1. Class distribution bar chart
# ---------------------------------------------------------------------------

def plot_class_distribution(
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    label_names: List[str],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Bar chart of class sample counts across train / validation / test splits.

    Parameters
    ----------
    y_train, y_val, y_test : array-like of int
        Label arrays for each split.
    label_names : list of str
        Human-readable names for each class (0-indexed).
    save_path : str or None
        If provided, save the figure here.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    classes = sorted(set(
        list(np.unique(y_train)) +
        list(np.unique(y_val)) +
        list(np.unique(y_test))
    ))

    def _counts(y):
        y = np.asarray(y)
        return [int(np.sum(y == c)) for c in classes]

    split_data = {
        "Train": _counts(y_train),
        "Validation": _counts(y_val),
        "Test": _counts(y_test),
    }

    x = np.arange(len(classes))
    n_splits = len(split_data)
    width = 0.25
    colors = sns.color_palette(_PALETTE, n_splits)

    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, (split_name, counts) in enumerate(split_data.items()):
        offset = (idx - n_splits / 2 + 0.5) * width
        ax.bar(x + offset, counts, width=width, label=split_name,
               color=colors[idx], edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    clabels = [
        label_names[c] if c < len(label_names) else str(c) for c in classes
    ]
    ax.set_xticklabels(clabels, rotation=30, ha="right", fontsize=9)
    ax.set_xlabel("Activity class")
    ax.set_ylabel("Number of windows")
    ax.legend(title="Split", framealpha=0.8)
    fig.suptitle(
        "Class Distribution Across Data Splits",
        fontsize=13, fontweight="bold", y=1.01,
    )
    ax.set_title(
        "Counts of labelled windows per activity in each partition",
        fontsize=9, color="gray",
    )
    fig.tight_layout()
    return _save_and_close(fig, save_path)


# ---------------------------------------------------------------------------
# 2. Confusion matrix heatmap
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str],
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Annotated confusion-matrix heatmap (counts + row-normalised %).

    Parameters
    ----------
    y_true : array-like of int
    y_pred : array-like of int
    label_names : list of str
    title : str
    save_path : str or None

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    from sklearn.metrics import confusion_matrix as sk_cm

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    cm = sk_cm(y_true, y_pred, labels=classes)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)

    clabels = [label_names[c] if c < len(label_names) else str(c) for c in classes]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, data, fmt, subtitle in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2f"],
        ["Raw counts", "Row-normalised (recall per class)"],
    ):
        sns.heatmap(
            data,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=clabels,
            yticklabels=clabels,
            linewidths=0.4,
            linecolor="lightgray",
            ax=ax,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_xlabel("Predicted label", fontsize=9)
        ax.set_ylabel("True label", fontsize=9)
        ax.set_title(subtitle, fontsize=9, color="gray")
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.tick_params(axis="y", rotation=0, labelsize=8)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save_and_close(fig, save_path)


# ---------------------------------------------------------------------------
# 3. Model comparison bar chart
# ---------------------------------------------------------------------------

def plot_model_comparison(
    metrics_df: pd.DataFrame,
    metric: str = "macro_f1",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Horizontal bar chart comparing models on a chosen scalar metric.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Must contain a ``"model"`` column and a column named *metric*.
        Rows are sorted descending by *metric*.
    metric : str
        Column name to plot (e.g. ``"macro_f1"``, ``"accuracy"``).
    save_path : str or None

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    df = metrics_df.copy()
    if "model" not in df.columns or metric not in df.columns:
        raise ValueError(
            f"metrics_df must contain 'model' and '{metric}' columns."
        )

    df = df.sort_values(metric, ascending=True)
    colors = sns.color_palette(_PALETTE, len(df))

    fig, ax = plt.subplots(figsize=(9, max(4, len(df) * 0.6)))
    bars = ax.barh(df["model"], df[metric], color=colors, edgecolor="white")

    # Annotate bar ends
    for bar, val in zip(bars, df[metric]):
        ax.text(
            bar.get_width() + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center", ha="left", fontsize=8,
        )

    ax.set_xlim(0, min(1.05, float(df[metric].max()) + 0.08))
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_ylabel("")
    fig.suptitle(
        f"Model Comparison — {metric.replace('_', ' ').title()}",
        fontsize=13, fontweight="bold",
    )
    ax.set_title(
        "Higher is better; error bars not shown",
        fontsize=9, color="gray",
    )
    fig.tight_layout()
    return _save_and_close(fig, save_path)


# ---------------------------------------------------------------------------
# 4. Per-class F1 bar chart
# ---------------------------------------------------------------------------

def plot_per_class_f1(
    metrics_dict: dict,
    label_names: List[str],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Grouped bar chart of per-class F1 across multiple models.

    Parameters
    ----------
    metrics_dict : dict
        ``{model_name: per_class_f1_dict}`` where each ``per_class_f1_dict``
        maps class label (str or int) -> F1 score (float).
    label_names : list of str
        Ordered list of class names.
    save_path : str or None

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    model_names = list(metrics_dict.keys())
    n_models = len(model_names)
    n_classes = len(label_names)

    # Build array (n_models, n_classes)
    f1_matrix = np.zeros((n_models, n_classes))
    for m_idx, name in enumerate(model_names):
        pcf = metrics_dict[name]
        for c_idx, cls_name in enumerate(label_names):
            val = pcf.get(cls_name, pcf.get(str(c_idx), pcf.get(c_idx, 0.0)))
            f1_matrix[m_idx, c_idx] = float(val)

    x = np.arange(n_classes)
    width = 0.8 / max(n_models, 1)
    colors = sns.color_palette(_PALETTE, n_models)

    fig, ax = plt.subplots(figsize=(12, 5))
    for m_idx, (name, color) in enumerate(zip(model_names, colors)):
        offset = (m_idx - n_models / 2 + 0.5) * width
        ax.bar(x + offset, f1_matrix[m_idx], width=width, label=name,
               color=color, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(label_names, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Activity class")
    ax.set_ylabel("F1 score")
    ax.legend(title="Model", bbox_to_anchor=(1.01, 1), loc="upper left",
              fontsize=8, framealpha=0.8)
    fig.suptitle("Per-class F1 Score by Model", fontsize=13, fontweight="bold")
    ax.set_title("Class-level discriminative performance across models",
                 fontsize=9, color="gray")
    fig.tight_layout()
    return _save_and_close(fig, save_path)


# ---------------------------------------------------------------------------
# 5. Activity timeline (ground truth vs prediction strip chart)
# ---------------------------------------------------------------------------

def plot_activity_timeline(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str],
    subject_id: Union[int, str] = "?",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Side-by-side colour-coded activity strips for a single subject.

    Parameters
    ----------
    y_true : array-like of int, shape (T,)
    y_pred : array-like of int, shape (T,)
    label_names : list of str
    subject_id : int or str
    save_path : str or None

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    T = len(y_true)
    n_classes = max(len(label_names), int(y_true.max()) + 1)

    cmap = plt.get_cmap("tab10", n_classes)
    norm = plt.Normalize(vmin=0, vmax=n_classes - 1)

    fig, axes = plt.subplots(2, 1, figsize=(14, 3.5), sharex=True)
    for ax, y, label in zip(axes, [y_true, y_pred], ["Ground truth", "Predicted"]):
        img = y[np.newaxis, :]
        ax.imshow(img, aspect="auto", cmap=cmap, norm=norm,
                  interpolation="nearest", extent=[0, T, 0, 1])
        ax.set_yticks([])
        ax.set_ylabel(label, fontsize=9, rotation=0, ha="right", va="center",
                      labelpad=5)
        ax.yaxis.set_label_coords(-0.01, 0.5)

    axes[-1].set_xlabel("Time step (window index)")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(
            facecolor=cmap(norm(i)),
            label=label_names[i] if i < len(label_names) else str(i),
        )
        for i in range(n_classes)
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=n_classes,
        fontsize=7,
        framealpha=0.8,
        bbox_to_anchor=(0.5, -0.15),
    )
    fig.suptitle(
        f"Activity Timeline — Subject {subject_id}",
        fontsize=13, fontweight="bold",
    )
    axes[0].set_title(
        "Colour encodes activity class; mismatches between rows are errors",
        fontsize=9, color="gray",
    )
    fig.tight_layout()
    return _save_and_close(fig, save_path)


# ---------------------------------------------------------------------------
# 6. HMM transition matrix heatmap
# ---------------------------------------------------------------------------

def plot_hmm_transition_heatmap(
    A: np.ndarray,
    label_names: List[str],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Annotated heatmap of the HMM transition matrix A[i, j] = P(j | i).

    Parameters
    ----------
    A : np.ndarray, shape (n_states, n_states)
        Row-stochastic transition matrix.
    label_names : list of str
    save_path : str or None

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    A = np.asarray(A)
    n = A.shape[0]
    clabels = [label_names[i] if i < len(label_names) else str(i) for i in range(n)]

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        A,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        xticklabels=clabels,
        yticklabels=clabels,
        linewidths=0.4,
        linecolor="lightgray",
        ax=ax,
        vmin=0, vmax=1,
        cbar_kws={"label": "P(next | current)", "shrink": 0.8},
    )
    ax.set_xlabel("Next activity (j)", fontsize=9)
    ax.set_ylabel("Current activity (i)", fontsize=9)
    ax.tick_params(axis="x", rotation=40, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    fig.suptitle("HMM Transition Matrix", fontsize=13, fontweight="bold")
    ax.set_title(
        "A[i, j] = P(activity j at t+1 | activity i at t)",
        fontsize=9, color="gray",
    )
    fig.tight_layout()
    return _save_and_close(fig, save_path)


# ---------------------------------------------------------------------------
# 7. PCA explained variance scree plot
# ---------------------------------------------------------------------------

def plot_pca_explained_variance(
    explained_var_ratio: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Scree plot of PCA explained variance ratio with cumulative overlay.

    Parameters
    ----------
    explained_var_ratio : array-like of float
        Explained variance ratio for each principal component.
    save_path : str or None

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    evr = np.asarray(explained_var_ratio)
    cumulative = np.cumsum(evr)
    n_comps = len(evr)
    x = np.arange(1, n_comps + 1)

    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax2 = ax1.twinx()

    ax1.bar(x, evr * 100, color=sns.color_palette("Blues_d", n_comps),
            alpha=0.85, label="Individual %")
    ax2.plot(x, cumulative * 100, color="coral", marker="o", ms=4,
             linewidth=2, label="Cumulative %")
    ax2.axhline(95, color="gray", linestyle="--", linewidth=1,
                label="95% threshold")

    ax1.set_xlabel("Principal component")
    ax1.set_ylabel("Individual explained variance (%)")
    ax2.set_ylabel("Cumulative explained variance (%)")
    ax2.set_ylim(0, 105)
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, framealpha=0.8, loc="center right")

    fig.suptitle("PCA Explained Variance", fontsize=13, fontweight="bold")
    ax1.set_title(
        "Select components where cumulative variance exceeds 95%",
        fontsize=9, color="gray",
    )
    fig.tight_layout()
    return _save_and_close(fig, save_path)


# ---------------------------------------------------------------------------
# 8. HMM state posterior (gamma) heatmap
# ---------------------------------------------------------------------------

def plot_hmm_state_sequence(
    gamma: np.ndarray,
    label_names: List[str],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Heatmap of HMM forward-backward posteriors gamma_t(s) over time.

    Parameters
    ----------
    gamma : np.ndarray, shape (T, n_states)
        State posterior probabilities.
    label_names : list of str
    save_path : str or None

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    gamma = np.asarray(gamma)
    T, S = gamma.shape
    clabels = [label_names[s] if s < len(label_names) else str(s) for s in range(S)]

    fig, ax = plt.subplots(figsize=(min(14, T // 5 + 4), 4))
    im = ax.imshow(
        gamma.T,
        aspect="auto",
        cmap="viridis",
        vmin=0, vmax=1,
        interpolation="nearest",
        origin="upper",
    )
    plt.colorbar(im, ax=ax, label="gamma_t(s)  [posterior probability]", shrink=0.85)
    ax.set_yticks(range(S))
    ax.set_yticklabels(clabels, fontsize=8)
    ax.set_xlabel("Time step (window index)")
    ax.set_ylabel("Activity state")
    fig.suptitle(
        "HMM State Posterior Probabilities",
        fontsize=13, fontweight="bold",
    )
    ax.set_title(
        "gamma_t(s) = P(state=s at t | full observation sequence)",
        fontsize=9, color="gray",
    )
    fig.tight_layout()
    return _save_and_close(fig, save_path)


# ---------------------------------------------------------------------------
# 9. Inference latency comparison
# ---------------------------------------------------------------------------

def plot_latency_comparison(
    model_names: List[str],
    latencies_ms: List[float],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Horizontal bar chart of per-sample inference latency (ms).

    Parameters
    ----------
    model_names : list of str
        Model names (y-axis labels).
    latencies_ms : list of float
        Per-sample latency in milliseconds (aligned with *model_names*).
    save_path : str or None

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    pairs = sorted(zip(latencies_ms, model_names), reverse=False)
    lat_sorted  = [p[0] for p in pairs]
    name_sorted = [p[1] for p in pairs]
    max_lat = max(lat_sorted) if lat_sorted else 1.0

    colors = sns.color_palette(_PALETTE, len(model_names))

    fig, ax = plt.subplots(figsize=(9, max(3, len(model_names) * 0.55)))
    bars = ax.barh(name_sorted, lat_sorted, color=colors, edgecolor="white")

    for bar, val in zip(bars, lat_sorted):
        ax.text(
            bar.get_width() + max_lat * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f} ms",
            va="center", ha="left", fontsize=8,
        )

    ax.set_xlabel("Latency per sample (ms)  [lower is better]")
    ax.set_ylabel("")
    ax.set_xlim(0, max_lat * 1.25)
    fig.suptitle(
        "Inference Latency Comparison",
        fontsize=13, fontweight="bold",
    )
    ax.set_title(
        "Average wall-clock time to predict a single window",
        fontsize=9, color="gray",
    )
    fig.tight_layout()
    return _save_and_close(fig, save_path)
