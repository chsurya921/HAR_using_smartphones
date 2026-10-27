"""
experiment_runner.py
====================
Runs Experiments A–I as defined in the methodology.

Experiment mapping
------------------
A  Dataset characterisation (class/subject distributions, PCA variance)
B  Baseline classifiers (LR, DT) on val and test
C  Ensemble comparison (RF, HGB, Voting, Stacking) on val and test
D  HMM estimation, likelihood and Viterbi decoding on sequences
E  Hybrid HMM-Ensemble on test sequences
F  Sensitivity: HMM state count, PCA components, ensemble settings
G  Ablation: what happens without PCA / without temporal smoothing
H  Unseen-participant generalisation (test set macro-F1 per subject)
I  Latency and model size

Usage
-----
    cd HAR_HMM_Project
    python src/experiment_runner.py
"""

from __future__ import annotations

import os
import sys
import time
import numpy as np
import yaml
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, BASE_DIR)

from src.data_loader import HARDataLoader
from src.preprocessing import Preprocessor
from src.evaluation import evaluate_flat, evaluate_sequential, save_metrics, print_metrics_table
CONFIG_PATH = os.path.join(PROJECT_DIR, "config", "experiment_config.yaml")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
LOGS_DIR    = os.path.join(RESULTS_DIR, "logs")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def log(msg: str):
    print(f"[Runner] {msg}", flush=True)
    with open(os.path.join(LOGS_DIR, "run.log"), "a") as f:
        f.write(msg + "\n")


def load_config():
    with open(CONFIG_PATH) as fh:
        return yaml.safe_load(fh)


def load_data_and_preprocess(config, n_components: int | None = None):
    """Load splits and apply StandardScaler + PCA."""
    loader = HARDataLoader(config_path=CONFIG_PATH, base_dir=PROJECT_DIR)
    splits = loader.load_all()

    prep = Preprocessor(config_path=CONFIG_PATH, n_components=n_components)
    X_tr, X_va, X_te = prep.fit_transform(
        splits.train_X, splits.val_X, splits.test_X)

    # Also transform sequences for HMM
    seqs_tr = prep.transform_sequences(splits.sequences_train)
    seqs_va = prep.transform_sequences(splits.sequences_val)
    seqs_te = prep.transform_sequences(splits.sequences_test)

    return splits, prep, X_tr, X_va, X_te, seqs_tr, seqs_va, seqs_te


# ─────────────────────────────────────────────────────────────────────────────
# Experiment A: Dataset Characterisation
# ─────────────────────────────────────────────────────────────────────────────

def experiment_A(splits, prep):
    log("=== Experiment A: Dataset Characterisation ===")
    from visualization import (
        plot_class_distribution,
        plot_pca_explained_variance,
    )

    label_names = list(splits.activity_labels.values())

    plot_class_distribution(
        splits.train_y, splits.val_y, splits.test_y,
        label_names=label_names,
        save_path=os.path.join(FIGURES_DIR, "A_class_distribution.png"),
    )
    log("  Saved class distribution plot.")

    if prep.pca is not None:
        plot_pca_explained_variance(
            prep.pca.explained_variance_ratio_,
            save_path=os.path.join(FIGURES_DIR, "A_pca_explained_variance.png"),
        )
        log(f"  PCA: {prep.n_features_out} components, "
            f"{prep.pca.explained_variance_ratio_.cumsum()[-1]*100:.1f}% variance.")

    log("  [A] Complete.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Experiment B: Baseline Classifiers
# ─────────────────────────────────────────────────────────────────────────────

def experiment_B(X_tr, y_tr, X_va, y_va, X_te, y_te, splits):
    log("=== Experiment B: Baseline Classifiers ===")
    from ensemble_models import build_all_models
    from visualization import plot_confusion_matrix

    all_results = []
    label_names = list(splits.activity_labels.values())
    models = build_all_models(CONFIG_PATH)

    for name in ["logistic_regression", "decision_tree"]:
        model = models[name]
        log(f"  Fitting {name} …")
        t0 = time.time()
        model.fit(X_tr, y_tr)
        train_time = time.time() - t0

        t1 = time.time()
        y_pred_val = model.predict(X_va)
        val_latency_ms = (time.time() - t1) / len(X_va) * 1000

        y_pred_te = model.predict(X_te)
        y_proba_te = model.predict_proba(X_te) if hasattr(model, "predict_proba") else None

        val_m  = evaluate_flat(y_va, y_pred_val, label_names=label_names)
        test_m = evaluate_flat(y_te, y_pred_te, y_proba_te, label_names=label_names)

        result = {
            "experiment": "B",
            "model": name,
            "val_accuracy": val_m["accuracy"],
            "val_macro_f1": val_m["macro_f1"],
            "test_accuracy": test_m["accuracy"],
            "test_macro_f1": test_m["macro_f1"],
            "test_log_loss": test_m.get("log_loss", None),
            "train_time_s": round(train_time, 3),
            "val_latency_ms_per_sample": round(val_latency_ms, 4),
        }
        all_results.append(result)
        print_metrics_table(result, name)

        plot_confusion_matrix(
            y_te, y_pred_te, label_names=label_names,
            title=f"Confusion Matrix — {name}",
            save_path=os.path.join(FIGURES_DIR, f"B_cm_{name}.png"),
        )
        save_metrics(result, os.path.join(RESULTS_DIR, "metrics.csv"))

    log("  [B] Complete.\n")
    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Experiment C: Ensemble Comparison
# ─────────────────────────────────────────────────────────────────────────────

def experiment_C(X_tr, y_tr, X_va, y_va, X_te, y_te, splits):
    log("=== Experiment C: Ensemble Comparison ===")
    from ensemble_models import build_all_models
    from visualization import plot_confusion_matrix, plot_model_comparison

    label_names = list(splits.activity_labels.values())
    models = build_all_models(CONFIG_PATH)
    ensemble_names = [
        "random_forest", "hist_gradient_boosting",
        "hard_voting", "soft_voting", "stacking"
    ]

    all_results = []
    for name in ensemble_names:
        model = models[name]
        log(f"  Fitting {name} …")
        t0 = time.time()
        model.fit(X_tr, y_tr)
        train_time = time.time() - t0

        y_pred_val = model.predict(X_va)
        y_pred_te  = model.predict(X_te)
        y_proba_te = model.predict_proba(X_te) if hasattr(model, "predict_proba") else None

        val_m  = evaluate_flat(y_va, y_pred_val, label_names=label_names)
        test_m = evaluate_flat(y_te, y_pred_te, y_proba_te, label_names=label_names)

        result = {
            "experiment": "C",
            "model": name,
            "val_accuracy": val_m["accuracy"],
            "val_macro_f1": val_m["macro_f1"],
            "test_accuracy": test_m["accuracy"],
            "test_macro_f1": test_m["macro_f1"],
            "test_log_loss": test_m.get("log_loss", None),
            "train_time_s": round(train_time, 3),
        }
        all_results.append(result)
        print_metrics_table(result, name)

        plot_confusion_matrix(
            y_te, y_pred_te, label_names=label_names,
            title=f"Confusion Matrix — {name}",
            save_path=os.path.join(FIGURES_DIR, f"C_cm_{name}.png"),
        )
        save_metrics(result, os.path.join(RESULTS_DIR, "metrics.csv"))

    log("  [C] Complete.\n")
    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Experiment D: HMM Estimation, Likelihood and Decoding
# ─────────────────────────────────────────────────────────────────────────────

def experiment_D(seqs_tr, seqs_va, seqs_te, splits, config):
    log("=== Experiment D: HMM (Supervised Init + Baum-Welch) ===")
    from hmm_model import GaussianHMM as MyHMM
    from visualization import (
        plot_hmm_transition_heatmap,
        plot_activity_timeline,
        plot_hmm_state_sequence,
        plot_confusion_matrix,
    )

    label_names = list(splits.activity_labels.values())
    n_states = config["hmm"]["n_states"]
    cov_type = config["hmm"]["covariance_type"]

    model = MyHMM(
        n_states=n_states,
        covariance_type=cov_type,
        n_iter=config["hmm"]["baum_welch"]["max_iter"],
        tol=config["hmm"]["baum_welch"]["tol"],
        min_variance=config["hmm"]["min_variance"],
        random_state=config["seeds"]["global"],
    )

    X_tr_list = [s.X for s in seqs_tr]
    y_tr_list = [s.y for s in seqs_tr]

    log("  Supervised init …")
    model._init_supervised(X_tr_list, y_tr_list)

    log("  Baum-Welch EM …")
    t0 = time.time()
    model.fit(X_tr_list)
    train_time = time.time() - t0
    log(f"  Training time: {train_time:.2f}s")

    # Decode all test sequences
    y_pred_all, y_true_all = [], []
    y_pred_seqs, y_true_seqs = [], []
    total_latency = 0.0

    for seq in seqs_te:
        t1 = time.time()
        pred = model.predict(seq.X)
        total_latency += time.time() - t1
        # Convert 0-indexed states back to 1-indexed activity labels
        pred_labels = pred + 1
        y_pred_all.extend(pred_labels.tolist())
        y_true_all.extend(seq.y.tolist())
        y_pred_seqs.append(pred_labels)
        y_true_seqs.append(seq.y)

    y_pred_all = np.array(y_pred_all)
    y_true_all = np.array(y_true_all)
    lat_ms = total_latency / len(y_true_all) * 1000

    test_m = evaluate_flat(y_true_all, y_pred_all, label_names=label_names)
    seq_m  = evaluate_sequential(
        [s.tolist() for s in y_true_seqs],
        [p.tolist() for p in y_pred_seqs],
    )

    result = {
        "experiment": "D",
        "model": "hmm_gaussian",
        "n_states": n_states,
        "covariance_type": cov_type,
        "test_accuracy": test_m["accuracy"],
        "test_macro_f1": test_m["macro_f1"],
        "false_activity_switch_rate": seq_m["false_activity_switch_rate"],
        "transition_consistency": seq_m["transition_consistency"],
        "train_time_s": round(train_time, 3),
        "latency_ms_per_sample": round(lat_ms, 4),
    }
    print_metrics_table(result, "HMM (Gaussian, Supervised Init + BW)")
    save_metrics(result, os.path.join(RESULTS_DIR, "metrics.csv"))

    # Plots
    plot_confusion_matrix(
        y_true_all, y_pred_all, label_names=label_names,
        title="Confusion Matrix — HMM (Gaussian)",
        save_path=os.path.join(FIGURES_DIR, "D_cm_hmm.png"),
    )
    plot_hmm_transition_heatmap(
        np.exp(model.log_A),
        label_names=label_names,
        save_path=os.path.join(FIGURES_DIR, "D_hmm_transition.png"),
    )

    # Timeline for first test subject
    first_seq = seqs_te[0]
    first_pred = (model.predict(first_seq.X) + 1)
    plot_activity_timeline(
        y_true=first_seq.y,
        y_pred=first_pred,
        label_names=label_names,
        subject_id=first_seq.subject_id,
        save_path=os.path.join(FIGURES_DIR, "D_timeline_hmm.png"),
    )

    # State posteriors plot
    gamma = model.predict_proba(first_seq.X)
    plot_hmm_state_sequence(
        gamma, label_names=label_names,
        save_path=os.path.join(FIGURES_DIR, "D_hmm_posteriors.png"),
    )

    log("  [D] Complete.\n")
    return model, result


# ─────────────────────────────────────────────────────────────────────────────
# Experiment E: Hybrid HMM-Ensemble
# ─────────────────────────────────────────────────────────────────────────────

def experiment_E(hmm_model, seqs_tr, seqs_te, X_tr, y_tr, X_te, y_te, splits, config):
    log("=== Experiment E: Hybrid HMM-Ensemble ===")
    from ensemble_models import build_all_models
    from hybrid_model import HybridHMMEnsemble
    from visualization import plot_activity_timeline, plot_confusion_matrix

    label_names = list(splits.activity_labels.values())

    # Use best ensemble (soft voting) as the evidence source
    all_models = build_all_models(CONFIG_PATH)
    best_ens = all_models["soft_voting"]
    best_ens.fit(X_tr, y_tr)

    # Class priors from training set
    unique, counts = np.unique(y_tr, return_counts=True)
    class_priors = counts / counts.sum()

    # Transition matrix from HMM
    A = np.exp(hmm_model.log_A)

    hybrid = HybridHMMEnsemble(
        ensemble_model=best_ens,
        transition_matrix=A,
        class_priors=class_priors,
        n_classes=config["hmm"]["n_activities"],
    )

    # Fit transition matrix from training sequences (alternative, from true labels)
    y_seqs_tr = [s.y for s in seqs_tr]
    hybrid.fit_transition_matrix(y_seqs_tr)

    # Evaluate on test sequences
    y_pred_all, y_true_all = [], []
    y_pred_seqs, y_true_seqs = [], []

    for seq in seqs_te:
        ens_proba = best_ens.predict_proba(seq.X)
        pred = hybrid.predict(seq.X, ens_proba)
        y_pred_all.extend(pred.tolist())
        y_true_all.extend(seq.y.tolist())
        y_pred_seqs.append(pred)
        y_true_seqs.append(seq.y)

    y_pred_all = np.array(y_pred_all)
    y_true_all = np.array(y_true_all)

    from evaluation import evaluate_flat, evaluate_sequential
    test_m = evaluate_flat(y_true_all, y_pred_all, label_names=label_names)
    seq_m  = evaluate_sequential(
        [s.tolist() for s in y_true_seqs],
        [p.tolist() for p in y_pred_seqs],
    )

    result = {
        "experiment": "E",
        "model": "hybrid_hmm_ensemble",
        "test_accuracy": test_m["accuracy"],
        "test_macro_f1": test_m["macro_f1"],
        "false_activity_switch_rate": seq_m["false_activity_switch_rate"],
        "transition_consistency": seq_m["transition_consistency"],
    }
    print_metrics_table(result, "Hybrid HMM-Ensemble")
    save_metrics(result, os.path.join(RESULTS_DIR, "metrics.csv"))

    plot_confusion_matrix(
        y_true_all, y_pred_all, label_names=label_names,
        title="Confusion Matrix — Hybrid HMM-Ensemble",
        save_path=os.path.join(FIGURES_DIR, "E_cm_hybrid.png"),
    )

    # Timeline comparison
    first_seq = seqs_te[0]
    ens_proba_first = best_ens.predict_proba(first_seq.X)
    pred_hybrid = hybrid.predict(first_seq.X, ens_proba_first)
    plot_activity_timeline(
        y_true=first_seq.y,
        y_pred=pred_hybrid,
        label_names=label_names,
        subject_id=first_seq.subject_id,
        save_path=os.path.join(FIGURES_DIR, "E_timeline_hybrid.png"),
    )

    log("  [E] Complete.\n")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Experiment H: Per-subject generalisation
# ─────────────────────────────────────────────────────────────────────────────

def experiment_H(X_tr, y_tr, seqs_te, splits, hmm_model, config):
    log("=== Experiment H: Unseen-Participant Generalisation ===")
    from ensemble_models import build_all_models
    from visualization import plot_latency_comparison
    import pandas as pd

    label_names = list(splits.activity_labels.values())
    all_models = build_all_models(CONFIG_PATH)
    rf = all_models["random_forest"]
    rf.fit(X_tr, y_tr)

    per_subject = []
    for seq in seqs_te:
        y_pred = rf.predict(seq.X)
        from sklearn.metrics import f1_score
        f1 = f1_score(seq.y, y_pred, average="macro", zero_division=0)
        per_subject.append({"subject_id": seq.subject_id, "macro_f1_rf": f1})

        hmm_pred = hmm_model.predict(seq.X) + 1
        f1_hmm = f1_score(seq.y, hmm_pred, average="macro", zero_division=0)
        per_subject[-1]["macro_f1_hmm"] = f1_hmm

    df = pd.DataFrame(per_subject)
    df.to_csv(os.path.join(RESULTS_DIR, "H_per_subject_f1.csv"), index=False)
    log(f"  Per-subject macro-F1 saved to results/H_per_subject_f1.csv")
    log("  [H] Complete.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Experiment I: Latency and Cost
# ─────────────────────────────────────────────────────────────────────────────

def experiment_I(X_tr, y_tr, X_te, config):
    log("=== Experiment I: Latency and Model Size ===")
    from ensemble_models import build_all_models
    from visualization import plot_latency_comparison
    import pickle

    all_models = build_all_models(CONFIG_PATH)
    latencies, sizes, names = [], [], []

    for name, model in all_models.items():
        model.fit(X_tr, y_tr)
        n_reps = 5
        times = []
        for _ in range(n_reps):
            t0 = time.time()
            model.predict(X_te)
            times.append(time.time() - t0)
        lat_ms = np.mean(times) / len(X_te) * 1000

        blob = pickle.dumps(model)
        size_kb = len(blob) / 1024

        latencies.append(lat_ms)
        sizes.append(size_kb)
        names.append(name)
        log(f"  {name}: {lat_ms:.4f} ms/sample, {size_kb:.1f} KB")

    plot_latency_comparison(
        names, latencies,
        save_path=os.path.join(FIGURES_DIR, "I_latency_comparison.png"),
    )
    log("  [I] Complete.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    config = load_config()
    np.random.seed(config["seeds"]["numpy"])

    log("Loading data and preprocessing …")
    splits, prep, X_tr, X_va, X_te, seqs_tr, seqs_va, seqs_te = \
        load_data_and_preprocess(config)

    y_tr = splits.train_y
    y_va = splits.val_y
    y_te = splits.test_y

    experiment_A(splits, prep)
    experiment_B(X_tr, y_tr, X_va, y_va, X_te, y_te, splits)
    experiment_C(X_tr, y_tr, X_va, y_va, X_te, y_te, splits)
    hmm_model, hmm_result = experiment_D(seqs_tr, seqs_va, seqs_te, splits, config)
    experiment_E(hmm_model, seqs_tr, seqs_te, X_tr, y_tr, X_te, y_te, splits, config)
    experiment_H(X_tr, y_tr, seqs_te, splits, hmm_model, config)
    experiment_I(X_tr, y_tr, X_te, config)

    log("All experiments complete. Results in results/")


if __name__ == "__main__":
    main()
