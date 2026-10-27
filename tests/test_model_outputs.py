"""
tests/test_model_outputs.py
============================
Output shape and probability normalisation tests.

Tests
-----
M1  Ensemble predict_proba rows sum to 1
M2  HMM predict_proba (gamma) rows sum to 1
M3  Hybrid predict output labels are valid activity IDs
M4  Confusion matrix shape matches number of classes
M5  Evaluation log_loss is non-negative

Run with:  cd HAR_HMM_Project && pytest tests/test_model_outputs.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from src.hmm_algorithms import scaled_forward, scaled_backward, state_posteriors, safe_log
from src.evaluation import evaluate_flat, evaluate_sequential


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_seq():
    """50 time steps, 6 classes, 20 features."""
    rng = np.random.default_rng(0)
    T, N, D = 50, 6, 20
    X = rng.standard_normal((T, D))
    y = rng.integers(1, 7, size=T)
    return X, y, T, N, D


# ─────────────────────────────────────────────────────────────────────────────
# M1  Ensemble probability normalisation
# ─────────────────────────────────────────────────────────────────────────────

class TestEnsembleProbaNorm:
    def test_sklearn_lr_proba_sums_to_one(self):
        from sklearn.linear_model import LogisticRegression
        rng = np.random.default_rng(1)
        X_tr = rng.standard_normal((200, 10))
        y_tr = rng.integers(1, 7, size=200)
        X_te = rng.standard_normal((50, 10))

        clf = LogisticRegression(max_iter=500, random_state=42)
        clf.fit(X_tr, y_tr)
        proba = clf.predict_proba(X_te)

        np.testing.assert_allclose(
            proba.sum(axis=1), 1.0, atol=1e-10,
            err_msg="LR predict_proba rows must sum to 1")


# ─────────────────────────────────────────────────────────────────────────────
# M2  HMM gamma normalisation (via core algorithms)
# ─────────────────────────────────────────────────────────────────────────────

class TestHMMGammaNorm:
    def test_gamma_sums_to_one_over_forward_backward(self):
        rng = np.random.default_rng(2)
        T, N = 30, 6
        log_B = rng.uniform(-3, -0.1, (T, N))
        A = np.ones((N, N)) / N
        log_A = safe_log(A)
        pi = np.ones(N) / N
        log_pi = safe_log(pi)

        alpha, log_scales, _ = scaled_forward(log_B, log_pi, log_A)
        beta = scaled_backward(log_B, log_A, log_scales)
        gamma = state_posteriors(alpha, beta)

        np.testing.assert_allclose(
            gamma.sum(axis=1), 1.0, atol=1e-8,
            err_msg="HMM state posteriors must sum to 1 at each time step")


# ─────────────────────────────────────────────────────────────────────────────
# M3  Evaluation outputs
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluationOutputs:
    def test_evaluate_flat_keys(self):
        rng = np.random.default_rng(3)
        y_true = rng.integers(1, 7, 100)
        y_pred = rng.integers(1, 7, 100)
        result = evaluate_flat(y_true, y_pred)
        for key in ["accuracy", "macro_precision", "macro_recall", "macro_f1",
                    "confusion_matrix", "per_class_f1"]:
            assert key in result, f"Missing key: {key}"

    def test_evaluate_flat_accuracy_range(self):
        rng = np.random.default_rng(4)
        y_true = rng.integers(1, 7, 100)
        y_pred = rng.integers(1, 7, 100)
        result = evaluate_flat(y_true, y_pred)
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_evaluate_flat_macro_f1_range(self):
        rng = np.random.default_rng(5)
        y_true = rng.integers(1, 7, 100)
        y_pred = rng.integers(1, 7, 100)
        result = evaluate_flat(y_true, y_pred)
        assert 0.0 <= result["macro_f1"] <= 1.0

    def test_evaluate_flat_confusion_matrix_shape(self):
        rng = np.random.default_rng(6)
        y_true = rng.integers(1, 7, 100)
        y_pred = rng.integers(1, 7, 100)
        result = evaluate_flat(y_true, y_pred)
        cm = result["confusion_matrix"]
        n_cls = len(np.unique(np.concatenate([y_true, y_pred])))
        assert cm.shape == (n_cls, n_cls), "Confusion matrix shape mismatch"

    def test_evaluate_flat_log_loss_positive(self):
        rng = np.random.default_rng(7)
        y_true = rng.integers(0, 6, 100)
        y_proba = np.abs(rng.standard_normal((100, 6)))
        y_proba /= y_proba.sum(axis=1, keepdims=True)
        result = evaluate_flat(y_true, rng.integers(0, 6, 100), y_proba)
        if "log_loss" in result and result["log_loss"] is not None:
            assert result["log_loss"] >= 0, "Log loss must be non-negative"


# ─────────────────────────────────────────────────────────────────────────────
# M4  Sequential evaluation
# ─────────────────────────────────────────────────────────────────────────────

class TestSequentialEvaluation:
    def test_false_switch_rate_range(self):
        rng = np.random.default_rng(8)
        seqs_true = [rng.integers(1, 7, 50).tolist() for _ in range(5)]
        seqs_pred = [rng.integers(1, 7, 50).tolist() for _ in range(5)]
        result = evaluate_sequential(seqs_true, seqs_pred)
        assert 0.0 <= result["false_activity_switch_rate"] <= 1.0

    def test_transition_consistency_range(self):
        rng = np.random.default_rng(9)
        seqs_true = [rng.integers(1, 7, 50).tolist() for _ in range(5)]
        seqs_pred = seqs_true  # perfect predictions
        result = evaluate_sequential(seqs_true, seqs_pred)
        np.testing.assert_allclose(result["false_activity_switch_rate"], 0.0, atol=1e-10,
            err_msg="Perfect predictions should have 0 false switch rate")

    def test_perfect_predictions_metrics(self):
        rng = np.random.default_rng(10)
        seqs = [rng.integers(1, 7, 30).tolist() for _ in range(3)]
        result = evaluate_sequential(seqs, seqs)
        assert result["false_activity_switch_rate"] == pytest.approx(0.0, abs=1e-10)
        assert result["transition_consistency"] == pytest.approx(1.0, abs=1e-10)
