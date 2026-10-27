"""
hybrid_model.py
===============
HMM-Ensemble hybrid classifier for Human Activity Recognition.

Strategy — "Emission Evidence Substitution"
-------------------------------------------
Rather than computing Gaussian emission probabilities from raw features (as a
standard HMM would), we leverage a pre-trained ensemble classifier to
estimate the posterior class distribution at each time step:

    P(y | x_t)  –  from ensemble.predict_proba(x_t)

We then convert posteriors to scaled likelihoods by dividing by class priors:

    L(x_t | state=s)  ∝  P(y=s | x_t) / P(y=s)

These scaled likelihoods are used as the emission distribution B[t, s] inside
the standard HMM forward–backward and Viterbi algorithms. The transition
matrix A is estimated from the label sequence of the training data, capturing
real-world temporal dynamics.

Labels:
Internally uses 0-indexed states (0..n_classes-1).
Accepts 1-indexed activity labels (1..6) and returns 1-indexed predictions
matching the UCI HAR convention.
"""

from __future__ import annotations

from typing import List, Optional
import numpy as np

from src.hmm_algorithms import (
    scaled_forward,
    scaled_backward,
    state_posteriors,
    viterbi,
    safe_log,
)

_EPS = 1e-12


class HybridHMMEnsemble:
    """Uses ensemble posterior class probabilities as HMM emission evidence."""

    def __init__(
        self,
        ensemble_model=None,
        transition_matrix: Optional[np.ndarray] = None,
        class_priors: Optional[np.ndarray] = None,
        n_classes: int = 6,
    ):
        self.ensemble_model = ensemble_model
        self.n_classes = n_classes

        if transition_matrix is not None:
            self.transition_matrix = np.asarray(transition_matrix, dtype=float)
            row_sums = self.transition_matrix.sum(axis=1, keepdims=True)
            self.transition_matrix /= np.where(row_sums == 0, 1.0, row_sums)
        else:
            self.transition_matrix = None

        if class_priors is not None:
            self.class_priors = np.asarray(class_priors, dtype=float)
            self.class_priors /= self.class_priors.sum()
        else:
            self.class_priors = np.ones(n_classes) / n_classes

        self.startprob_: Optional[np.ndarray] = None

    def fit_transition_matrix(self, y_sequences_list: List[np.ndarray]) -> "HybridHMMEnsemble":
        """
        Estimate transition matrix A and initial-state distribution π from
        label sequences using maximum-likelihood estimation with Laplace smoothing.
        Accepts 1-indexed (1..n_classes) or 0-indexed sequences.
        """
        S = self.n_classes
        trans_counts = np.zeros((S, S), dtype=float)
        init_counts = np.zeros(S, dtype=float)
        prior_counts = np.zeros(S, dtype=float)

        for y_seq in y_sequences_list:
            y_seq = np.asarray(y_seq, dtype=int)
            if len(y_seq) == 0:
                continue

            # Convert to 0-indexed if labels are 1-indexed (e.g. max label == S)
            if np.max(y_seq) == S and np.min(y_seq) >= 1:
                seq = y_seq - 1
            else:
                seq = y_seq

            # Initial state
            s0 = seq[0]
            if 0 <= s0 < S:
                init_counts[s0] += 1.0

            # Prior counts and transitions
            for t in range(len(seq)):
                st = seq[t]
                if 0 <= st < S:
                    prior_counts[st] += 1.0
                if t < len(seq) - 1:
                    st_next = seq[t + 1]
                    if 0 <= st < S and 0 <= st_next < S:
                        trans_counts[st, st_next] += 1.0

        # Laplace smoothing
        trans_counts += 1e-6
        row_sums = trans_counts.sum(axis=1, keepdims=True)
        self.transition_matrix = trans_counts / row_sums

        init_total = init_counts.sum() + _EPS
        self.startprob_ = (init_counts + 1e-6) / (init_total + S * 1e-6)

        prior_total = prior_counts.sum() + _EPS
        self.class_priors = prior_counts / prior_total

        return self

    def predict(
        self,
        X_sequence: np.ndarray,
        ensemble_proba: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Viterbi-decode the most likely activity sequence.
        Returns 1-indexed predicted activity labels (1..n_classes).
        """
        self._check_fitted()
        B = self._compute_emission_matrix(X_sequence, ensemble_proba)
        pi = self.startprob_ if self.startprob_ is not None else np.ones(self.n_classes) / self.n_classes
        A = self.transition_matrix

        log_pi = safe_log(pi)
        log_A  = safe_log(A)
        log_B  = safe_log(B)

        path, _ = viterbi(log_B, log_pi, log_A)
        return path + 1  # Return 1-indexed activity labels

    def predict_proba(
        self,
        X_sequence: np.ndarray,
        ensemble_proba: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Forward-backward posterior probabilities γ_t(s) of shape (T, n_classes)."""
        self._check_fitted()
        B = self._compute_emission_matrix(X_sequence, ensemble_proba)
        pi = self.startprob_ if self.startprob_ is not None else np.ones(self.n_classes) / self.n_classes
        A  = self.transition_matrix

        log_pi = safe_log(pi)
        log_A  = safe_log(A)
        log_B  = safe_log(B)

        alpha, log_scales, _ = scaled_forward(log_B, log_pi, log_A)
        beta = scaled_backward(log_B, log_A, log_scales)
        gamma = state_posteriors(alpha, beta)
        return gamma

    def _compute_emission_matrix(
        self,
        X_sequence: np.ndarray,
        ensemble_proba: Optional[np.ndarray],
    ) -> np.ndarray:
        """Convert ensemble posteriors to scaled emission likelihoods B[t, s]."""
        if ensemble_proba is None:
            if self.ensemble_model is None:
                raise ValueError("Either ensemble_model or ensemble_proba must be supplied.")
            ensemble_proba = self.ensemble_model.predict_proba(X_sequence)

        proba = np.asarray(ensemble_proba, dtype=float)
        priors = np.maximum(self.class_priors, _EPS)
        B = proba / priors[None, :]

        # Row-normalise for numerical stability
        row_sums = B.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        return B / row_sums

    def _check_fitted(self) -> None:
        if self.transition_matrix is None:
            raise RuntimeError("HybridHMMEnsemble is not fitted. Call fit_transition_matrix first.")
