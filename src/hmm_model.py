"""
hmm_model.py
============
Gaussian Hidden Markov Model for Human Activity Recognition.

API compatibility with hmm_algorithms.py
-----------------------------------------
All internal calls use the log-space signatures defined in hmm_algorithms.py:
  scaled_forward(log_B, log_pi, log_A)  -> (alpha_scaled, log_scales, A)
  scaled_backward(log_B, log_A, log_scales) -> beta_scaled
  state_posteriors(alpha_scaled, beta_scaled) -> gamma
  xi_posteriors(alpha_scaled, beta_scaled, log_B, log_A) -> xi
  viterbi(log_B, log_pi, log_A) -> (path, log_prob)
  log_likelihood(log_scales) -> float

Labels
------
y_list elements contain 1-indexed activity labels (1..n_states).
Internally, state indexing is 0-based (state s corresponds to label s+1).
predict() returns 0-indexed states; callers add +1 to get activity labels.

Typical usage::

    model = GaussianHMM(n_states=6, covariance_type="diag")
    model._init_supervised(X_list, y_list)   # y_list: labels 1-6
    model.fit(X_list)
    states = model.predict(X_seq)            # returns 0-indexed states
    activity_labels = states + 1             # 1-indexed labels
"""

from __future__ import annotations

import numpy as np

from src.hmm_algorithms import (
    scaled_forward,
    scaled_backward,
    log_likelihood,
    state_posteriors,
    xi_posteriors,
    viterbi,
    safe_log,
)

_LOG_2PI = np.log(2.0 * np.pi)


class GaussianHMM:
    """
    Hidden Markov Model with Gaussian emission distributions.

    Parameters
    ----------
    n_states : int
        Number of hidden states (= number of activity classes).
    covariance_type : {"diag", "full"}
        Structure of emission covariance matrices.
    min_variance : float
        Variance floor to prevent covariance singularity.
    n_iter : int
        Maximum Baum-Welch EM iterations.
    tol : float
        Convergence threshold on log-likelihood improvement.
    verbose : bool
        Print per-iteration log-likelihood if True.
    random_state : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_states: int = 6,
        covariance_type: str = "diag",
        min_variance: float = 1e-3,
        n_iter: int = 100,
        tol: float = 1e-4,
        verbose: bool = False,
        random_state: int = 42,
    ):
        if covariance_type not in ("diag", "full"):
            raise ValueError("covariance_type must be 'diag' or 'full'.")
        self.n_states = n_states
        self.covariance_type = covariance_type
        self.min_variance = min_variance
        self.n_iter = n_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state

        # Fitted parameters (None until _init_supervised / fit is called)
        self.startprob_: np.ndarray = None    # (n_states,) – π in prob space
        self.transmat_:  np.ndarray = None    # (n_states, n_states) – A in prob space
        self.means_:     np.ndarray = None    # (n_states, n_features)
        self.covars_:    np.ndarray = None    # (n_states, n_features[, n_features])

        # Convenient log-space versions (set in _update_log_params)
        self.log_pi: np.ndarray = None
        self.log_A:  np.ndarray = None

    # ── Public API ─────────────────────────────────────────────────────────

    def fit(self, sequences_X_list: list) -> "GaussianHMM":
        """
        Multi-sequence Baum-Welch EM.

        Parameters
        ----------
        sequences_X_list : list of np.ndarray, each shape (T_i, n_features)
        """
        if self.startprob_ is None:
            self._random_init(sequences_X_list)
        self._update_log_params()

        prev_ll = -np.inf
        for iteration in range(self.n_iter):
            (
                gamma_sum,
                gamma_obs_sum,
                gamma_obs_sq_sum,
                xi_sum,
                gamma_t0_sum,
                total_ll,
            ) = self._e_step(sequences_X_list)

            self._m_step(
                gamma_sum,
                gamma_obs_sum,
                gamma_obs_sq_sum,
                xi_sum,
                gamma_t0_sum,
            )
            self._update_log_params()

            delta_ll = total_ll - prev_ll
            if self.verbose:
                print(f"  [HMM iter {iteration+1:3d}] log-lik = {total_ll:.4f}  Δ = {delta_ll:.6f}")

            if iteration > 0 and delta_ll < self.tol:
                if self.verbose:
                    print(f"  Converged after {iteration+1} iterations.")
                break
            prev_ll = total_ll

        return self

    def predict(self, X_sequence: np.ndarray) -> np.ndarray:
        """
        Viterbi decoding — most likely state sequence.

        Returns
        -------
        path : (T,) int array of 0-indexed states (add +1 for activity labels)
        """
        log_B = self._compute_log_emission_probs(X_sequence)
        path, _ = viterbi(log_B, self.log_pi, self.log_A)
        return path

    def predict_proba(self, X_sequence: np.ndarray) -> np.ndarray:
        """
        Forward-backward posteriors γ_t(i) = P(state=i | X, λ).

        Returns
        -------
        gamma : (T, n_states) — each row sums to 1
        """
        log_B = self._compute_log_emission_probs(X_sequence)
        alpha, log_scales, _ = scaled_forward(log_B, self.log_pi, self.log_A)
        beta = scaled_backward(log_B, self.log_A, log_scales)
        gamma = state_posteriors(alpha, beta)
        return gamma

    def score(self, X_sequence: np.ndarray) -> float:
        """
        Log-likelihood  ln P(X | λ).
        """
        log_B = self._compute_log_emission_probs(X_sequence)
        _, log_scales, _ = scaled_forward(log_B, self.log_pi, self.log_A)
        return log_likelihood(log_scales)

    # ── Supervised Initialisation ─────────────────────────────────────────

    def _init_supervised(self, X_list: list, y_list: list) -> None:
        """
        Initialise all HMM parameters from labelled (X, y) data.

        Parameters
        ----------
        X_list : list of np.ndarray, each shape (T_i, n_features)
        y_list : list of np.ndarray, each shape (T_i,)
            1-indexed activity labels (1 … n_states).

        Implementation
        --------------
        State s (0-indexed) corresponds to activity label (s + 1).
        - means_[s]  = mean of X windows with label s+1
        - covars_[s] = variance (diag) or covariance (full) of those windows
        - A[s, s']   = count of transitions (label s+1) → (label s'+1) / total
        - π[s]       = fraction of sequences starting with label s+1
        """
        n_features = X_list[0].shape[1]
        S = self.n_states

        # ── Emissions ────────────────────────────────────────────────────
        means = np.zeros((S, n_features))
        if self.covariance_type == "diag":
            covars = np.full((S, n_features), self.min_variance)
        else:
            covars = np.stack([np.eye(n_features) * self.min_variance
                               for _ in range(S)])

        for s in range(S):
            label = s + 1   # 1-indexed activity label for state s
            Xs_parts = [X[y == label] for X, y in zip(X_list, y_list)
                        if np.any(y == label)]
            if not Xs_parts:
                continue
            Xs = np.vstack(Xs_parts)
            if len(Xs) == 0:
                continue
            means[s] = Xs.mean(axis=0)
            if self.covariance_type == "diag":
                var = Xs.var(axis=0)
                covars[s] = np.maximum(var, self.min_variance)
            else:
                cov = np.cov(Xs.T) if Xs.shape[0] > 1 else \
                      np.diag(np.full(n_features, self.min_variance))
                np.fill_diagonal(cov, np.maximum(np.diag(cov), self.min_variance))
                covars[s] = cov

        self.means_ = means
        self.covars_ = covars

        # ── Transition matrix (1-indexed labels → 0-indexed states) ──────
        A = np.zeros((S, S)) + 1e-6   # Laplace smoothing
        for y in y_list:
            y_arr = np.asarray(y)
            for t in range(len(y_arr) - 1):
                i = int(y_arr[t]) - 1    # 0-indexed state
                j = int(y_arr[t + 1]) - 1
                if 0 <= i < S and 0 <= j < S:
                    A[i, j] += 1.0
        self.transmat_ = A / A.sum(axis=1, keepdims=True)

        # ── Initial state distribution ───────────────────────────────────
        pi = np.zeros(S) + 1e-6
        for y in y_list:
            y_arr = np.asarray(y)
            if len(y_arr) > 0:
                s0 = int(y_arr[0]) - 1   # 0-indexed
                if 0 <= s0 < S:
                    pi[s0] += 1.0
        self.startprob_ = pi / pi.sum()

        self._update_log_params()

    # ── Emission log-probability ────────────────────────────────────────

    def _compute_log_emission_probs(self, X: np.ndarray) -> np.ndarray:
        """
        log N(x_t | μ_s, Σ_s) for each (t, s) pair.

        Returns
        -------
        log_B : (T, n_states)
        """
        T, n_features = X.shape
        S = self.n_states
        log_B = np.zeros((T, S))

        for s in range(S):
            mu   = self.means_[s]
            diff = X - mu        # (T, n_features)

            if self.covariance_type == "diag":
                var     = self.covars_[s]                         # (n_features,)
                log_det = np.sum(np.log(var))
                mah     = np.sum(diff ** 2 / var, axis=1)        # (T,)
            else:
                cov = self.covars_[s]
                try:
                    L       = np.linalg.cholesky(cov)
                    log_det = 2.0 * np.sum(np.log(np.diag(L)))
                    z       = np.linalg.solve(L, diff.T)          # (n_features, T)
                    mah     = np.sum(z ** 2, axis=0)              # (T,)
                except np.linalg.LinAlgError:
                    sign, log_det = np.linalg.slogdet(cov)
                    log_det = log_det if sign > 0 else 0.0
                    cov_inv = np.linalg.pinv(cov)
                    mah     = np.einsum("ti,ij,tj->t", diff, cov_inv, diff)

            log_B[:, s] = -0.5 * (n_features * _LOG_2PI + log_det + mah)

        return log_B

    # ── E-step / M-step ────────────────────────────────────────────────

    def _e_step(self, sequences_X_list: list):
        """Accumulate sufficient statistics (E-step) over all sequences."""
        S = self.n_states
        F = self.means_.shape[1]

        gamma_sum        = np.zeros(S)
        gamma_obs_sum    = np.zeros((S, F))
        gamma_obs_sq_sum = np.zeros((S, F)) if self.covariance_type == "diag" \
                           else np.zeros((S, F, F))
        xi_sum           = np.zeros((S, S))
        gamma_t0_sum     = np.zeros(S)
        total_ll         = 0.0

        for X in sequences_X_list:
            if len(X) == 0:
                continue
            log_B = self._compute_log_emission_probs(X)          # (T, S)
            alpha, log_scales, _ = scaled_forward(log_B, self.log_pi, self.log_A)
            beta = scaled_backward(log_B, self.log_A, log_scales)
            gamma = state_posteriors(alpha, beta)                 # (T, S)
            xi    = xi_posteriors(alpha, beta, log_B, self.log_A)  # (T-1, S, S)

            total_ll     += log_likelihood(log_scales)
            gamma_t0_sum += gamma[0]

            for s in range(S):
                g_s = gamma[:, s]                                 # (T,)
                gamma_sum[s]     += g_s.sum()
                gamma_obs_sum[s] += (g_s[:, None] * X).sum(axis=0)
                if self.covariance_type == "diag":
                    gamma_obs_sq_sum[s] += (g_s[:, None] * X ** 2).sum(axis=0)
                else:
                    gamma_obs_sq_sum[s] += np.einsum("t,ti,tj->ij", g_s, X, X)

            xi_sum += xi.sum(axis=0)

        return (gamma_sum, gamma_obs_sum, gamma_obs_sq_sum,
                xi_sum, gamma_t0_sum, total_ll)

    def _m_step(
        self,
        gamma_sum:        np.ndarray,
        gamma_obs_sum:    np.ndarray,
        gamma_obs_sq_sum: np.ndarray,
        xi_sum:           np.ndarray,
        gamma_t0_sum:     np.ndarray,
    ) -> None:
        """Update parameters from sufficient statistics (M-step)."""
        S = self.n_states

        # π
        pi = gamma_t0_sum + 1e-10
        self.startprob_ = pi / pi.sum()

        # A
        A = xi_sum + 1e-10
        self.transmat_ = A / A.sum(axis=1, keepdims=True)

        # Emissions
        for s in range(S):
            denom = gamma_sum[s] + 1e-10
            self.means_[s] = gamma_obs_sum[s] / denom

            if self.covariance_type == "diag":
                e_x2  = gamma_obs_sq_sum[s] / denom
                var   = e_x2 - self.means_[s] ** 2
                self.covars_[s] = np.maximum(var, self.min_variance)
            else:
                e_xxT = gamma_obs_sq_sum[s] / denom
                mu    = self.means_[s]
                cov   = e_xxT - np.outer(mu, mu)
                np.fill_diagonal(cov, np.maximum(np.diag(cov), self.min_variance))
                self.covars_[s] = cov

    # ── Log-space parameter cache ────────────────────────────────────────

    def _update_log_params(self) -> None:
        """Refresh log_pi and log_A from prob-space counterparts."""
        self.log_pi = safe_log(self.startprob_)
        self.log_A  = safe_log(self.transmat_)

    # ── Random initialisation ────────────────────────────────────────────

    def _random_init(self, sequences_X_list: list) -> None:
        """Random parameter initialisation (fallback when no labels)."""
        rng = np.random.RandomState(self.random_state)
        S = self.n_states
        X_all = np.vstack(sequences_X_list)
        F = X_all.shape[1]
        global_mean = X_all.mean(axis=0)
        global_std  = X_all.std(axis=0) + 1e-6

        self.startprob_ = np.ones(S) / S
        self.transmat_  = rng.dirichlet(np.ones(S), size=S)
        self.means_     = global_mean[None, :] + rng.randn(S, F) * global_std
        if self.covariance_type == "diag":
            self.covars_ = np.tile(np.maximum(global_std ** 2, self.min_variance), (S, 1))
        else:
            cov = np.diag(np.maximum(global_std ** 2, self.min_variance))
            self.covars_ = np.stack([cov.copy() for _ in range(S)])

    def __repr__(self) -> str:
        return (f"GaussianHMM(n_states={self.n_states}, "
                f"covariance_type='{self.covariance_type}', "
                f"min_variance={self.min_variance})")
