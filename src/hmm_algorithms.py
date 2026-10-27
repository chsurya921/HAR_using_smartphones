"""
hmm_algorithms.py
=================
From-scratch NumPy implementation of core HMM algorithms (Unit 4).

Mathematical definitions
------------------------
A discrete-time HMM is defined by the triple  λ = (π, A, B)  where:
  π_i   = P(q_1 = i)                        — initial state distribution
  A_ij  = P(q_{t+1} = j | q_t = i)          — state transition matrix
  B_i(x_t) = p(x_t | q_t = i)              — emission probability / density

Implemented algorithms
----------------------
1. scaled_forward       — α̃_t(i) with scaling factors c_t to prevent underflow
2. scaled_backward      — β̃_t(i) using the same scaling factors
3. log_likelihood       — ln P(O | λ) = -Σ ln c_t
4. state_posteriors     — γ_t(i) = P(q_t = i | O, λ)
5. xi_posteriors        — ξ_t(i,j) = P(q_t=i, q_{t+1}=j | O, λ)
6. viterbi              — argmax_{q} P(q | O, λ)  via dynamic programming
7. log_forward          — alternative log-space forward (for verification)

All functions operate on 1-D observation sequences of pre-computed emission
log-probabilities so that the same code works for any emission model
(Gaussian, GMM, …).

Every function is tested in  tests/test_hmm_algorithms.py.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Scaled Forward Algorithm
# ─────────────────────────────────────────────────────────────────────────────

def scaled_forward(
    log_B: np.ndarray,      # (T, N) — log emission probabilities
    log_pi: np.ndarray,     # (N,)   — log initial distribution
    log_A: np.ndarray,      # (N, N) — log transition matrix
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Scaled forward algorithm.

    At each time step t the raw forward variable is:
        α_t(i) = P(x_1,...,x_t, q_t=i | λ)

    To avoid numerical underflow for long sequences, we compute the scaled
    version:
        c_t    = Σ_i α_t(i)         (normalisation constant)
        α̃_t(i) = α_t(i) / c_t

    Recursion
    ---------
        α̃_1(i) = π_i · B_i(x_1) / c_1
        α̃_t(j) = [Σ_i α̃_{t-1}(i) · A_ij] · B_j(x_t) / c_t

    Log-likelihood
    --------------
        ln P(O | λ) = Σ_t ln c_t   (returned separately — use log_likelihood)

    Parameters
    ----------
    log_B  : (T, N) log emission probabilities — can be -inf for zero density
    log_pi : (N,)   log initial state distribution
    log_A  : (N, N) log transition matrix (rows must sum to 1 in prob space)

    Returns
    -------
    alpha_scaled : (T, N)  scaled forward variables α̃_t(i)
    log_scales   : (T,)    log(c_t) for each time step
    A            : (N, N)  transition matrix in probability space (convenience)
    """
    T, N = log_B.shape
    log_A_arr = np.asarray(log_A)
    A = np.exp(log_A_arr)

    alpha_scaled = np.zeros((T, N))
    log_scales = np.zeros(T)

    # ── t = 0 (initialisation) ────────────────────────────────────────────
    alpha_0_raw = np.exp(log_pi + log_B[0])          # π_i · B_i(x_1)
    c_0 = alpha_0_raw.sum()
    assert c_0 > 0, "All emission probabilities are zero at t=0; check model."
    alpha_scaled[0] = alpha_0_raw / c_0
    log_scales[0] = np.log(c_0)

    # ── t = 1 … T-1 (recursion) ───────────────────────────────────────────
    for t in range(1, T):
        # shape: (N,) — sum over previous states
        alpha_t_raw = (alpha_scaled[t - 1] @ A) * np.exp(log_B[t])
        c_t = alpha_t_raw.sum()
        if c_t == 0:
            # All paths vanish — set uniform (will show as -inf in log-lik)
            alpha_scaled[t] = 1.0 / N
            log_scales[t] = -np.inf
        else:
            alpha_scaled[t] = alpha_t_raw / c_t
            log_scales[t] = np.log(c_t)

    return alpha_scaled, log_scales, A


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Scaled Backward Algorithm
# ─────────────────────────────────────────────────────────────────────────────

def scaled_backward(
    log_B: np.ndarray,      # (T, N)
    log_A: np.ndarray,      # (N, N)
    log_scales: np.ndarray, # (T,) — same scales from scaled_forward
) -> np.ndarray:
    """
    Scaled backward algorithm.

    β_t(i) = P(x_{t+1}, …, x_T | q_t = i, λ)

    Scaled version uses the SAME scaling factors c_t from the forward pass:
        β̃_t(i) = β_t(i) · ∏_{s=t+1}^{T} c_s

    Recursion (backward from T-1 to 0):
        β̃_T(i) = 1  (divided by c_T)
        β̃_t(i) = Σ_j A_ij · B_j(x_{t+1}) · β̃_{t+1}(j) / c_{t+1}

    Parameters
    ----------
    log_B      : (T, N)
    log_A      : (N, N)
    log_scales : (T,) from scaled_forward

    Returns
    -------
    beta_scaled : (T, N)
    """
    T, N = log_B.shape
    A = np.exp(log_A)
    beta_scaled = np.zeros((T, N))

    # ── t = T-1 (initialisation) ──────────────────────────────────────────
    beta_scaled[T - 1] = np.exp(-log_scales[T - 1])   # 1 / c_T

    # ── t = T-2 … 0 (recursion) ──────────────────────────────────────────
    for t in range(T - 2, -1, -1):
        # β̃_t(i) = Σ_j A_ij · B_j(x_{t+1}) · β̃_{t+1}(j)
        beta_t_raw = A @ (np.exp(log_B[t + 1]) * beta_scaled[t + 1])
        scale = np.exp(-log_scales[t])                 # divide by c_{t+1}? No
        beta_scaled[t] = beta_t_raw * scale

    return beta_scaled


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Log-likelihood from scaling factors
# ─────────────────────────────────────────────────────────────────────────────

def log_likelihood(log_scales: np.ndarray) -> float:
    """
    ln P(O | λ) = Σ_t ln c_t

    Parameters
    ----------
    log_scales : (T,) output of scaled_forward

    Returns
    -------
    float — log-likelihood of the observation sequence
    """
    return float(np.sum(log_scales))


# ─────────────────────────────────────────────────────────────────────────────
# 4.  State posterior probabilities γ_t(i)
# ─────────────────────────────────────────────────────────────────────────────

def state_posteriors(
    alpha_scaled: np.ndarray,   # (T, N)
    beta_scaled: np.ndarray,    # (T, N)
) -> np.ndarray:
    """
    γ_t(i) = P(q_t = i | O, λ)
           = α_t(i) · β_t(i) / Σ_j α_t(j) · β_t(j)

    With scaled variables the denominator reduces to Σ_j α̃_t(j)·β̃_t(j),
    which we re-normalise row-wise.

    Returns
    -------
    gamma : (T, N)  each row sums to 1.0
    """
    gamma_raw = alpha_scaled * beta_scaled              # element-wise (T, N)
    row_sums = gamma_raw.sum(axis=1, keepdims=True)
    # Guard against division by zero (degenerate sequences)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    gamma = gamma_raw / row_sums
    return gamma


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Joint state-pair posteriors ξ_t(i, j)
# ─────────────────────────────────────────────────────────────────────────────

def xi_posteriors(
    alpha_scaled: np.ndarray,   # (T, N)
    beta_scaled: np.ndarray,    # (T, N)
    log_B: np.ndarray,          # (T, N)
    log_A: np.ndarray,          # (N, N)
) -> np.ndarray:
    """
    ξ_t(i,j) = P(q_t=i, q_{t+1}=j | O, λ)
             ∝ α̃_t(i) · A_ij · B_j(x_{t+1}) · β̃_{t+1}(j)

    Returns
    -------
    xi : (T-1, N, N)  each slice sums to 1 over i,j
    """
    T, N = alpha_scaled.shape
    A = np.exp(log_A)
    xi = np.zeros((T - 1, N, N))

    for t in range(T - 1):
        # outer product: alpha_t (N,1) * A (N,N) * B_{t+1} (1,N) * beta_{t+1} (1,N)
        xi[t] = (
            alpha_scaled[t, :, None]        # (N, 1)
            * A                              # (N, N)
            * np.exp(log_B[t + 1])[None, :] # (1, N)
            * beta_scaled[t + 1][None, :]   # (1, N)
        )
        denom = xi[t].sum()
        if denom > 0:
            xi[t] /= denom

    return xi


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Viterbi Algorithm
# ─────────────────────────────────────────────────────────────────────────────

def viterbi(
    log_B: np.ndarray,      # (T, N)
    log_pi: np.ndarray,     # (N,)
    log_A: np.ndarray,      # (N, N)
) -> Tuple[np.ndarray, float]:
    """
    Viterbi algorithm — finds the most probable state sequence.

    Dynamic programming in log space:
        δ_t(i) = max_{q_1,…,q_{t-1}} ln P(q_1,…,q_{t-1}, q_t=i, x_1,…,x_t | λ)

    Recursion:
        δ_1(i)  = ln π_i + ln B_i(x_1)
        δ_t(j)  = max_i [δ_{t-1}(i) + ln A_ij] + ln B_j(x_t)
        ψ_t(j)  = argmax_i [δ_{t-1}(i) + ln A_ij]   (backpointer)

    Traceback:
        q*_T = argmax_i δ_T(i)
        q*_t = ψ_{t+1}(q*_{t+1})   for t = T-1 … 1

    Parameters
    ----------
    log_B  : (T, N) — log emission probabilities
    log_pi : (N,)   — log initial distribution
    log_A  : (N, N) — log transition matrix

    Returns
    -------
    path      : (T,) int array — most probable state sequence
    log_prob  : float — log-probability of the best path
    """
    T, N = log_B.shape
    log_A_arr = np.asarray(log_A, dtype=float)

    # ── Initialisation ────────────────────────────────────────────────────
    delta = np.full((T, N), -np.inf)
    psi   = np.zeros((T, N), dtype=int)

    delta[0] = log_pi + log_B[0]

    # ── Recursion ─────────────────────────────────────────────────────────
    for t in range(1, T):
        # (N, N): each row i is delta[t-1, i] + log_A[i, :]
        scores = delta[t - 1, :, None] + log_A_arr   # (N, N)
        psi[t]   = np.argmax(scores, axis=0)          # (N,)
        delta[t] = scores[psi[t], np.arange(N)] + log_B[t]

    # ── Traceback ─────────────────────────────────────────────────────────
    path = np.zeros(T, dtype=int)
    path[T - 1] = int(np.argmax(delta[T - 1]))
    log_prob = float(delta[T - 1, path[T - 1]])

    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]

    return path, log_prob


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Log-space Forward (alternative for verification)
# ─────────────────────────────────────────────────────────────────────────────

def log_forward(
    log_B: np.ndarray,      # (T, N)
    log_pi: np.ndarray,     # (N,)
    log_A: np.ndarray,      # (N, N)
) -> Tuple[np.ndarray, float]:
    """
    Log-space forward algorithm using log-sum-exp for numerical stability.

    This is independent of the scaled forward algorithm and used to
    cross-validate log-likelihood values in tests.

    Returns
    -------
    log_alpha : (T, N) — log forward variables
    log_lik   : float  — ln P(O | λ)
    """
    T, N = log_B.shape
    log_A_arr = np.asarray(log_A, dtype=float)
    log_alpha = np.full((T, N), -np.inf)

    # ── t = 0 ─────────────────────────────────────────────────────────────
    log_alpha[0] = log_pi + log_B[0]

    # ── t = 1 … T-1 ───────────────────────────────────────────────────────
    for t in range(1, T):
        # log_alpha[t, j] = log Σ_i exp(log_alpha[t-1, i] + log_A[i, j]) + log_B[t, j]
        # Use log-sum-exp trick for stability
        scores = log_alpha[t - 1, :, None] + log_A_arr   # (N, N)
        log_alpha[t] = _log_sum_exp_cols(scores) + log_B[t]

    # ── log-likelihood = log Σ_i α_T(i) ──────────────────────────────────
    log_lik = _log_sum_exp(log_alpha[T - 1])

    return log_alpha, log_lik


# ─────────────────────────────────────────────────────────────────────────────
# Helper: numerically stable log-sum-exp
# ─────────────────────────────────────────────────────────────────────────────

def _log_sum_exp(x: np.ndarray) -> float:
    """log Σ exp(x_i)  — numerically stable."""
    max_x = np.max(x)
    if max_x == -np.inf:
        return -np.inf
    return float(max_x + np.log(np.sum(np.exp(x - max_x))))


def _log_sum_exp_cols(x: np.ndarray) -> np.ndarray:
    """log Σ_i exp(x[i, j])  over axis 0 — column-wise log-sum-exp."""
    max_x = x.max(axis=0, keepdims=True)
    max_x = np.where(np.isneginf(max_x), 0.0, max_x)
    return (max_x + np.log(np.sum(np.exp(x - max_x), axis=0))).ravel()


# ─────────────────────────────────────────────────────────────────────────────
# Utility: safe log of a probability matrix
# ─────────────────────────────────────────────────────────────────────────────

def safe_log(x: np.ndarray, floor: float = 1e-300) -> np.ndarray:
    """Element-wise log with a floor to avoid log(0) = -inf propagation."""
    return np.log(np.maximum(x, floor))
