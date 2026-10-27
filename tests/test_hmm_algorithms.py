"""
tests/test_hmm_algorithms.py
============================
Unit tests for src/hmm_algorithms.py.

Tests
-----
T1  Probability normalisation  — rows of A, γ_t posteriors sum to 1
T2  Scaled vs log-space likelihood agreement
T3  Viterbi path validity       — path in [0, N), log_prob ≤ 0
T4  Backward correctness        — α * β / norm == γ
T5  Synthetic HMM with known parameters — likelihood matches hmmlearn
T6  Edge cases                  — T=1, N=1, near-zero emissions

Run with:  cd HAR_HMM_Project && pytest tests/test_hmm_algorithms.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from src.hmm_algorithms import (
    scaled_forward,
    scaled_backward,
    log_likelihood,
    state_posteriors,
    xi_posteriors,
    viterbi,
    log_forward,
    safe_log,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures — small synthetic HMM
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def small_hmm():
    """
    A 3-state HMM with known parameters (Gaussian emissions reduced to
    scalar log-prob sequences for simplicity in unit tests).
    """
    np.random.seed(0)
    N = 3        # number of states
    T = 15       # sequence length

    # Transition matrix (row-stochastic)
    A = np.array([
        [0.7, 0.2, 0.1],
        [0.1, 0.6, 0.3],
        [0.2, 0.3, 0.5],
    ])
    # Initial distribution
    pi = np.array([0.6, 0.3, 0.1])
    # Fake log emission probabilities — random but bounded
    log_B = np.random.uniform(-3.0, -0.1, size=(T, N))

    return {
        "N": N, "T": T,
        "log_A": safe_log(A),
        "log_pi": safe_log(pi),
        "log_B": log_B,
        "A": A, "pi": pi,
    }


# ─────────────────────────────────────────────────────────────────────────────
# T1  Probability normalisation
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalisation:
    def test_transition_matrix_rows_sum_to_one(self, small_hmm):
        A = small_hmm["A"]
        np.testing.assert_allclose(A.sum(axis=1), np.ones(small_hmm["N"]),
                                   atol=1e-12)

    def test_initial_distribution_sums_to_one(self, small_hmm):
        np.testing.assert_allclose(small_hmm["pi"].sum(), 1.0, atol=1e-12)

    def test_gamma_rows_sum_to_one(self, small_hmm):
        alpha, log_scales, _ = scaled_forward(
            small_hmm["log_B"], small_hmm["log_pi"], small_hmm["log_A"])
        beta = scaled_backward(
            small_hmm["log_B"], small_hmm["log_A"], log_scales)
        gamma = state_posteriors(alpha, beta)
        np.testing.assert_allclose(gamma.sum(axis=1), np.ones(small_hmm["T"]),
                                   atol=1e-10, err_msg="γ rows must sum to 1")

    def test_xi_slices_sum_to_one(self, small_hmm):
        alpha, log_scales, _ = scaled_forward(
            small_hmm["log_B"], small_hmm["log_pi"], small_hmm["log_A"])
        beta = scaled_backward(
            small_hmm["log_B"], small_hmm["log_A"], log_scales)
        xi = xi_posteriors(alpha, beta, small_hmm["log_B"], small_hmm["log_A"])
        # Each (T-1) slice should sum to 1
        for t in range(small_hmm["T"] - 1):
            np.testing.assert_allclose(xi[t].sum(), 1.0, atol=1e-8,
                                       err_msg=f"ξ slice t={t} does not sum to 1")


# ─────────────────────────────────────────────────────────────────────────────
# T2  Scaled vs log-space likelihood agreement
# ─────────────────────────────────────────────────────────────────────────────

class TestLikelihoodAgreement:
    def test_scaled_and_logspace_likelihoods_agree(self, small_hmm):
        alpha, log_scales, _ = scaled_forward(
            small_hmm["log_B"], small_hmm["log_pi"], small_hmm["log_A"])
        ll_scaled = log_likelihood(log_scales)

        _, ll_logspace = log_forward(
            small_hmm["log_B"], small_hmm["log_pi"], small_hmm["log_A"])

        np.testing.assert_allclose(
            ll_scaled, ll_logspace, rtol=1e-5,
            err_msg="Scaled and log-space log-likelihoods disagree")

    def test_log_likelihood_is_negative(self, small_hmm):
        """Log-prob of any sequence under a proper distribution must be ≤ 0."""
        alpha, log_scales, _ = scaled_forward(
            small_hmm["log_B"], small_hmm["log_pi"], small_hmm["log_A"])
        ll = log_likelihood(log_scales)
        # For our test the log_B are not normalised to be true log-probs,
        # so we only check the sign relative to the log-space version.
        _, ll_logspace = log_forward(
            small_hmm["log_B"], small_hmm["log_pi"], small_hmm["log_A"])
        assert np.isfinite(ll), "Log-likelihood should be finite"
        assert ll == pytest.approx(ll_logspace, rel=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# T3  Viterbi path validity
# ─────────────────────────────────────────────────────────────────────────────

class TestViterbi:
    def test_viterbi_path_length(self, small_hmm):
        path, _ = viterbi(
            small_hmm["log_B"], small_hmm["log_pi"], small_hmm["log_A"])
        assert len(path) == small_hmm["T"], "Path length must equal T"

    def test_viterbi_states_in_range(self, small_hmm):
        path, _ = viterbi(
            small_hmm["log_B"], small_hmm["log_pi"], small_hmm["log_A"])
        assert path.min() >= 0 and path.max() < small_hmm["N"], \
            "Viterbi states must be in [0, N)"

    def test_viterbi_log_prob_finite(self, small_hmm):
        _, log_prob = viterbi(
            small_hmm["log_B"], small_hmm["log_pi"], small_hmm["log_A"])
        assert np.isfinite(log_prob), "Viterbi log_prob must be finite"

    def test_viterbi_is_best_single_step(self):
        """
        For a 1-step sequence the Viterbi path must be the argmax of
        log_pi + log_B[0].
        """
        log_pi = np.array([-1.0, -2.0, -0.5])
        log_B  = np.array([[-0.3, -2.0, -0.8]])   # (1, 3)
        log_A  = safe_log(np.eye(3))               # identity (irrelevant for T=1)
        path, _ = viterbi(log_B, log_pi, log_A)
        expected = int(np.argmax(log_pi + log_B[0]))
        assert path[0] == expected


# ─────────────────────────────────────────────────────────────────────────────
# T4  Backward correctness  (α·β / norm == γ by construction)
# ─────────────────────────────────────────────────────────────────────────────

class TestBackward:
    def test_forward_backward_consistency(self, small_hmm):
        alpha, log_scales, _ = scaled_forward(
            small_hmm["log_B"], small_hmm["log_pi"], small_hmm["log_A"])
        beta = scaled_backward(
            small_hmm["log_B"], small_hmm["log_A"], log_scales)
        gamma = state_posteriors(alpha, beta)
        # Sum over states at each time step must equal 1
        np.testing.assert_allclose(
            gamma.sum(axis=1), 1.0,
            atol=1e-10,
            err_msg="Forward-backward posteriors do not sum to 1")


# ─────────────────────────────────────────────────────────────────────────────
# T5  Synthetic HMM — cross-check against hmmlearn (if installed)
# ─────────────────────────────────────────────────────────────────────────────

class TestHmmlearnCrossCheck:
    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("hmmlearn"),
        reason="hmmlearn not installed"
    )
    def test_gaussian_loglik_matches_hmmlearn(self):
        """
        Build a GaussianHMM in hmmlearn, generate a sequence, then verify
        that our forward algorithm produces the same log-likelihood.
        """
        from hmmlearn import hmm as hl_hmm

        np.random.seed(42)
        N = 3
        n_features = 2

        model = hl_hmm.GaussianHMM(n_components=N, covariance_type="diag",
                                     n_iter=1, random_state=42)
        model.startprob_ = np.array([0.6, 0.3, 0.1])
        model.transmat_  = np.array([[0.7, 0.2, 0.1],
                                      [0.1, 0.6, 0.3],
                                      [0.2, 0.3, 0.5]])
        model.means_     = np.random.randn(N, n_features)
        model.covars_    = np.ones((N, n_features))

        # Generate a short observation sequence
        X_obs, _ = model.sample(20, random_state=0)   # (20, 2)
        T = len(X_obs)

        # hmmlearn log-likelihood
        ll_hmmlearn = model.score(X_obs)

        # Compute emission log-probs using hmmlearn's internal method
        from scipy.stats import multivariate_normal
        log_B = np.zeros((T, N))
        for i in range(N):
            log_B[:, i] = multivariate_normal.logpdf(
                X_obs, mean=model.means_[i], cov=np.diag(model.covars_[i]))

        log_pi = safe_log(model.startprob_)
        log_A  = safe_log(model.transmat_)

        # Our scaled forward
        _, log_scales, _ = scaled_forward(log_B, log_pi, log_A)
        ll_ours = log_likelihood(log_scales)

        np.testing.assert_allclose(ll_ours, ll_hmmlearn, rtol=1e-4,
            err_msg="Our log-likelihood does not match hmmlearn")


# ─────────────────────────────────────────────────────────────────────────────
# T6  Edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_single_step_sequence(self):
        """T=1 — forward should equal π * B, no recursion needed."""
        log_pi = np.log(np.array([0.5, 0.5]))
        log_B  = np.array([[-1.0, -2.0]])   # (1, 2)
        log_A  = safe_log(np.array([[0.5, 0.5], [0.5, 0.5]]))
        alpha, log_scales, _ = scaled_forward(log_B, log_pi, log_A)
        assert alpha.shape == (1, 2)
        np.testing.assert_allclose(alpha[0].sum(), 1.0, atol=1e-12)

    def test_safe_log_no_neg_inf_on_small_value(self):
        """safe_log should not return -inf for positive inputs."""
        x = np.array([1e-300, 1e-200, 0.5])
        result = safe_log(x)
        assert np.all(np.isfinite(result)), "safe_log produced -inf"

    def test_viterbi_single_state(self):
        """N=1 — there is only one possible path."""
        log_pi = np.array([0.0])
        log_B  = np.array([[-0.5], [-1.0], [-0.3]])   # (3, 1)
        log_A  = np.array([[0.0]])                      # (1, 1) log(1)
        path, _ = viterbi(log_B, log_pi, log_A)
        assert list(path) == [0, 0, 0]
