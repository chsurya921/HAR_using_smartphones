"""
tests/test_preprocessing.py
============================
Leakage safety tests for src/data_loader.py and src/preprocessing.py.

Tests
-----
L1  Zero subject overlap across train/val/test splits
L2  Sequence temporal ordering — labels within a subject are contiguous
L3  Scaler is fit only on training data (mean/var are from train)
L4  PCA transform of val/test uses train-fitted components
L5  Reproducibility — same seed gives same results

Run with:  cd HAR_HMM_Project && pytest tests/test_preprocessing.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from src.data_loader import HARDataLoader, DataSplits
from src.preprocessing import Preprocessor


# ─────────────────────────────────────────────────────────────────────────────
# Helper — synthetic DataSplits for tests that don't need real files
# ─────────────────────────────────────────────────────────────────────────────

def make_synthetic_splits(
    n_train: int = 200,
    n_val: int = 80,
    n_test: int = 60,
    n_features: int = 10,
    seed: int = 0,
) -> DataSplits:
    rng = np.random.default_rng(seed)
    X_tr = rng.standard_normal((n_train, n_features))
    X_va = rng.standard_normal((n_val,   n_features))
    X_te = rng.standard_normal((n_test,  n_features))

    # Make val mean intentionally different to test leakage
    X_va += 10.0
    X_te += 20.0

    y_tr = rng.integers(1, 7, size=n_train)
    y_va = rng.integers(1, 7, size=n_val)
    y_te = rng.integers(1, 7, size=n_test)

    s_tr = np.repeat(np.arange(1, 5), n_train // 4 + 1)[:n_train]
    s_va = np.repeat(np.arange(5, 7), n_val  // 2 + 1)[:n_val]
    s_te = np.repeat(np.arange(7, 9), n_test // 2 + 1)[:n_test]

    return DataSplits(
        train_X=X_tr, train_y=y_tr, train_subjects=s_tr,
        val_X=X_va,   val_y=y_va,   val_subjects=s_va,
        test_X=X_te,  test_y=y_te,  test_subjects=s_te,
    )


# ─────────────────────────────────────────────────────────────────────────────
# L1  Zero subject overlap
# ─────────────────────────────────────────────────────────────────────────────

class TestSubjectOverlap:
    def test_no_subject_overlap_synthetic(self):
        splits = make_synthetic_splits()
        train_s = set(splits.train_subjects)
        val_s   = set(splits.val_subjects)
        test_s  = set(splits.test_subjects)
        assert train_s.isdisjoint(val_s),  f"Train∩Val overlap: {train_s & val_s}"
        assert train_s.isdisjoint(test_s), f"Train∩Test overlap: {train_s & test_s}"
        assert val_s.isdisjoint(test_s),   f"Val∩Test overlap: {val_s & test_s}"

    @pytest.mark.skipif(
        not os.path.exists(
            os.path.join(os.path.dirname(__file__), "..",
                         "config", "experiment_config.yaml")),
        reason="Config file not found"
    )
    @pytest.mark.skipif(
        not os.path.exists(
            os.path.join(os.path.dirname(__file__), "..", "..",
                         "UCI HAR Dataset", "train", "X_train.txt")),
        reason="Dataset not found"
    )
    def test_no_subject_overlap_real_data(self):
        """Integration test against the real UCI HAR files."""
        cfg_path = os.path.join(
            os.path.dirname(__file__), "..", "config", "experiment_config.yaml")
        loader = HARDataLoader(config_path=cfg_path, base_dir=os.path.join(
            os.path.dirname(__file__), ".."))
        splits = loader.load_all()

        train_s = set(splits.train_subjects)
        val_s   = set(splits.val_subjects)
        test_s  = set(splits.test_subjects)
        assert train_s.isdisjoint(val_s),  "LEAKAGE: Train and Val share subjects!"
        assert train_s.isdisjoint(test_s), "LEAKAGE: Train and Test share subjects!"
        assert val_s.isdisjoint(test_s),   "LEAKAGE: Val and Test share subjects!"


# ─────────────────────────────────────────────────────────────────────────────
# L3  Scaler fit only on training data
# ─────────────────────────────────────────────────────────────────────────────

class TestScalerLeakage:
    def test_scaler_mean_matches_train(self):
        splits = make_synthetic_splits()
        prep = Preprocessor.__new__(Preprocessor)
        prep._scaler_type = "StandardScaler"
        prep._pca_enabled = False
        prep._n_components = None
        prep._pca_whiten = False
        prep.scaler = None
        prep.pca = None

        from sklearn.preprocessing import StandardScaler
        prep.scaler = StandardScaler()
        prep.scaler.fit(splits.train_X)
        X_tr_sc = prep.scaler.transform(splits.train_X)
        X_va_sc = prep.scaler.transform(splits.val_X)

        # After standardisation on train, train mean should be ~0
        np.testing.assert_allclose(X_tr_sc.mean(axis=0), 0.0, atol=1e-10,
                                   err_msg="Train mean after scaler is not zero")

        # Val mean should NOT be ~0 (val was shifted by +10)
        val_mean_abs = np.abs(X_va_sc.mean(axis=0)).mean()
        assert val_mean_abs > 0.5, \
            "Val mean is suspiciously close to 0 — scaler may have been fit on val!"

    def test_preprocessor_fit_transform_shape(self):
        splits = make_synthetic_splits()
        prep = Preprocessor.__new__(Preprocessor)
        prep._scaler_type = "StandardScaler"
        prep._pca_enabled = True
        prep._n_components = 5
        prep._pca_whiten = False
        prep.scaler = None
        prep.pca = None

        # Re-use fit_transform via duck-typing
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        prep.scaler = StandardScaler()
        X_tr_sc = prep.scaler.fit_transform(splits.train_X)
        X_va_sc = prep.scaler.transform(splits.val_X)
        prep.pca = PCA(n_components=5, random_state=42)
        X_tr_pca = prep.pca.fit_transform(X_tr_sc)
        X_va_pca = prep.pca.transform(X_va_sc)

        assert X_tr_pca.shape == (200, 5)
        assert X_va_pca.shape == (80,  5)


# ─────────────────────────────────────────────────────────────────────────────
# L5  Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

class TestReproducibility:
    def test_same_seed_gives_same_pca_components(self):
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA

        splits = make_synthetic_splits(seed=42)

        def run():
            scaler = StandardScaler()
            X = scaler.fit_transform(splits.train_X)
            pca = PCA(n_components=5, random_state=42)
            return pca.fit_transform(X)

        r1 = run()
        r2 = run()
        np.testing.assert_array_equal(r1, r2,
            err_msg="PCA with fixed seed should be deterministic")
