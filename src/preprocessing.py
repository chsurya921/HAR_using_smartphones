"""
preprocessing.py
================
Leakage-safe feature scaling and PCA dimensionality reduction.

Rules enforced here
-------------------
* StandardScaler and PCA are fit ONLY on training data.
* The same fitted objects are then applied (transform only) to
  validation and test data.
* Inverse-transform utilities are provided for interpretability.

Usage
-----
    from src.preprocessing import Preprocessor
    prep = Preprocessor(config_path="config/experiment_config.yaml")
    X_tr, X_val, X_te = prep.fit_transform(X_train, X_val, X_test)
    # prep.scaler, prep.pca are now fitted and reusable.
"""

from __future__ import annotations

import numpy as np
import yaml
from typing import Optional, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class Preprocessor:
    """
    Fit a StandardScaler (and optionally PCA) on training data, then
    apply the same transform to validation and test data.

    Parameters
    ----------
    config_path : str
        Path to experiment_config.yaml.
    n_components : int, optional
        Override the PCA n_components from config.
    """

    def __init__(self, config_path: str = "config/experiment_config.yaml",
                 n_components: Optional[int] = None):
        with open(config_path, "r") as fh:
            cfg = yaml.safe_load(fh)

        self._scaler_type = cfg["preprocessing"]["scaler"]
        pca_cfg = cfg["preprocessing"]["pca"]
        self._pca_enabled = pca_cfg["enabled"]
        self._n_components = n_components if n_components is not None \
            else pca_cfg["n_components"]
        self._pca_whiten = pca_cfg["whiten"]

        # Fitted objects (None until fit_transform is called)
        self.scaler: Optional[StandardScaler] = None
        self.pca: Optional[PCA] = None

    # ── public API ────────────────────────────────────────────────────────────

    def fit_transform(
        self,
        X_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        X_test: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, ...]:
        """
        Fit on X_train, then transform all provided arrays.

        Returns
        -------
        Tuple of transformed arrays in the same order as inputs.
        Always returns at least (X_train_transformed,).
        """
        # ── Step 1: StandardScaler (fit on train only) ────────────────────
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)

        result = [X_train_scaled]
        if X_val is not None:
            result.append(self.scaler.transform(X_val))
        if X_test is not None:
            result.append(self.scaler.transform(X_test))

        # ── Step 2: PCA (fit on train only) ───────────────────────────────
        if self._pca_enabled:
            self.pca = PCA(
                n_components=self._n_components,
                whiten=self._pca_whiten,
                random_state=42,
            )
            result[0] = self.pca.fit_transform(result[0])
            if X_val is not None:
                result[1] = self.pca.transform(result[1])
            if X_test is not None:
                result[2] = self.pca.transform(result[2])

            explained = self.pca.explained_variance_ratio_.cumsum()[-1]
            print(f"[Preprocessor] PCA: {self._n_components} components → "
                  f"{explained*100:.1f}% variance explained")

        return tuple(result)

    def transform_sequences(self, sequences: list) -> list:
        """
        Apply the fitted scaler (and PCA) to a list of SubjectSequence objects.

        Parameters
        ----------
        sequences : list of SubjectSequence
            Must be called after fit_transform.

        Returns
        -------
        New list of SubjectSequence with transformed X arrays.
        """
        try:
            from src.data_loader import SubjectSequence
        except ImportError:
            from data_loader import SubjectSequence

        if self.scaler is None:
            raise RuntimeError("Call fit_transform before transform_sequences.")
        out = []
        for seq in sequences:
            X_scaled = self.scaler.transform(seq.X)
            X_out = self.pca.transform(X_scaled) if self.pca is not None else X_scaled
            out.append(SubjectSequence(
                subject_id=seq.subject_id,
                X=X_out,
                y=seq.y,
            ))
        return out

    # ── utility ───────────────────────────────────────────────────────────────

    @property
    def n_features_out(self) -> int:
        """Dimensionality of the transformed feature space."""
        if self.pca is not None:
            return self.pca.n_components_
        if self.scaler is not None:
            return self.scaler.n_features_in_
        raise RuntimeError("Preprocessor has not been fitted yet.")

    def explained_variance_table(self) -> Optional[np.ndarray]:
        """Cumulative explained variance ratios from PCA, or None."""
        if self.pca is None:
            return None
        return self.pca.explained_variance_ratio_.cumsum()
