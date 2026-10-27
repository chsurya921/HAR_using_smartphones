"""
data_loader.py
==============
Leakage-safe data loading for the UCI HAR dataset.

Key guarantees
--------------
* Subject-independent splits: train / validation / test subjects are disjoint.
* Sequences per subject are built in genuine temporal order (no shuffling).
* Scalers and PCA are NEVER fit on validation or test data (enforced here by
  only exposing raw arrays; fitting happens in preprocessing.py).
* Zero-subject-overlap is verified by assertions that are also exercised in
  tests/test_preprocessing.py.

Usage
-----
    from src.data_loader import HARDataLoader
    loader = HARDataLoader(config_path="config/experiment_config.yaml")
    splits = loader.load_all()
    # splits.train_X, splits.train_y, splits.train_subjects, ...
    # splits.sequences_train  -> list of (X_seq, y_seq) per subject, in order
"""

from __future__ import annotations

import os
import numpy as np
import yaml
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SubjectSequence:
    """All windows belonging to one subject, in temporal order."""
    subject_id: int
    X: np.ndarray          # shape (n_windows, n_features)
    y: np.ndarray          # shape (n_windows,)   integer labels 1-6


@dataclass
class DataSplits:
    """Holds the three leakage-safe splits and their per-subject sequences."""
    # ── flat arrays ──────────────────────────────────────────────────────────
    train_X: np.ndarray
    train_y: np.ndarray
    train_subjects: np.ndarray

    val_X: np.ndarray
    val_y: np.ndarray
    val_subjects: np.ndarray

    test_X: np.ndarray
    test_y: np.ndarray
    test_subjects: np.ndarray

    # ── per-subject sequences (temporal order preserved) ─────────────────────
    sequences_train: List[SubjectSequence] = field(default_factory=list)
    sequences_val: List[SubjectSequence] = field(default_factory=list)
    sequences_test: List[SubjectSequence] = field(default_factory=list)

    # ── metadata ─────────────────────────────────────────────────────────────
    feature_names: List[str] = field(default_factory=list)
    activity_labels: dict = field(default_factory=dict)   # {1: "WALKING", ...}


# ─────────────────────────────────────────────────────────────────────────────
# Loader class
# ─────────────────────────────────────────────────────────────────────────────

class HARDataLoader:
    """
    Load the UCI HAR dataset and partition it into subject-independent splits.

    Parameters
    ----------
    config_path : str
        Path to experiment_config.yaml.
    base_dir : str, optional
        Directory that contains the config file; all relative paths in the
        config are resolved relative to this directory.
    """

    ACTIVITY_NAMES = {
        1: "WALKING",
        2: "WALKING_UPSTAIRS",
        3: "WALKING_DOWNSTAIRS",
        4: "SITTING",
        5: "STANDING",
        6: "LAYING",
    }

    def __init__(self, config_path: str = "config/experiment_config.yaml",
                 base_dir: Optional[str] = None):
        if base_dir is None:
            base_dir = os.path.dirname(os.path.abspath(config_path))
        self.base_dir = base_dir

        with open(config_path, "r") as fh:
            self.config = yaml.safe_load(fh)

        self._paths = self.config["paths"]
        self._split_cfg = self.config["splits"]

    # ── public API ────────────────────────────────────────────────────────────

    def load_all(self) -> DataSplits:
        """Load data files and return a fully-populated DataSplits object."""
        print("[DataLoader] Loading raw data files …")
        X_train_raw, y_train_raw, s_train_raw = self._load_split("train")
        X_test_raw, y_test_raw, s_test_raw = self._load_split("test")
        feature_names = self._load_feature_names()

        # Merge raw train + test back into one pool, then re-split by subject.
        # This lets us use the config-defined val_subject_ids from the
        # official training set while still respecting the fixed test IDs.
        X_all = np.vstack([X_train_raw, X_test_raw])
        y_all = np.concatenate([y_train_raw, y_test_raw])
        s_all = np.concatenate([s_train_raw, s_test_raw])

        print(f"[DataLoader] Total samples: {len(y_all)}  |  "
              f"Subjects: {sorted(np.unique(s_all))}")

        train_ids = set(self._split_cfg["train_subject_ids"])
        val_ids   = set(self._split_cfg["val_subject_ids"])
        test_ids  = set(self._split_cfg["test_subject_ids"])

        # Validation subjects are a subset of the official training subjects
        # Remove them from training so they are truly held-out.
        train_only_ids = train_ids - val_ids

        self._assert_no_overlap(train_only_ids, val_ids, test_ids)

        splits = DataSplits(
            train_X=self._select(X_all, s_all, train_only_ids),
            train_y=self._select(y_all, s_all, train_only_ids),
            train_subjects=self._select(s_all, s_all, train_only_ids),
            val_X=self._select(X_all, s_all, val_ids),
            val_y=self._select(y_all, s_all, val_ids),
            val_subjects=self._select(s_all, s_all, val_ids),
            test_X=self._select(X_all, s_all, test_ids),
            test_y=self._select(y_all, s_all, test_ids),
            test_subjects=self._select(s_all, s_all, test_ids),
            feature_names=feature_names,
            activity_labels=dict(self.ACTIVITY_NAMES),
        )

        # Build per-subject sequences in temporal order
        splits.sequences_train = self._build_sequences(
            splits.train_X, splits.train_y, splits.train_subjects)
        splits.sequences_val = self._build_sequences(
            splits.val_X, splits.val_y, splits.val_subjects)
        splits.sequences_test = self._build_sequences(
            splits.test_X, splits.test_y, splits.test_subjects)

        self._print_summary(splits)
        return splits

    # ── private helpers ───────────────────────────────────────────────────────

    def _resolve(self, rel_path: str) -> str:
        """Resolve a path relative to the project base directory."""
        if os.path.isabs(rel_path):
            return rel_path
        return os.path.normpath(os.path.join(self.base_dir, rel_path))

    def _load_split(self, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load X, y, subjects for 'train' or 'test'."""
        key_X = f"{split}_X"
        key_y = f"{split}_y"
        key_s = f"{split}_subjects"

        X = np.loadtxt(self._resolve(self._paths[key_X]))
        y = np.loadtxt(self._resolve(self._paths[key_y]), dtype=int).ravel()
        s = np.loadtxt(self._resolve(self._paths[key_s]), dtype=int).ravel()

        assert X.shape[0] == len(y) == len(s), (
            f"Shape mismatch in {split}: X={X.shape[0]}, y={len(y)}, s={len(s)}"
        )
        print(f"[DataLoader]   {split}: X{X.shape}, labels={np.unique(y)}, "
              f"subjects={sorted(np.unique(s))}")
        return X, y, s

    def _load_feature_names(self) -> List[str]:
        """Load feature names from features.txt, de-duplicating if needed."""
        path = self._resolve(self._paths["features"])
        names = []
        with open(path, "r") as fh:
            for line in fh:
                parts = line.strip().split(None, 1)
                if len(parts) == 2:
                    names.append(parts[1])
        # UCI HAR has a handful of duplicate feature names — make unique.
        seen: dict = {}
        unique_names = []
        for n in names:
            if n in seen:
                seen[n] += 1
                unique_names.append(f"{n}_dup{seen[n]}")
            else:
                seen[n] = 0
                unique_names.append(n)
        return unique_names

    @staticmethod
    def _select(arr: np.ndarray, subjects: np.ndarray,
                ids: set) -> np.ndarray:
        """Boolean-index rows whose subject ID is in `ids`."""
        mask = np.isin(subjects, list(ids))
        return arr[mask]

    @staticmethod
    def _assert_no_overlap(train: set, val: set, test: set) -> None:
        """Hard assertion: no subject appears in more than one split."""
        tv = train & val
        tt = train & test
        vt = val & test
        assert not tv, f"Subject overlap train∩val: {tv}"
        assert not tt, f"Subject overlap train∩test: {tt}"
        assert not vt, f"Subject overlap val∩test: {vt}"

    @staticmethod
    def _build_sequences(X: np.ndarray, y: np.ndarray,
                          subjects: np.ndarray) -> List[SubjectSequence]:
        """
        Build one SubjectSequence per subject.

        Row order within each subject is preserved exactly as it appears in
        the raw files (which follows the UCI HAR temporal ordering guarantee).
        No shuffling is applied.
        """
        sequences = []
        for sid in sorted(np.unique(subjects)):
            mask = subjects == sid
            sequences.append(SubjectSequence(
                subject_id=int(sid),
                X=X[mask],
                y=y[mask],
            ))
        return sequences

    @staticmethod
    def _print_summary(splits: DataSplits) -> None:
        print("\n[DataLoader] ── Split Summary ──────────────────────────────")
        for name, X, y, seqs in [
            ("Train",      splits.train_X, splits.train_y, splits.sequences_train),
            ("Validation", splits.val_X,   splits.val_y,   splits.sequences_val),
            ("Test",       splits.test_X,  splits.test_y,  splits.sequences_test),
        ]:
            unique_s = sorted({seq.subject_id for seq in seqs})
            class_counts = {c: int((y == c).sum()) for c in sorted(np.unique(y))}
            print(f"  {name:12s}: {X.shape[0]:5d} samples | "
                  f"{X.shape[1]} features | "
                  f"subjects={unique_s} | "
                  f"class_counts={class_counts}")
        print("[DataLoader] ── Overlap check: PASS ─────────────────────────\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI convenience
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    cfg = sys.argv[1] if len(sys.argv) > 1 else "config/experiment_config.yaml"
    loader = HARDataLoader(config_path=cfg, base_dir=".")
    splits = loader.load_all()
    print(f"Feature names (first 5): {splits.feature_names[:5]}")
    print(f"Activity labels: {splits.activity_labels}")
