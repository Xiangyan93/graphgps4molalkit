"""Integration tests for graphgps_cv and graphgps_optuna CLI functions.

These tests run the real CLI on a small fake CSV dataset.
They require torch, torch_geometric, pytorch_lightning, rdkit, mgktools, etc.
"""

import os
import csv
import shutil
import tempfile
import pytest

torch = pytest.importorskip("torch")

# A minimal set of SMILES and regression targets
_SMILES = [
    "CCO",
    "CCCO",
    "CC(=O)O",
    "c1ccccc1",
    "CC(C)O",
    "CCN",
    "CC=O",
    "CCCC",
    "CC(=O)N",
    "c1ccc(O)cc1",
]
_TARGETS = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

_BINARY_TARGETS = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]


_FEATURES_A = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
_FEATURES_B = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]


def _write_csv(path, smiles, targets, target_col="target",
               features=None, feature_cols=None):
    """Write a minimal CSV with smiles, target, and optional feature columns."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["smiles", target_col]
        if feature_cols:
            header.extend(feature_cols)
        writer.writerow(header)
        for i, (s, t) in enumerate(zip(smiles, targets)):
            row = [s, t]
            if features:
                for feat_list in features:
                    row.append(feat_list[i])
            writer.writerow(row)


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp(prefix="graphgps_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def regression_csv(tmp_dir):
    path = os.path.join(tmp_dir, "data.csv")
    _write_csv(path, _SMILES, _TARGETS)
    return path


@pytest.fixture
def binary_csv(tmp_dir):
    path = os.path.join(tmp_dir, "data_bin.csv")
    _write_csv(path, _SMILES, _BINARY_TARGETS)
    return path


@pytest.fixture
def regression_features_csv(tmp_dir):
    path = os.path.join(tmp_dir, "data_feat.csv")
    _write_csv(path, _SMILES, _TARGETS,
               features=[_FEATURES_A, _FEATURES_B],
               feature_cols=["feat_a", "feat_b"])
    return path


@pytest.fixture
def ext_test_features_csv(tmp_dir):
    path = os.path.join(tmp_dir, "test_feat.csv")
    _write_csv(path, _SMILES[:3], _TARGETS[:3],
               features=[_FEATURES_A[:3], _FEATURES_B[:3]],
               feature_cols=["feat_a", "feat_b"])
    return path


@pytest.fixture
def ext_test_csv(tmp_dir):
    """A separate small test set."""
    path = os.path.join(tmp_dir, "test.csv")
    _write_csv(path, _SMILES[:3], _TARGETS[:3])
    return path


class TestGraphGPSCVIntegration:
    """Integration tests for graphgps_cv with real model and small data."""

    def test_kfold_regression(self, tmp_dir, regression_csv):
        from graphgps.optuna.cross_validation import graphgps_cv

        save_dir = os.path.join(tmp_dir, "kfold_out")
        graphgps_cv([
            "--data_path", regression_csv,
            "--smiles_columns", "smiles",
            "--targets_columns", "target",
            "--task_type", "regression",
            "--metric", "rmse",
            "--cross_validation", "kFold",
            "--n_splits", "2",
            "--num_folds", "1",
            "--ensemble_size", "1",
            "--n_jobs", "1",
            "--save_dir", save_dir,
        ])

        assert os.path.isdir(save_dir)
        assert os.path.exists(os.path.join(save_dir, "kFold_metrics.csv"))

    def test_monte_carlo_regression(self, tmp_dir, regression_csv):
        from graphgps.optuna.cross_validation import graphgps_cv

        save_dir = os.path.join(tmp_dir, "mc_out")
        graphgps_cv([
            "--data_path", regression_csv,
            "--smiles_columns", "smiles",
            "--targets_columns", "target",
            "--task_type", "regression",
            "--metric", "rmse",
            "--cross_validation", "Monte-Carlo",
            "--split_type", "random",
            "--split_sizes", "0.8", "0.2",
            "--num_folds", "1",
            "--ensemble_size", "1",
            "--n_jobs", "1",
            "--save_dir", save_dir,
        ])

        assert os.path.exists(os.path.join(save_dir, "Monte-Carlo_metrics.csv"))

    def test_external_test_regression(self, tmp_dir, regression_csv, ext_test_csv):
        from graphgps.optuna.cross_validation import graphgps_cv

        save_dir = os.path.join(tmp_dir, "ext_out")
        graphgps_cv([
            "--data_path", regression_csv,
            "--smiles_columns", "smiles",
            "--targets_columns", "target",
            "--task_type", "regression",
            "--metric", "rmse",
            "--ensemble_size", "1",
            "--n_jobs", "1",
            "--save_dir", save_dir,
            "--separate_test_path", ext_test_csv,
        ])

        assert os.path.exists(os.path.join(save_dir, "test_ext_prediction.csv"))
        assert os.path.exists(os.path.join(save_dir, "test_ext_metrics.csv"))

    def test_kfold_regression_with_features(self, tmp_dir, regression_features_csv):
        from graphgps.optuna.cross_validation import graphgps_cv

        save_dir = os.path.join(tmp_dir, "kfold_feat_out")
        graphgps_cv([
            "--data_path", regression_features_csv,
            "--smiles_columns", "smiles",
            "--targets_columns", "target",
            "--features_columns", "feat_a", "feat_b",
            "--task_type", "regression",
            "--metric", "rmse",
            "--cross_validation", "kFold",
            "--n_splits", "2",
            "--num_folds", "1",
            "--ensemble_size", "1",
            "--n_jobs", "1",
            "--save_dir", save_dir,
        ])

        assert os.path.isdir(save_dir)
        assert os.path.exists(os.path.join(save_dir, "kFold_metrics.csv"))

    def test_external_test_with_features(self, tmp_dir, regression_features_csv,
                                         ext_test_features_csv):
        from graphgps.optuna.cross_validation import graphgps_cv

        save_dir = os.path.join(tmp_dir, "ext_feat_out")
        graphgps_cv([
            "--data_path", regression_features_csv,
            "--smiles_columns", "smiles",
            "--targets_columns", "target",
            "--features_columns", "feat_a", "feat_b",
            "--task_type", "regression",
            "--metric", "rmse",
            "--ensemble_size", "1",
            "--n_jobs", "1",
            "--save_dir", save_dir,
            "--separate_test_path", ext_test_features_csv,
        ])

        assert os.path.exists(os.path.join(save_dir, "test_ext_prediction.csv"))
        assert os.path.exists(os.path.join(save_dir, "test_ext_metrics.csv"))

    def test_binary_kfold(self, tmp_dir, binary_csv):
        from graphgps.optuna.cross_validation import graphgps_cv

        save_dir = os.path.join(tmp_dir, "bin_out")
        graphgps_cv([
            "--data_path", binary_csv,
            "--smiles_columns", "smiles",
            "--targets_columns", "target",
            "--task_type", "binary",
            "--metric", "roc_auc",
            "--cross_validation", "kFold",
            "--n_splits", "2",
            "--num_folds", "1",
            "--ensemble_size", "1",
            "--n_jobs", "1",
            "--save_dir", save_dir,
        ])

        assert os.path.exists(os.path.join(save_dir, "kFold_metrics.csv"))


class TestGraphGPSOptunaIntegration:
    """Integration tests for graphgps_optuna with real model and small data."""

    def test_optuna_single_trial_regression(self, tmp_dir, regression_csv):
        from graphgps.optuna.optuna import graphgps_optuna

        save_dir = os.path.join(tmp_dir, "optuna_out")
        graphgps_optuna([
            "--data_path", regression_csv,
            "--smiles_columns", "smiles",
            "--targets_columns", "target",
            "--task_type", "regression",
            "--metric", "rmse",
            "--cross_validation", "kFold",
            "--n_splits", "2",
            "--num_folds", "1",
            "--n_jobs", "1",
            "--save_dir", save_dir,
            "--n_trials", "1",
        ])

        assert os.path.exists(os.path.join(save_dir, "optuna.db"))
        assert os.path.isdir(os.path.join(save_dir, "trial-0"))

    def test_optuna_resume_skips_done(self, tmp_dir, regression_csv):
        """Run 1 trial, then call again with n_trials=1 — should not run more."""
        from graphgps.optuna.optuna import graphgps_optuna

        save_dir = os.path.join(tmp_dir, "optuna_resume")
        common_args = [
            "--data_path", regression_csv,
            "--smiles_columns", "smiles",
            "--targets_columns", "target",
            "--task_type", "regression",
            "--metric", "rmse",
            "--cross_validation", "kFold",
            "--n_splits", "2",
            "--num_folds", "1",
            "--n_jobs", "1",
            "--save_dir", save_dir,
            "--n_trials", "1",
        ]
        graphgps_optuna(common_args)
        # Second call: n_trials=1, 1 already done => 0 to run
        graphgps_optuna(common_args)

        # Still only trial-0 directory (no trial-1)
        assert os.path.isdir(os.path.join(save_dir, "trial-0"))
        assert not os.path.isdir(os.path.join(save_dir, "trial-1"))
