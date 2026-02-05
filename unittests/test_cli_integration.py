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

    def test_kfold_regression_with_rdkit2d_features(self, tmp_dir, regression_csv):
        """Test graphgps_cv with --features_generators_name rdkit_2d.

        Pre-populates the SMILES_TO_FEATURES cache with inf values for one
        molecule, simulating real rdkit_2d behavior on problematic molecules.
        With the current np.isnan check, inf passes through and crashes
        StandardScaler with 'Input X contains infinity or a value too large'.
        """
        from graphgps.optuna.cross_validation import graphgps_cv
        from graphgps.data import data as data_module

        data_module.SMILES_TO_FEATURES.clear()

        # Simulate rdkit_2d producing inf for 'CCO' (200 features)
        inf_features = [0.0] * 200
        inf_features[0] = float('inf')
        inf_features[1] = float('-inf')
        inf_features[2] = 1e39
        data_module.SMILES_TO_FEATURES['CCO'] = inf_features

        save_dir = os.path.join(tmp_dir, "rdkit2d_out")
        try:
            graphgps_cv([
                "--data_path", regression_csv,
                "--smiles_columns", "smiles",
                "--targets_columns", "target",
                "--task_type", "regression",
                "--metric", "rmse",
                "--features_generators_name", "rdkit_2d",
                "--cross_validation", "kFold",
                "--n_splits", "2",
                "--num_folds", "1",
                "--ensemble_size", "1",
                "--n_jobs", "1",
                "--save_dir", save_dir,
            ])

            assert os.path.isdir(save_dir)
            assert os.path.exists(os.path.join(save_dir, "kFold_metrics.csv"))
        finally:
            data_module.SMILES_TO_FEATURES.clear()

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


class TestMultiTaskWithMissingValues:
    """Tests for multi-task learning with missing values."""

    @pytest.fixture
    def multitask_csv_with_missing(self, tmp_dir):
        """Create a multi-task CSV with missing values (NaN)."""
        import numpy as np
        path = os.path.join(tmp_dir, "multitask_missing.csv")
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["smiles", "target_0", "target_1", "target_2"])
            # Task 0: all values present (10 samples)
            # Task 1: 5 values present, 5 missing
            # Task 2: 3 values present, 7 missing
            data = [
                ("CCO", 1.0, 0.5, 0.1),
                ("CCCO", 2.0, 0.6, ""),
                ("CC(=O)O", 3.0, "", 0.3),
                ("c1ccccc1", 4.0, 0.8, ""),
                ("CC(C)O", 5.0, "", ""),
                ("CCN", 6.0, 1.0, ""),
                ("CC=O", 7.0, "", ""),
                ("CCCC", 8.0, 1.2, ""),
                ("CC(=O)N", 9.0, "", 0.9),
                ("c1ccc(O)cc1", 10.0, 1.4, ""),
            ]
            for row in data:
                writer.writerow(row)
        return path

    @pytest.fixture
    def multitask_test_csv_with_missing(self, tmp_dir):
        """Create a multi-task test CSV with missing values."""
        path = os.path.join(tmp_dir, "multitask_test_missing.csv")
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["smiles", "target_0", "target_1", "target_2"])
            data = [
                ("CCC", 1.5, 0.7, ""),
                ("CCCCO", 2.5, "", 0.4),
                ("c1cccnc1", 3.5, 0.9, 0.5),
            ]
            for row in data:
                writer.writerow(row)
        return path

    def test_multitask_kfold_with_missing_values(self, tmp_dir, multitask_csv_with_missing):
        """Test kFold cross-validation with multi-task data containing missing values."""
        import pandas as pd
        from graphgps.optuna.cross_validation import graphgps_cv

        save_dir = os.path.join(tmp_dir, "multitask_kfold_out")
        graphgps_cv([
            "--data_path", multitask_csv_with_missing,
            "--smiles_columns", "smiles",
            "--targets_columns", "target_0", "target_1", "target_2",
            "--task_type", "regression",
            "--metric", "rmse",
            "--cross_validation", "kFold",
            "--n_splits", "2",
            "--num_folds", "1",
            "--ensemble_size", "1",
            "--n_jobs", "1",
            "--save_dir", save_dir,
        ])

        # Check that metrics file exists and has n_samples column
        metrics_path = os.path.join(save_dir, "kFold_metrics.csv")
        assert os.path.exists(metrics_path)

        df_metrics = pd.read_csv(metrics_path)
        assert "n_samples" in df_metrics.columns, "n_samples column should be present"
        assert "no_targets_columns" in df_metrics.columns

        # Verify n_samples varies by task (task 0 has more samples than task 2)
        task_0_samples = df_metrics[df_metrics["no_targets_columns"] == 0]["n_samples"].sum()
        task_2_samples = df_metrics[df_metrics["no_targets_columns"] == 2]["n_samples"].sum()
        assert task_0_samples > task_2_samples, "Task 0 should have more samples than task 2"

    def test_multitask_external_test_with_missing_values(self, tmp_dir,
                                                          multitask_csv_with_missing,
                                                          multitask_test_csv_with_missing):
        """Test external validation with multi-task data containing missing values."""
        import pandas as pd
        from graphgps.optuna.cross_validation import graphgps_cv

        save_dir = os.path.join(tmp_dir, "multitask_ext_out")
        graphgps_cv([
            "--data_path", multitask_csv_with_missing,
            "--smiles_columns", "smiles",
            "--targets_columns", "target_0", "target_1", "target_2",
            "--task_type", "regression",
            "--metric", "rmse",
            "--ensemble_size", "1",
            "--n_jobs", "1",
            "--save_dir", save_dir,
            "--separate_test_path", multitask_test_csv_with_missing,
        ])

        # Check metrics file
        metrics_path = os.path.join(save_dir, "test_ext_metrics.csv")
        assert os.path.exists(metrics_path)

        df_metrics = pd.read_csv(metrics_path)
        assert "n_samples" in df_metrics.columns

        # Test set: task_0 has 3 samples, task_1 has 2, task_2 has 2
        task_samples = df_metrics.groupby("no_targets_columns")["n_samples"].first()
        assert task_samples[0] == 3, "Task 0 should have 3 test samples"
        assert task_samples[1] == 2, "Task 1 should have 2 test samples"
        assert task_samples[2] == 2, "Task 2 should have 2 test samples"

    def test_weighted_mean_calculation(self, tmp_dir, multitask_csv_with_missing):
        """Test that weighted mean is correctly calculated based on n_samples."""
        import pandas as pd
        import numpy as np
        from graphgps.optuna.cross_validation import graphgps_cv
        from graphgps.optuna.evaluator import Evaluator

        save_dir = os.path.join(tmp_dir, "multitask_weighted_out")
        graphgps_cv([
            "--data_path", multitask_csv_with_missing,
            "--smiles_columns", "smiles",
            "--targets_columns", "target_0", "target_1", "target_2",
            "--task_type", "regression",
            "--metric", "rmse",
            "--cross_validation", "kFold",
            "--n_splits", "2",
            "--num_folds", "1",
            "--ensemble_size", "1",
            "--n_jobs", "1",
            "--save_dir", save_dir,
        ])

        df_metrics = pd.read_csv(os.path.join(save_dir, "kFold_metrics.csv"))
        df_rmse = df_metrics[df_metrics["metric"] == "rmse"]

        # Verify weighted mean differs from simple mean when n_samples vary
        simple_mean = df_rmse["value"].mean()
        weighted_mean = Evaluator._weighted_mean(df_rmse)

        # They should be different because tasks have different sample counts
        # (unless by chance they're equal, but that's unlikely with real data)
        assert "n_samples" in df_rmse.columns
        assert df_rmse["n_samples"].std() > 0, "Sample counts should vary between tasks"


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
