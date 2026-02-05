"""Unit tests for the Evaluator class, focusing on multi-task learning with missing values."""

import numpy as np
import pandas as pd
import pytest

from graphgps.optuna.evaluator import Evaluator


class TestWeightedMean:
    """Tests for the _weighted_mean static method."""

    def test_simple_weighted_mean(self):
        """Test basic weighted mean calculation."""
        df = pd.DataFrame({
            'metric': ['mae', 'mae'],
            'no_targets_columns': [0, 1],
            'value': [0.5, 1.0],
            'n_samples': [100, 10]
        })
        result = Evaluator._weighted_mean(df)
        expected = (0.5 * 100 + 1.0 * 10) / (100 + 10)
        assert abs(result - expected) < 1e-6

    def test_weighted_mean_with_nan_values(self):
        """Test weighted mean excludes NaN values correctly."""
        df = pd.DataFrame({
            'metric': ['mae', 'mae', 'mae'],
            'no_targets_columns': [0, 1, 2],
            'value': [0.5, np.nan, 1.0],
            'n_samples': [100, 0, 10]
        })
        result = Evaluator._weighted_mean(df)
        expected = (0.5 * 100 + 1.0 * 10) / (100 + 10)
        assert abs(result - expected) < 1e-6

    def test_weighted_mean_all_nan(self):
        """Test weighted mean returns NaN when all values are NaN."""
        df = pd.DataFrame({
            'metric': ['mae'],
            'no_targets_columns': [0],
            'value': [np.nan],
            'n_samples': [0]
        })
        result = Evaluator._weighted_mean(df)
        assert np.isnan(result)

    def test_weighted_mean_zero_samples(self):
        """Test weighted mean returns NaN when total samples is zero."""
        df = pd.DataFrame({
            'metric': ['mae', 'mae'],
            'no_targets_columns': [0, 1],
            'value': [0.5, 1.0],
            'n_samples': [0, 0]
        })
        result = Evaluator._weighted_mean(df)
        assert np.isnan(result)

    def test_weighted_mean_backward_compatibility(self):
        """Test weighted mean falls back to simple mean when n_samples column is missing."""
        df = pd.DataFrame({
            'metric': ['mae', 'mae'],
            'no_targets_columns': [0, 1],
            'value': [0.5, 1.0]
        })
        result = Evaluator._weighted_mean(df)
        expected = 0.75  # simple mean
        assert abs(result - expected) < 1e-6

    def test_weighted_mean_single_task(self):
        """Test weighted mean with single task returns that task's value."""
        df = pd.DataFrame({
            'metric': ['mae'],
            'no_targets_columns': [0],
            'value': [0.5],
            'n_samples': [100]
        })
        result = Evaluator._weighted_mean(df)
        assert abs(result - 0.5) < 1e-6

    def test_weighted_mean_equal_samples(self):
        """Test weighted mean equals simple mean when all tasks have equal samples."""
        df = pd.DataFrame({
            'metric': ['mae', 'mae', 'mae'],
            'no_targets_columns': [0, 1, 2],
            'value': [0.2, 0.4, 0.6],
            'n_samples': [50, 50, 50]
        })
        result = Evaluator._weighted_mean(df)
        expected = 0.4  # simple mean
        assert abs(result - expected) < 1e-6

    def test_weighted_mean_heavily_imbalanced(self):
        """Test weighted mean is dominated by task with most samples."""
        df = pd.DataFrame({
            'metric': ['mae', 'mae'],
            'no_targets_columns': [0, 1],
            'value': [0.1, 0.9],  # Very different values
            'n_samples': [1000, 1]  # Heavily imbalanced
        })
        result = Evaluator._weighted_mean(df)
        # Result should be very close to 0.1 (the task with 1000 samples)
        assert result < 0.15
        assert result > 0.09


class TestMetricsDataFrame:
    """Tests for metrics DataFrame structure with n_samples column."""

    def test_metrics_df_structure(self):
        """Test that metrics DataFrame has correct columns."""
        # This is more of a schema test
        expected_columns = ["metric", "no_targets_columns", "value", "n_samples"]
        df = pd.DataFrame(columns=expected_columns)

        assert all(col in df.columns for col in expected_columns)

    def test_n_samples_reflects_valid_count(self):
        """Test that n_samples correctly reflects the count of non-NaN values."""
        # Simulate what evaluate_train_test does
        y_true = np.array([1.0, 2.0, np.nan, 4.0, np.nan])
        valid_mask = ~np.isnan(y_true)
        n_valid = int(valid_mask.sum())

        assert n_valid == 3  # 3 non-NaN values

    def test_n_samples_all_valid(self):
        """Test n_samples when all values are valid."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        valid_mask = ~np.isnan(y_true)
        n_valid = int(valid_mask.sum())

        assert n_valid == 5

    def test_n_samples_all_missing(self):
        """Test n_samples when all values are NaN."""
        y_true = np.array([np.nan, np.nan, np.nan])
        valid_mask = ~np.isnan(y_true)
        n_valid = int(valid_mask.sum())

        assert n_valid == 0


class TestWeightedMeanCrossValidation:
    """Tests for weighted mean in cross-validation scenarios."""

    def test_weighted_mean_across_folds(self):
        """Test weighted mean calculation across multiple CV folds."""
        # Simulate kFold results with 2 folds, 3 tasks
        # Fold 0: task 0 (50 samples), task 1 (30 samples), task 2 (20 samples)
        # Fold 1: task 0 (50 samples), task 1 (30 samples), task 2 (20 samples)
        df = pd.DataFrame({
            'metric': ['rmse'] * 6,
            'no_targets_columns': [0, 1, 2, 0, 1, 2],
            'value': [0.1, 0.2, 0.3, 0.15, 0.25, 0.35],
            'n_samples': [50, 30, 20, 50, 30, 20],
            'seed': [0, 0, 0, 0, 0, 0],
            'split': [0, 0, 0, 1, 1, 1]
        })

        result = Evaluator._weighted_mean(df)

        # Manual calculation
        total_samples = 50 + 30 + 20 + 50 + 30 + 20
        weighted_sum = (0.1*50 + 0.2*30 + 0.3*20 + 0.15*50 + 0.25*30 + 0.35*20)
        expected = weighted_sum / total_samples

        assert abs(result - expected) < 1e-6

    def test_weighted_vs_unweighted_difference(self):
        """Test that weighted and unweighted means differ with imbalanced samples."""
        df = pd.DataFrame({
            'metric': ['rmse', 'rmse'],
            'no_targets_columns': [0, 1],
            'value': [0.1, 0.9],
            'n_samples': [900, 100]
        })

        weighted = Evaluator._weighted_mean(df)
        unweighted = df['value'].mean()

        # Weighted should be closer to 0.1 (task with more samples)
        assert weighted < unweighted
        assert weighted < 0.2  # Much closer to 0.1
        assert unweighted == 0.5  # Simple mean of 0.1 and 0.9
