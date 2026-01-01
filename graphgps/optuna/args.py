from tap import Tap
from typing import List, Literal, Optional
import os
import copy
import pandas as pd
from sklearn.model_selection import KFold
from mgktools.data.split import data_split_index
from mgktools.features_mol.features_generators import FeaturesGenerator
from mgktools.evaluators.cross_validation import Metric
from graphgps.data.data_molalkit import get_data
CWD = os.path.dirname(__file__)


class TrainArgs(Tap):
    cfg_file: str = None
    """The configuration file path for the GPS model."""
    save_dir: str = None
    """Directory where model checkpoints will be saved."""
    n_jobs: int = 1
    """The cpu numbers used for parallel computing."""
    data_path: str = None
    """The Path of input data CSV file."""
    smiles_columns: List[str] = None
    """List of names of the columns containing SMILES strings.
    By default, uses the first :code:`number_of_molecules` columns."""
    targets_columns: List[str] = None
    """
    Name of the columns containing target values.
    """
    dataset_type: Literal["regression", "classification", "multiclass"] = None
    """
    Type of dataset.
    """
    features_generators_name: List[str] = None
    """Method(s) of generating additional features_mol."""
    cross_validation: Literal["kFold", "leave-one-out", "Monte-Carlo", "no"] = "no"
    """The way to split data for cross-validation."""
    n_splits: int = None
    """The number of fold for kFold CV."""
    split_type: Literal["random", "scaffold_order", "scaffold_random", "stratified"] = None
    """Method of splitting the data into train/test sets."""
    split_sizes: List[float] = None
    """Split proportions for train/test sets."""
    num_folds: int = 1
    """Number of folds when performing cross validation."""
    seed: int = 0
    """Random seed."""
    metric: Metric = None
    """metric"""
    extra_metrics: List[Metric] = []
    """Metrics"""
    separate_test_path: str = None
    """Path to separate test set, optional."""
    separate_val_path: str = None
    """Path to separate validation set, optional."""
    ensemble_size: int = 1
    """Number of models in the ensemble."""

    @property
    def metrics(self) -> List[Metric]:
        return [self.metric] + self.extra_metrics

    @property
    def features_generators(self) -> Optional[List[FeaturesGenerator]]:
        if self.features_generators_name is None:
            return None
        else:
            return [FeaturesGenerator(features_generator_name=fg) for fg in self.features_generators_name]
        
    def process_args(self) -> None:
        self.opts = ['wandb.use', 'False']
        if self.cfg_file is None:
            self.cfg_file = f'{CWD}/GPS_template.yaml'
        self.dataset = get_data(path=self.data_path,
                                save_dir=self.save_dir,
                                smiles_columns=self.smiles_columns,
                                targets_columns=self.targets_columns,
                                features_generators=self.features_generators)
        if self.separate_val_path is not None:
            self.dataset_val = get_data(path=self.separate_val_path,
                                        save_dir=self.save_dir,
                                        smiles_columns=self.smiles_columns,
                                        targets_columns=self.targets_columns,
                                        features_generators=self.features_generators)
            self.dataset_train_val = copy.copy(self.dataset)
            self.dataset_train_val.data = self.dataset.data + self.dataset_val.data
        if self.separate_test_path is not None:
            self.dataset_test = get_data(path=self.separate_test_path,
                                         save_dir=self.save_dir,
                                         smiles_columns=self.smiles_columns,
                                         targets_columns=self.targets_columns,
                                         features_generators=self.features_generators)


class OptunaArgs(TrainArgs):
    n_trials: int = 100
    """Number of Optuna trials to perform."""
