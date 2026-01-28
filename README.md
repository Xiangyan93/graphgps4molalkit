# GraphGPS for MolALKit

A customized version of [GraphGPS](https://github.com/rampasek/GraphGPS) (General Powerful Scalable Graph Transformer) optimized for molecular active learning. It serves as the molecular property prediction backend for [MolALKit](https://github.com/RekerLab/MolALKit).

## Key Modifications

### Easy-to-Use CLI Tools

Two command-line interfaces for training and optimization:

- **`graphgps_cv`**: Cross-validation pipeline for model training and evaluation
- **`graphgps_optuna`**: Hyperparameter optimization using Optuna with TPE sampler

### Graph-Feature Combination Learning

Supports combining graph neural network representations with molecular fingerprints:

- **RDKit 2D descriptors**: `rdkit_2d`, `rdkit_2d_normalized` (200 features)
- **Morgan fingerprints**: `morgan`, `morgan_count` (2048 features)
- **Custom features**: Additional feature columns from CSV input

### CSV-Based Data Input

Accepts standard CSV files with SMILES strings and target values, eliminating the need for pre-processed graph datasets.

**Dependencies**: PyTorch 2.6.0, PyTorch Geometric, PyTorch Lightning, Optuna, OGB, mgktools

## Usage

### Cross-Validation

```bash
graphgps_cv --data_path data.csv --save_dir output \
    --smiles_columns smiles --targets_columns target \
    --dataset_type regression --metric mae \
    --n_jobs 8 --seed 0
```

With molecular features:
```bash
graphgps_cv --data_path data.csv --save_dir output \
    --smiles_columns smiles --targets_columns target \
    --features_generators_name morgan rdkit_2d \
    --dataset_type regression --metric mae
```

With separate test set:
```bash
graphgps_cv --data_path train.csv --save_dir output \
    --smiles_columns smiles --targets_columns target \
    --separate_test_path test.csv \
    --dataset_type regression --metric mae
```

### Hyperparameter Optimization

```bash
graphgps_optuna --data_path data.csv --save_dir output \
    --smiles_columns smiles --targets_columns target \
    --dataset_type regression --metric mae \
    --n_trials 100
```

## CLI Arguments

| Argument | Description |
|----------|-------------|
| `--data_path` | Input CSV file path |
| `--smiles_columns` | Column name(s) containing SMILES |
| `--targets_columns` | Column name(s) for target values |
| `--features_columns` | Additional feature columns (optional) |
| `--features_generators_name` | Molecular features: `rdkit_2d`, `rdkit_2d_normalized`, `morgan`, `morgan_count` |
| `--dataset_type` | `regression`, `classification`, or `multiclass` |
| `--metric` | Primary metric: `mae`, `rmse`, `r2`, `auc`, etc. |
| `--extra_metrics` | Additional metrics to compute |
| `--cross_validation` | CV method: `kFold`, `leave-one-out`, `Monte-Carlo`, `no` |
| `--n_splits` | Number of folds for kFold CV |
| `--ensemble_size` | Number of models in ensemble |
| `--separate_test_path` | Separate test set CSV |
| `--separate_val_path` | Separate validation set CSV |
| `--n_trials` | Number of Optuna trials (for `graphgps_optuna`) |
| `--n_jobs` | Number of parallel workers |
| `--seed` | Random seed |

## Model Configuration

The model uses a YACS-based configuration system. Key configurable components:

- **Positional Encodings**: RWSE, LapPE, SignNet
- **Local GNN**: GCN, GINE, GatedGCN
- **Global Attention**: Transformer, Performer
- **Graph Pooling**: mean, add, max

The default template config file is located at `graphgps/optuna/GPS_template.yaml`. Custom configurations can be provided via `--cfg_file`.

## License

MIT License
