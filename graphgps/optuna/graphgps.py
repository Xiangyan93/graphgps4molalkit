#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import List, Optional
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import trange
from graphgps.run.utils import *
from torch_geometric.graphgym.config import cfg
from torch_geometric.loader import DataLoader
from torch_geometric.graphgym.loss import compute_loss


CWD = os.path.dirname(__file__)


class GraphGPS:
    def __init__(self, save_dir: str, cfg_path: str, n_features: int = 0,
                 features_generators_name: List[str] = None,
                 generator_feature_sizes: Optional[List[int]] = None,
                 ensemble_size: int = 1, number_of_molecules: int = 1,
                 n_jobs: int = 8, seed: int = 0, features_scaling: bool = True):
        self.save_dir = save_dir
        if cfg_path is None or os.path.exists(cfg_path):
            self.cfg_path = cfg_path
        elif os.path.exists(os.path.join(CWD, cfg_path)):
            self.cfg_path = os.path.join(CWD, cfg_path)
        else:
            raise FileNotFoundError(f"Config file {cfg_path} not found.")
        self.n_features = n_features
        self.features_generators_name = features_generators_name
        self.generator_feature_sizes = generator_feature_sizes
        self.ensemble_size = ensemble_size
        self.number_of_molecules = number_of_molecules
        self.n_jobs = n_jobs
        self.seed = seed
        self.features_scaling = features_scaling
        self.scaler: Optional[StandardScaler] = None
        torch.set_num_threads(self.n_jobs)

    def _get_no_scale_indices(self) -> Optional[List[int]]:
        """Compute feature indices that should not be scaled.

        Features from rdkit_2d_normalized are already pre-normalized to [0, 1],
        so they should not be scaled again by StandardScaler.

        Feature order: features_columns first, then features_generators (per molecule).

        Returns:
            List of feature indices to skip during scaling, or None if all should be scaled.
        """
        if self.features_generators_name is None or self.generator_feature_sizes is None:
            return None
        if 'rdkit_2d_normalized' not in self.features_generators_name:
            return None

        no_scale_indices = []
        # Start offset after features_columns
        offset = self.n_features

        # For each molecule, iterate through generators in order
        for _ in range(self.number_of_molecules):
            for fg, size in zip(self.features_generators_name, self.generator_feature_sizes):
                if fg == 'rdkit_2d_normalized':
                    no_scale_indices.extend(range(offset, offset + size))
                offset += size

        return no_scale_indices if no_scale_indices else None

    def _restore_and_scale_features(self, dataset_pyg, fit: bool = False):
        """Restore features from features_raw and optionally fit/transform with StandardScaler.

        Args:
            dataset_pyg: PyG dataset whose Data objects have features and features_raw.
            fit: If True, fit the scaler on this data then transform.
                 If False, transform using the already-fitted scaler.

        Note:
            Features from rdkit_2d_normalized are NOT scaled (identity transform)
            since they are already pre-normalized to [0, 1] range.
        """
        if not self.features_scaling:
            return
        if len(dataset_pyg) == 0:
            return
        if not hasattr(dataset_pyg[0], 'features_raw'):
            return
        if dataset_pyg[0].features_raw.numel() == 0:
            return

        for data in dataset_pyg:
            data.features = data.features_raw.clone()

        raw_matrix = torch.cat([data.features for data in dataset_pyg], dim=0).numpy()

        if fit:
            self.scaler = StandardScaler()
            self.scaler.fit(raw_matrix)

            # Apply identity transform for rdkit_2d_normalized features
            no_scale_indices = self._get_no_scale_indices()
            if no_scale_indices is not None:
                self.scaler.mean_[no_scale_indices] = 0.0
                self.scaler.scale_[no_scale_indices] = 1.0

        if self.scaler is None:
            return

        scaled = self.scaler.transform(raw_matrix)

        for i, data in enumerate(dataset_pyg):
            data.features = torch.tensor(scaled[i:i+1], dtype=torch.float32)

    def fit_molalkit(self, train_data, iteration: int = 0):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.cfg_init()

        train_pyg = train_data.dataset_pyg
        self._restore_and_scale_features(train_pyg, fit=True)

        train_data_loader = DataLoader(train_pyg,
                                       batch_size=cfg.train.batch_size,
                                       shuffle=True)

        df_loss = pd.DataFrame({})
        self.models = []
        for model_idx in range(self.ensemble_size):
            custom_set_run_dir(cfg, iteration * self.ensemble_size + model_idx)
            seed_everything(self.seed + model_idx)
            model = create_model(dim_in=cfg.gnn.dim_inner,
                                 dim_out=train_data.y.shape[1])
            optimizer = create_optimizer(model.parameters(),
                                         new_optimizer_config(cfg))
            scheduler = create_scheduler(optimizer, new_scheduler_config(cfg))
            losses = []
            for i in trange(cfg.optim.max_epoch):
                loss = self.train_epoch(train_data_loader, model, optimizer,
                                        scheduler, cfg.optim.batch_accumulation)
                losses.append(loss)
            df_loss[f"model_{model_idx}"] = losses
            self.models.append(model)
        df_loss.to_csv(os.path.join(self.save_dir, "loss-iter%d.csv" % iteration), index=False)
        
    def predict_value(self, pred_data):
        self.cfg_init()
        test_pyg = pred_data.dataset_pyg
        self._restore_and_scale_features(test_pyg, fit=False)

        test_data_loader = DataLoader(test_pyg,
                                      batch_size=cfg.train.batch_size,
                                      shuffle=False)
        predictions = []
        with torch.no_grad():
            for model_idx, model in enumerate(self.models):
                preds = []
                model.eval()
                for batch in test_data_loader:
                    batch.to(torch.device(cfg.accelerator))
                    pred = model(batch)
                    preds.append(pred[0].detach().cpu().numpy())
                predictions.append(np.concatenate(preds))
        predictions = np.mean(predictions, axis=0)
        return predictions

    def predict_uncertainty(self, pred_data):
        self.cfg_init()
        if cfg.dataset.task_type == 'regression':
            raise ValueError("Uncertainty estimation is not supported for regression tasks.")
        else:
            preds = self.predict_value(pred_data)
            preds = np.array([preds, 1-preds]).T
            return (0.25 - np.var(preds, axis=1)) * 4

    def cfg_init(self):
        if self.cfg_path is not None:
            set_cfg(cfg)
            cfg.out_dir = os.path.join(self.save_dir, "graphgps")
            cfg.merge_from_file(self.cfg_path)
            dump_cfg(cfg)
            n_generator_features = 0
            if self.generator_feature_sizes is not None:
                n_generator_features = sum(self.generator_feature_sizes) * self.number_of_molecules
            total_features = self.n_features + n_generator_features
            if total_features > 0:
                cfg.gnn.use_features = True
                cfg.gnn.n_features = total_features
            auto_select_device()

    def train_epoch(self, loader, model, optimizer, scheduler, batch_accumulation):
        model.train()
        optimizer.zero_grad()
        loss_sum = 0.
        for iter, batch in enumerate(loader):
            # batch.split = 'train'
            if len(batch) == 1:
                continue
            batch.to(torch.device(cfg.accelerator))
            pred, true = model(batch)
            # Create mask to exclude NaN values
            mask = ~torch.isnan(true)
            # Only compute loss on non-NaN values
            if mask.any():
                # Apply mask BEFORE computing loss
                masked_pred = pred[mask]
                masked_true = true[mask]
                loss, pred_score = compute_loss(masked_pred, masked_true)
                # _true = true.detach().to('cpu', non_blocking=True)
                # _pred = pred_score.detach().to('cpu', non_blocking=True)
                loss.backward()
                # Parameters update after accumulating gradients for given num. batches.
                if ((iter + 1) % batch_accumulation == 0) or (iter + 1 == len(loader)):
                    if cfg.optim.clip_grad_norm:
                        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                    cfg.optim.clip_grad_norm_value)
                    optimizer.step()
                    optimizer.zero_grad()
                loss_sum += loss.detach().cpu().item()
        scheduler.step()
        return loss_sum / len(loader)
