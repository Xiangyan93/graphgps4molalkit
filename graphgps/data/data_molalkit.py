#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import List, Set, Tuple, Union, Dict
import time
import numpy as np
from functools import partial
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loader
from graphgps.data.data import DatasetFromCSVFile
from graphgps.transform.transforms import pre_transform_in_memory
from graphgps.transform.posenc_stats import compute_posenc_stats


class DataPoint:
    def __init__(self, smiles: List[str], targets: List[float], id_pyg: int):
        self.smiles = smiles
        self.targets = targets
        self.id_pyg = id_pyg


class Dataset:
    def __init__(self, dataset_pyg_full):
        self.dataset_pyg_full = dataset_pyg_full
        self.compute_posenc_stats()
        self.data = []
        for i, smiles_list in enumerate(dataset_pyg_full.smiles):
            # assert len(smiles_list) == 1, f"Expected 1 SMILES string, got {len(smiles_list)}"
            self.data.append(DataPoint(smiles=smiles_list, targets=dataset_pyg_full.y[i].tolist(), id_pyg=i))

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, item):
        return self.data[item]

    @property
    def dataset_pyg(self):
        ids_pyg = [data.id_pyg for data in self.data]
        return self.dataset_pyg_full[ids_pyg]

    @property
    def y(self):
        return np.array(self.dataset_pyg.y)

    def num_tasks(self):
        return self.y.shape[1]

    def compute_posenc_stats(self):
        # Precompute necessary statistics for positional encodings.
        pe_enabled_list = []
        for key, pecfg in cfg.items():
            if key.startswith('posenc_') and pecfg.enable:
                pe_name = key.split('_', 1)[1]
                pe_enabled_list.append(pe_name)
                if hasattr(pecfg, 'kernel'):
                    # Generate kernel times if functional snippet is set.
                    if pecfg.kernel.times_func:
                        pecfg.kernel.times = list(eval(pecfg.kernel.times_func))
                    print(f"Parsed {pe_name} PE kernel times / steps: "
                                f"{pecfg.kernel.times}")
        if pe_enabled_list:
            start = time.perf_counter()
            print(f"Precomputing Positional Encoding statistics: "
                        f"{pe_enabled_list} for all graphs...")
            # Estimate directedness based on 10 graphs to save time.
            is_undirected = all(d.is_undirected() for d in self.dataset_pyg_full[:10])
            print(f"  ...estimated to be undirected: {is_undirected}")
            pre_transform_in_memory(self.dataset_pyg_full,
                                    partial(compute_posenc_stats,
                                            pe_types=pe_enabled_list,
                                            is_undirected=is_undirected,
                                            cfg=cfg),
                                    show_progress=True)
            elapsed = time.perf_counter() - start
            timestr = time.strftime('%H:%M:%S', time.gmtime(elapsed)) \
                    + f'{elapsed:.2f}'[-3:]
            print(f"Done! Took {timestr}")


def get_data(path: str, save_dir: str,
             smiles_columns: List[str] = None,
             targets_columns: List[str] = None,
             features_generators: List[str] = None,
             n_jobs: int = 8):
    dataset_pyg_full = DatasetFromCSVFile(data_path=path,
                                          smiles_columns=smiles_columns,
                                          target_columns=targets_columns,
                                          features_generators=features_generators,
                                          root='%s/graphgps' % save_dir)
    return Dataset(dataset_pyg_full)
