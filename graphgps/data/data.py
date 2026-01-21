from typing import Callable, Optional, List, Literal, Dict
import os
import os.path as osp
import shutil
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
import torch
from torch_geometric.data import Dataset, InMemoryDataset
from torch_geometric.datasets.molecule_net import MoleculeNet
from torch_geometric.utils import from_smiles
from mgktools.features_mol.features_generators import FeaturesGenerator


SMILES_TO_FEATURES: Dict[str, torch.tensor] = {}


class DatasetFromCSVFile(InMemoryDataset):
    def __init__(
        self,
        data_path: str,
        smiles_columns: List[str],
        target_columns: List[str] = None,
        features_columns: List[str] = None,
        features_generators: List[FeaturesGenerator] = None,
        task_type: Literal['classification', 'regression'] = 'regression',
        root: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        log: bool = True,
    ):
        self.data_path = data_path
        self.file_name = data_path.split('/')[-1].split('.')[0]
        self.smiles_columns = smiles_columns
        self.target_columns = target_columns
        self.features_columns = features_columns
        self.features_generators = features_generators or []
        self.task_type = task_type
        super(DatasetFromCSVFile, self).__init__(root, transform, pre_transform, pre_filter, log)
        path = osp.join(self.processed_dir, self.file_name + '.pt')
        self.data, self.slices = torch.load(path, weights_only=False)

    @property
    def raw_file_names(self) -> List[str]:
        return [self.file_name + '.csv']

    @property
    def processed_file_names(self) -> List[str]:
        return [self.file_name + '.pt']

    def download(self):
        # copy file from data_path to root
        shutil.copyfile(self.data_path, osp.join(self.raw_dir, self.file_name + '.csv'))

    def process(self):
        for i, file in enumerate(self.raw_paths):
            data_list = []
            df = pd.read_csv(file)
            for smiles_column in self.smiles_columns:
                assert smiles_column in df.columns
            df['smiles'] = df.apply(lambda x: '.'.join([x[smiles_column] for smiles_column in self.smiles_columns]), axis=1)
            for j, row in tqdm(df.iterrows(), total=len(df)):
                data = from_smiles(row['smiles'])
                # Remap RDKit bond types to OGB bond indices (0-4):
                # OGB expects: 0=SINGLE, 1=DOUBLE, 2=TRIPLE, 3=AROMATIC, 4=misc
                # RDKit returns: 1=SINGLE, 2=DOUBLE, 3=TRIPLE, 12=AROMATIC, 17=DATIVE, etc.
                bond_type_map = {1: 0, 2: 1, 3: 2, 12: 3}  # RDKit -> OGB
                edge_types = data.edge_attr[:, 0].clone()
                for rdkit_type, ogb_type in bond_type_map.items():
                    edge_types[data.edge_attr[:, 0] == rdkit_type] = ogb_type
                # Map any unmapped bond types (DATIVE, etc.) to 4 (misc)
                unmapped_mask = ~torch.isin(data.edge_attr[:, 0], torch.tensor(list(bond_type_map.keys())))
                edge_types[unmapped_mask] = 4
                data.edge_attr[:, 0] = edge_types
                if self.target_columns is None:
                    data.y = None
                elif self.task_type == 'regression':
                    data.y = torch.tensor([row[self.target_columns].to_numpy().tolist()], dtype=torch.float32)# .view(1, -1)
                else:
                    data.y = torch.tensor([row[self.target_columns].to_numpy().tolist()], dtype=torch.int32)# .view(1, -1)
                data.smiles = row[self.smiles_columns].tolist()

                if self.features_columns is not None:
                    features = row[self.features_columns].to_numpy().tolist()
                else:
                    features = []

                if len(self.features_generators) > 0:
                    for smiles in data.smiles:
                        if smiles not in SMILES_TO_FEATURES:
                            mol = Chem.MolFromSmiles(smiles)
                            features_mol = []
                            for fg in self.features_generators:
                                fs = fg(mol)
                                fs = np.where(np.isnan(fs), 0, fs)
                                features_mol.append(fs)
                            features_mol = np.concatenate(features_mol).tolist()
                            SMILES_TO_FEATURES[smiles] = features_mol
                        features.append(SMILES_TO_FEATURES[smiles])
                    features += SMILES_TO_FEATURES[smiles]
                data.features = torch.tensor(features, dtype=torch.float32).view(1, -1)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
            torch.save(self.collate(data_list), self.processed_paths[i])
