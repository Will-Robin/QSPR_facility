import torch
import sqlite3
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from dataclasses import dataclass
from torch_geometric.data import Data
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator


@dataclass
class SplitDataset:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


@dataclass
class VectorDataset:
    compound_ids_train: np.ndarray
    compound_ids_val: np.ndarray
    compound_ids_test: np.ndarray

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray

    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray

    feature_names: list[str]

    @property
    def train_targets(self):
        return self.y_train

    @property
    def validation_targets(self):
        return self.y_val

    @property
    def test_targets(self):
        return self.y_test

    @property
    def train_compound_ids(self):
        return self.compound_ids_train

    @property
    def validation_compound_ids(self):
        return self.compound_ids_val

    @property
    def test_compound_ids(self):
        return self.compound_ids_test

    def get_training_data(self, include_validation=False):
        if include_validation:
            return (
                np.concatenate([self.X_train, self.X_val]),
                np.concatenate([self.y_train, self.y_val]),
            )

        return self.X_train, self.y_train

    def get_validation_data(self):
        return self.X_val, self.y_val

    def get_test_data(self):
        return self.X_test, self.y_test


@dataclass
class GraphDataset:
    compound_ids_train: list[int]
    compound_ids_val: list[int]
    compound_ids_test: list[int]

    train_graphs: list
    val_graphs: list
    test_graphs: list

    num_node_features: int
    num_edge_features: int

    target_name: str
    num_targets: int

    @property
    def train_compound_ids(self):
        return self.compound_ids_train

    @property
    def validation_compound_ids(self):
        return self.compound_ids_val

    @property
    def test_compound_ids(self):
        return self.compound_ids_test

    @property
    def train_targets(self):
        return np.array([g.y.item() for g in self.train_graphs])

    @property
    def validation_targets(self):
        return np.array([g.y.item() for g in self.val_graphs])

    @property
    def test_targets(self):
        return np.array([g.y.item() for g in self.test_graphs])

    def get_training_graphs(self, include_validation=False):
        if include_validation:
            return self.train_graphs + self.val_graphs

        return self.train_graphs

    def get_validation_graphs(self):
        return self.val_graphs

    def get_test_graphs(self):
        return self.test_graphs


class Featurizer:
    output_type = None

    def transform(
        self,
        dataset: SplitDataset,
        target: str,
    ):
        raise NotImplementedError


class DescriptorFeaturizer(Featurizer):
    output_type = VectorDataset

    DESCRIPTOR_NAMES = [d[0] for d in Descriptors._descList if "BCUT" not in d[0]]

    FEATURIZER = MoleculeDescriptors.MolecularDescriptorCalculator(DESCRIPTOR_NAMES)

    def featurize_smiles(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        return self.FEATURIZER.CalcDescriptors(mol)

    def transform(
        self,
        dataset: SplitDataset,
        target: str,
    ):
        train_df = dataset.train
        val_df = dataset.validation
        test_df = dataset.test

        feature_names = self.DESCRIPTOR_NAMES

        X_train = np.array([self.featurize_smiles(s) for s in train_df["SMILES"]])

        X_val = np.array([self.featurize_smiles(s) for s in val_df["SMILES"]])

        X_test = np.array([self.featurize_smiles(s) for s in test_df["SMILES"]])

        return VectorDataset(
            compound_ids_train=train_df["compound_id"].values,
            compound_ids_val=val_df["compound_id"].values,
            compound_ids_test=test_df["compound_id"].values,
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=train_df[target].values,
            y_val=val_df[target].values,
            y_test=test_df[target].values,
            feature_names=feature_names,
        )


class ECFPFeaturizer(Featurizer):
    def __init__(
        self,
        radius=2,
        n_bits=2048,
    ):
        self.radius = radius
        self.n_bits = n_bits

        self.generator = GetMorganGenerator(
            radius=radius,
            fpSize=n_bits,
        )

    def featurize_smiles(self, smiles):
        mol = Chem.MolFromSmiles(smiles)

        fp = self.generator.GetFingerprint(mol)

        return np.array(fp)

    def transform(
        self,
        dataset: SplitDataset,
        target: str,
    ):
        train_df = dataset.train
        val_df = dataset.validation
        test_df = dataset.test

        X_train = np.array([self.featurize_smiles(s) for s in train_df["SMILES"]])

        X_val = np.array([self.featurize_smiles(s) for s in val_df["SMILES"]])

        X_test = np.array([self.featurize_smiles(s) for s in test_df["SMILES"]])

        return VectorDataset(
            compound_ids_train=train_df["compound_id"].values,
            compound_ids_val=val_df["compound_id"].values,
            compound_ids_test=test_df["compound_id"].values,
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=train_df[target].values,
            y_val=val_df[target].values,
            y_test=test_df[target].values,
            feature_names=[f"bit_{i}" for i in range(self.n_bits)],
        )


class GraphFeaturizer(Featurizer):
    def smiles_to_graph(
        self,
        smiles,
        y=None,
        use_chirality=True,
    ):
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        atom_features = []

        for atom in mol.GetAtoms():
            hybridisation = atom.GetHybridization()

            features = [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                int(atom.GetIsAromatic()),
                atom.GetNumRadicalElectrons(),
                atom.GetTotalNumHs(),
            ]

            known_hybridisations = [
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2,
            ]

            features += [int(hybridisation == x) for x in known_hybridisations]

            features.append(int(hybridisation not in known_hybridisations))

            atom_features.append(features)

        edge_index = []
        edge_attr = []

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_type = bond.GetBondType()
            bond_stereo = str(bond.GetStereo())

            features = [
                bond.GetBondTypeAsDouble(),
                int(bond.GetIsAromatic()),
                int(bond.GetIsConjugated()),
                int(bond.IsInRing()),
                int(bond_type == Chem.rdchem.BondType.SINGLE),
                int(bond_type == Chem.rdchem.BondType.DOUBLE),
                int(bond_type == Chem.rdchem.BondType.TRIPLE),
                int(bond_type == Chem.rdchem.BondType.AROMATIC),
            ]

            if use_chirality:
                features += [
                    int(bond_stereo == x)
                    for x in ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"]
                ]

            edge_index.append([i, j])
            edge_attr.append(features)

            edge_index.append([j, i])
            edge_attr.append(features)

        data = Data(
            x=torch.tensor(
                atom_features,
                dtype=torch.float,
            ),
            edge_index=torch.tensor(
                edge_index,
                dtype=torch.long,
            )
            .t()
            .contiguous(),
            edge_attr=torch.tensor(
                edge_attr,
                dtype=torch.float,
            ),
        )
        if y is not None:
            data.y = torch.tensor(
                [y],
                dtype=torch.float,
            )

        return data

    def transform(
        self,
        dataset: SplitDataset,
        target: str,
    ):
        train_df = dataset.train
        val_df = dataset.validation
        test_df = dataset.test

        train_graphs = [
            self.smiles_to_graph(
                row.SMILES,
                row[target],
            )
            for _, row in train_df.iterrows()
        ]

        val_graphs = [
            self.smiles_to_graph(
                row.SMILES,
                row[target],
            )
            for _, row in val_df.iterrows()
        ]

        test_graphs = [
            self.smiles_to_graph(
                row.SMILES,
                row[target],
            )
            for _, row in test_df.iterrows()
        ]

        return GraphDataset(
            train_graphs=train_graphs,
            val_graphs=val_graphs,
            test_graphs=test_graphs,
            target_name=target,
            compound_ids_train=train_df["compound_id"].values,
            compound_ids_val=val_df["compound_id"].values,
            compound_ids_test=test_df["compound_id"].values,
            num_node_features=train_graphs[0].x.shape[1],
            num_edge_features=train_graphs[0].edge_attr.shape[1],
            num_targets=1,
        )


class SQLiteDataLoader:
    def load(
        self,
        ml_database,
        split_name,
        target,
    ) -> SplitDataset:
        query = f"""
        SELECT
            s.split_type,
            d.*
        FROM surfpro_dataset_v1 d -- uses the surfpro_dataset_v1 VIEW
        JOIN ml_splits s
            ON d.compound_id = s.compound_id
        WHERE s.split_name = ?
        AND d.{target} IS NOT NULL;
        """

        with sqlite3.connect(ml_database) as conn:
            df = pd.read_sql_query(
                query,
                conn,
                params=(split_name,),
            )

            split_dataset = SplitDataset(
                train=df[df["split_type"] == "train"].reset_index(drop=True),
                validation=df[df["split_type"] == "validation"].reset_index(drop=True),
                test=df[df["split_type"] == "test"].reset_index(drop=True),
            )

            return split_dataset
