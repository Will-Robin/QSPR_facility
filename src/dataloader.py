import torch
import sqlite3
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from dataclasses import dataclass
from torch_geometric.data import Data

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
    def test_targets(self):
        return self.y_test

    @property
    def test_compound_ids(self):
        return self.compound_ids_test


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
    def test_targets(self):
        return np.array([g.y.item() for g in self.test_graphs])

    @property
    def test_compound_ids(self):
        return self.compound_ids_test


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

    DESCRIPTORS = {
        "MolWt": Descriptors.MolWt,
        "TPSA": Descriptors.TPSA,
        "MolLogP": Descriptors.MolLogP,
        "NumHDonors": Descriptors.NumHDonors,
        "NumHAcceptors": Descriptors.NumHAcceptors,
    }

    def featurize_smiles(self, smiles):
        mol = Chem.MolFromSmiles(smiles)

        return [f(mol) for f in self.DESCRIPTORS.values()]

    def transform(
        self,
        dataset: SplitDataset,
        target: str,
    ):
        train_df = dataset.train
        val_df = dataset.validation
        test_df = dataset.test

        feature_names = list(self.DESCRIPTORS.keys())

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
    ):
        mol = Chem.MolFromSmiles(smiles)

        atom_features = []

        for atom in mol.GetAtoms():
            atom_features.append(
                [
                    atom.GetAtomicNum(),
                    atom.GetDegree(),
                    atom.GetFormalCharge(),
                    atom.GetHybridization(),
                    atom.GetIsAromatic(),
                ]
            )
        edge_index = []
        edge_attr = []

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            features = [
                bond.GetBondTypeAsDouble(),
                int(bond.GetIsAromatic()),
                int(bond.GetIsConjugated()),
                int(bond.IsInRing()),
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
