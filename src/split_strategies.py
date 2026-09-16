import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
from src.chemoinformatics import get_murcko_scaffold


class SplitStrategy:
    required_columns = ["compound_id"]

    def __init__(
        self,
        test_size,
        validation_size,
        random_seed,
        **kwargs,
    ):
        self.test_size = test_size
        self.validation_size = validation_size
        self.random_seed = random_seed

    def split(self, df):
        raise NotImplementedError


class RandomSplitStrategy(SplitStrategy):
    def split(self, df):
        ids = df["compound_id"]

        val_fraction = self.validation_size / (self.validation_size + self.test_size)

        train_ids, temp_ids = train_test_split(
            ids,
            test_size=self.test_size + self.validation_size,
            random_state=self.random_seed,
        )

        validation_ids, test_ids = train_test_split(
            temp_ids,
            train_size=val_fraction,
            random_state=self.random_seed,
        )

        return train_ids, validation_ids, test_ids


class StratifiedSplitStrategy(SplitStrategy):
    @property
    def required_columns(self):
        return [
            "compound_id",
            self.stratify_column,
        ]

    def __init__(
        self,
        stratify_column,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.stratify_column = stratify_column

    def split(self, df):
        ids = df["compound_id"]

        val_fraction = self.validation_size / (self.validation_size + self.test_size)

        train_ids, temp_ids = train_test_split(
            ids,
            test_size=self.test_size + self.validation_size,
            stratify=df[self.stratify_column],
            random_state=self.random_seed,
        )

        class_counts = df[self.stratify_column].value_counts()

        if class_counts.min() < 3:
            raise ValueError(
                f"Cannot stratify because some classes have fewer "
                f"than 3 examples:\n{class_counts}"
            )

        temp_df = df[df["compound_id"].isin(temp_ids)]

        temp_counts = temp_df[self.stratify_column].value_counts()

        if temp_counts.min() < 2:
            raise ValueError(
                "Temporary validation/test set contains "
                "classes with fewer than two samples:\n"
                f"{temp_counts}"
            )

        validation_ids, test_ids = train_test_split(
            temp_ids,
            train_size=val_fraction,
            stratify=temp_df[self.stratify_column],
            random_state=self.random_seed,
        )

        return train_ids, validation_ids, test_ids


class MurckoScaffoldSplitStrategy(SplitStrategy):
    required_columns = [
        "compound_id",
        "SMILES",
    ]

    def split(self, df):
        ids = df["compound_id"]

        val_fraction = self.validation_size / (self.validation_size + self.test_size)

        df = df.copy()
        df["scaffold"] = (
            df["SMILES"]
            .apply(get_murcko_scaffold)
            .replace("", pd.NA)
            .fillna(df["SMILES"])
        )

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=self.test_size + self.validation_size,
            random_seed=self.random_seed,
        )

        train_idx, temp_idx = next(
            splitter.split(
                df,
                groups=df["scaffold"],
            )
        )

        train_ids = ids.iloc[train_idx]

        temp_df = df.iloc[temp_idx]

        splitter = GroupShuffleSplit(
            n_splits=1,
            train_size=val_fraction,
            random_seed=self.random_seed,
        )

        validation_idx, test_idx = next(
            splitter.split(
                temp_df,
                groups=temp_df["scaffold"],
            )
        )

        validation_ids = temp_df.iloc[validation_idx]["compound_id"]
        test_ids = temp_df.iloc[test_idx]["compound_id"]

        # check for leakage
        train_scaffolds = set(df.iloc[train_idx]["scaffold"])
        validation_scaffolds = set(temp_df.iloc[validation_idx]["scaffold"])
        test_scaffolds = set(temp_df.iloc[test_idx]["scaffold"])

        assert train_scaffolds.isdisjoint(validation_scaffolds)
        assert train_scaffolds.isdisjoint(test_scaffolds)
        assert validation_scaffolds.isdisjoint(test_scaffolds)

        return train_ids, validation_ids, test_ids
