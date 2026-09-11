"""
Using SQLite to store train/validation/test splits and retrieve subsets for
modelling.

This approach is interesting. If we assign train/validation/test splits in a
way that we know there can be no data leakage (e.g. by splitting on structures),
then we can indiscriminately select subsets from the database via any criterion
and let the splits take care of the preventing data leakage.
This means that the query given in `surfpro_view` can be replaced with any
valid query, and can be the subject of optimisation for predicting the test
set?
"""

import toml
import json
import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import StratifiedGroupKFold
from hashlib import sha256

VALID_SPLIT_STRATEGIES = {
    "random",
    "stratified",
    "murcko_scaffold",
}


def create_split_hash(split_definition):
    return sha256(
        json.dumps(
            split_definition,
            sort_keys=True,
        ).encode()
    ).hexdigest()


def get_murcko_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    return MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol,
        includeChirality=False,
    )


def random_split(
    ids,
    test_size,
    validation_size,
    random_state,
):
    val_fraction = validation_size / (validation_size + test_size)

    train_ids, temp_ids = train_test_split(
        ids,
        test_size=test_size + validation_size,
        random_state=random_state,
    )

    validation_ids, test_ids = train_test_split(
        temp_ids,
        train_size=val_fraction,
        random_state=random_state,
    )

    return train_ids, validation_ids, test_ids


def stratified_split(
    df,
    ids,
    stratify_column,
    test_size,
    validation_size,
    random_state,
):
    val_fraction = validation_size / (validation_size + test_size)

    train_ids, temp_ids = train_test_split(
        ids,
        test_size=test_size + validation_size,
        stratify=df[stratify_column],
        random_state=random_state,
    )

    class_counts = df[stratify_column].value_counts()

    if class_counts.min() < 3:
        raise ValueError(
            f"Cannot stratify because some classes have fewer "
            f"than 3 examples:\n{class_counts}"
        )

    temp_df = df[df["compound_id"].isin(temp_ids)]

    temp_counts = temp_df[stratify_column].value_counts()

    if temp_counts.min() < 2:
        raise ValueError(
            "Temporary validation/test set contains "
            "classes with fewer than two samples:\n"
            f"{temp_counts}"
        )

    validation_ids, test_ids = train_test_split(
        temp_ids,
        train_size=val_fraction,
        stratify=temp_df[stratify_column],
        random_state=random_state,
    )

    return train_ids, validation_ids, test_ids


def scaffold_split(
    df,
    ids,
    test_size,
    validation_size,
    random_state,
):
    val_fraction = validation_size / (validation_size + test_size)

    df = df.copy()
    df["scaffold"] = (
        df["SMILES"].apply(get_murcko_scaffold).replace("", pd.NA).fillna(df["SMILES"])
    )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size + validation_size,
        random_state=random_state,
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
        random_state=random_state,
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


def validate_split_definition(split_definition):
    if split_definition["strategy"] not in VALID_SPLIT_STRATEGIES:
        raise ValueError(f"Unknown strategy: {split_definition['strategy']}")

    test_size = split_definition["test_size"]
    validation_size = split_definition["validation_size"]

    if test_size <= 0 or validation_size <= 0:
        raise ValueError("Split sizes must be positive")

    if test_size + validation_size >= 1:
        raise ValueError("test_size + validation_size must be less than 1")


def get_required_columns(split_definition):
    strategy = split_definition["strategy"]

    columns = ["compound_id"]
    stratify_column = split_definition.get("stratify_column")

    if strategy == "murcko_scaffold":
        columns.append("SMILES")

    if strategy == "stratified":
        columns.append(stratify_column)

    return columns


def load_split_data(conn, split_definition):
    columns = get_required_columns(split_definition)

    return pd.read_sql_query(
        f"""
        SELECT
            {",".join(columns)}
        FROM compounds
        ORDER BY compound_id
        """,
        conn,
    )


def generate_split(df, split_definition):
    ids = df["compound_id"]

    strategy = split_definition["strategy"]
    test_size = split_definition["test_size"]
    validation_size = split_definition["validation_size"]
    random_state = split_definition["random_seed"]
    stratify_column = split_definition.get("stratify_column")

    if strategy == "random":
        train_ids, validation_ids, test_ids = random_split(
            ids,
            test_size,
            validation_size,
            random_state,
        )

    elif strategy == "stratified":
        train_ids, validation_ids, test_ids = stratified_split(
            df,
            ids,
            stratify_column,
            test_size,
            validation_size,
            random_state,
        )

    elif strategy == "murcko_scaffold":
        train_ids, validation_ids, test_ids = scaffold_split(
            df,
            ids,
            test_size,
            validation_size,
            random_state,
        )

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return train_ids, validation_ids, test_ids


def build_split_dataframe(
    train_ids,
    validation_ids,
    test_ids,
    split_name,
    split_hash,
    split_definition,
):
    strategy = split_definition["strategy"]
    random_state = split_definition["random_seed"]

    df_splits = pd.concat(
        [
            pd.DataFrame({"compound_id": train_ids, "split_type": "train"}),
            pd.DataFrame({"compound_id": validation_ids, "split_type": "validation"}),
            pd.DataFrame({"compound_id": test_ids, "split_type": "test"}),
        ],
        ignore_index=True,
    )

    df_splits["split_name"] = split_name
    df_splits["split_strategy"] = strategy

    df_splits["random_seed"] = random_state
    df_splits["split_hash"] = split_hash

    return df_splits


def store_split(
    conn,
    split_name,
    split_hash,
    split_definition,
    df_splits,
):
    # wipe previous version of splitmurcko_scaffold
    conn.execute(
        "DELETE FROM ml_splits WHERE split_name = ?",
        (split_name,),
    )

    conn.execute(
        """
        DELETE FROM split_definitions
        WHERE split_name = ?
        """,
        (split_name,),
    )

    conn.execute(
        """
        INSERT INTO split_definitions (
            split_name,
            split_hash,
            split_strategy,
            split_definition_json,
            created_at
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            split_name,
            split_hash,
            split_definition["strategy"],
            json.dumps(
                split_definition,
                sort_keys=True,
            ),
            datetime.now().isoformat(),
        ),
    )

    df_splits.to_sql(
        "ml_splits",
        conn,
        if_exists="append",
        index=False,
    )


def prepare_split(ml_database, split_name, split_definition):
    """
    Create and store a train/validation/test split.

    Strategies:
        random
        stratified
    """

    validate_split_definition(split_definition)

    split_hash = create_split_hash(split_definition)

    with sqlite3.connect(ml_database) as conn:
        df = load_split_data(conn, split_definition)
        train_ids, validation_ids, test_ids = generate_split(df, split_definition)
        df_splits = build_split_dataframe(
            train_ids,
            validation_ids,
            test_ids,
            split_name,
            split_hash,
            split_definition,
        )

        store_split(
            conn,
            split_name,
            split_hash,
            split_definition,
            df_splits,
        )


def get_data_split(ml_database, output_file_name, split_name="random_v1"):
    """
    Get a dataset for ML with a specified set of train/validation/test labels
    """

    query = """
    SELECT
        s.split_type,
        d.*
    FROM surfpro_dataset_v1 d -- uses the surfpro_dataset_v1 VIEW
    JOIN ml_splits s
        ON d.compound_id = s.compound_id
    WHERE s.split_name = ?;
    """

    with sqlite3.connect(ml_database) as conn:
        df = pd.read_sql_query(
            query,
            conn,
            params=(split_name,),
        )

        df.to_csv(output_file_name, index=False)


def main():
    config = toml.loads(Path("config.toml").read_text())
    ml_database = Path(config["ML_DATABASE"])
    split_file = Path(config["SPLIT_DEFINITIONS"])

    split_definitions = json.loads(split_file.read_text())

    for split_name, split_definition in split_definitions.items():
        prepare_split(ml_database, split_name, split_definition)


if __name__ == "__main__":
    main()
