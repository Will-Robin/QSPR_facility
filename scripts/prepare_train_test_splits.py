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
from hashlib import sha256
from src.registry import SPLIT_STRATEGIES


def create_split_hash(split_definition):
    return sha256(
        json.dumps(
            split_definition,
            sort_keys=True,
        ).encode()
    ).hexdigest()


def create_splitter(split_definition):
    strategy = split_definition["strategy"]

    splitter_cls = SPLIT_STRATEGIES.get(strategy)

    if splitter_cls is None:
        raise ValueError(f"Unknown strategy: {strategy}")

    return splitter_cls(
        test_size=split_definition["test_size"],
        validation_size=split_definition["validation_size"],
        random_seed=split_definition["random_seed"],
        stratify_column=split_definition.get("stratify_column"),
        test_elements=split_definition.get("test_elements"),
    )


def validate_split_definition(split_definition):
    if split_definition["strategy"] not in SPLIT_STRATEGIES:
        raise ValueError(f"Unknown strategy: {split_definition['strategy']}")

    test_size = split_definition["test_size"]
    validation_size = split_definition["validation_size"]

    if test_size <= 0 or validation_size <= 0:
        raise ValueError("Split sizes must be positive")

    if test_size + validation_size >= 1:
        raise ValueError("test_size + validation_size must be less than 1")


def load_split_data(conn, splitter):
    columns = splitter.required_columns

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
    splitter = create_splitter(split_definition)

    return splitter.split(df)


def build_split_dataframe(
    train_ids,
    validation_ids,
    test_ids,
    split_name,
    split_hash,
    split_definition,
):
    strategy = split_definition["strategy"]
    random_seed = split_definition["random_seed"]

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

    df_splits["random_seed"] = random_seed
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
    splitter = create_splitter(split_definition)

    with sqlite3.connect(ml_database) as conn:
        df = load_split_data(conn, splitter)
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
