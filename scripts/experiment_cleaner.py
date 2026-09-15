import sqlite3
import toml
from pathlib import Path
from src.experiment import Experiment, ExperimentRunner, experiment_exists
from src.experiment import Experiment


def delete_experiment(ml_database, experiment_hash):
    with sqlite3.connect(ml_database) as conn:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT experiment_id
            FROM experiments
            WHERE experiment_hash = ?
            """,
            (experiment_hash,),
        )

        row = cursor.fetchone()

        if row is None:
            return

        experiment_id = row[0]

        cursor.execute(
            """
            DELETE FROM experiment_predictions
            WHERE experiment_id = ?
            """,
            (experiment_id,),
        )

        cursor.execute(
            """
            DELETE FROM experiment_metrics
            WHERE experiment_id = ?
            """,
            (experiment_id,),
        )

        cursor.execute(
            """
            DELETE FROM experiments
            WHERE experiment_id = ?
            """,
            (experiment_id,),
        )

        conn.commit()


def clear_experiments(ml_database):
    with sqlite3.connect(ml_database) as conn:
        conn.execute("DELETE FROM experiment_predictions")

        conn.execute("DELETE FROM experiment_metrics")

        conn.execute("DELETE FROM experiments")

        conn.commit()


def main():
    config = toml.loads(Path("config.toml").read_text())
    ml_database = config["ML_DATABASE"]

    for file in Path("experiments").iterdir():
        if any(
            [
                x in file.name
                for x in ["lasso", "linear", "ridge", "random_forest", "svr"]
            ]
        ):
            experiment = Experiment.from_toml(file)
            print(file.name)
            delete_experiment(ml_database, experiment.experiment_hash)
    # clear_experiments(ml_database)


if __name__ == "__main__":
    main()
