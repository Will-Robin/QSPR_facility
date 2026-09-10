import toml
import json
import hashlib
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from src.registry import MODEL_REGISTRY, FEATURIZER_REGISTRY
from src.dataloader import SQLiteDataLoader


@dataclass
class Experiment:
    experiment_name: str

    split_name: str
    target: str

    featurizer: str
    model: str

    parameters: dict

    raw_toml: str | None = None

    source_file: str | None = None

    @classmethod
    def from_toml(cls, path):
        raw_toml = Path(path).read_text()

        config = toml.loads(raw_toml)

        return cls(
            experiment_name=config["experiment_name"],
            split_name=config["split_name"],
            target=config["target"],
            featurizer=config["featurizer"],
            model=config["model"],
            parameters=config["parameters"],
            raw_toml=raw_toml,
            source_file=str(path),
        )

    @property
    def experiment_hash(self):
        payload = {
            "split_name": self.split_name,
            "target": self.target,
            "featurizer": self.featurizer,
            "model": self.model,
            "parameters": self.parameters,
        }

        return hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
            ).encode()
        ).hexdigest()

    def to_dict(self):
        return {
            "experiment_name": self.experiment_name,
            "split_name": self.split_name,
            "target": self.target,
            "featurizer": self.featurizer,
            "model": self.model,
            "parameters": self.parameters,
            "experiment_hash": self.experiment_hash,
        }


@dataclass
class ExperimentResult:
    experiment: Experiment

    metrics: dict

    compound_ids: np.ndarray
    observed: np.ndarray
    predicted: np.ndarray

    training_loss: list[float] | None = None

    def save_to_database(self, ml_database):
        with sqlite3.connect(ml_database) as conn:
            cursor = conn.cursor()

            cursor.execute("PRAGMA foreign_keys = ON")
            cursor.execute(
                """
                INSERT INTO experiments (
                    experiment_name,
                    experiment_hash,
                    source_file,
                    split_name,
                    target,
                    featurizer,
                    model,
                    parameters_json,
                    raw_toml,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.experiment.experiment_name,
                    self.experiment.experiment_hash,
                    self.experiment.source_file,
                    self.experiment.split_name,
                    self.experiment.target,
                    self.experiment.featurizer,
                    self.experiment.model,
                    json.dumps(self.experiment.parameters),
                    self.experiment.raw_toml,
                    datetime.now().isoformat(),
                ),
            )

            experiment_id = cursor.lastrowid

            for metric, value in self.metrics.items():
                cursor.execute(
                    """
                    INSERT INTO
                    experiment_metrics (
                        experiment_id,
                        metric,
                        value
                    )
                    VALUES (?, ?, ?)
                    """,
                    (
                        experiment_id,
                        metric,
                        value,
                    ),
                )
            for compound_id, obs, pred in zip(
                self.compound_ids,
                self.observed,
                self.predicted,
            ):
                cursor.execute(
                    """
                    INSERT INTO
                    experiment_predictions (
                        experiment_id,
                        compound_id,
                        observed,
                        predicted
                    )
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        experiment_id,
                        int(compound_id),
                        float(obs),
                        float(pred),
                    ),
                )

            conn.commit()

    def save_artifacts(self, output_dir):
        if self.training_loss is None:
            return

        output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(
            {
                "epoch": range(len(self.training_loss)),
                "loss": self.training_loss,
            }
        ).to_csv(
            output_dir / "loss.csv",
            index=False,
        )


class ExperimentRunner:
    def __init__(self, ml_database):
        self.ml_database = ml_database

    def run(self, experiment: Experiment) -> ExperimentResult:
        loader = SQLiteDataLoader()

        dataset = loader.load(
            self.ml_database,
            split_name=experiment.split_name,
            target=experiment.target,
        )

        featurizer = FEATURIZER_REGISTRY[experiment.featurizer]()

        representation = featurizer.transform(dataset, target=experiment.target)

        model = MODEL_REGISTRY[experiment.model](**experiment.parameters)

        model.fit(representation)

        metrics = model.evaluate(representation)

        predictions = model.predict(representation)

        return ExperimentResult(
            experiment=experiment,
            metrics=metrics,
            compound_ids=representation.compound_ids_test,
            observed=representation.test_targets,
            predicted=predictions,
            training_loss=getattr(
                model,
                "training_loss",
                None,
            ),
        )


def experiment_exists(ml_database, experiment):
    result = None
    with sqlite3.connect(ml_database) as conn:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT 1
            FROM experiments
            WHERE experiment_hash = ?
            """,
            (experiment.experiment_hash,),
        )

        result = cursor.fetchone() is not None

    return result
