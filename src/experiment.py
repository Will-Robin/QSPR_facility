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

MODEL_TRAINING_KEYS = {
    "learning_rate",
    "batch_size",
    "n_epochs",
}


@dataclass
class Experiment:
    experiment_name: str

    split_name: str
    target: str

    featurizer: str
    featurizer_parameters: dict

    model: str

    parameters: dict

    training: dict

    raw_toml: str | None = None

    source_file: str | None = None

    @classmethod
    def from_toml(cls, path):
        raw_toml = Path(path).read_text()

        config = toml.loads(raw_toml)

        featurizer_parameters = config.get(
            "featurizer_parameters",
            {},
        )

        parameters = config.get(
            "parameters",
            {},
        )

        training = config.get("training", {})

        return cls(
            experiment_name=config["experiment_name"],
            split_name=config["split_name"],
            target=config["target"],
            featurizer=config["featurizer"],
            featurizer_parameters=featurizer_parameters,
            model=config["model"],
            parameters=parameters,
            training=training,
            raw_toml=raw_toml,
            source_file=str(path),
        )

    @classmethod
    def from_database_row(cls, row):
        return cls(
            experiment_name=row["experiment_name"],
            split_name=row["split_name"],
            target=row["target"],
            featurizer=row["featurizer"],
            featurizer_parameters=json.loads(row["featurizer_parameters_json"]),
            model=row["model"],
            parameters=json.loads(row["parameters_json"]),
            training=json.loads(row["training_json"]),
            raw_toml=row["raw_toml"],
        )

    @property
    def experiment_hash(self):
        payload = {
            "split_name": self.split_name,
            "target": self.target,
            "featurizer": self.featurizer,
            "featurizer_parameters": self.featurizer_parameters,
            "model": self.model,
            "parameters": self.parameters,
            "training": self.training,
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
            "featurizer_parameters": self.featurizer_parameters,
            "model": self.model,
            "parameters": self.parameters,
            "experiment_hash": self.experiment_hash,
            "training": self.training,
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
                    featurizer_parameters_json,
                    model,
                    parameters_json,
                    training_json,
                    raw_toml,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.experiment.experiment_name,
                    self.experiment.experiment_hash,
                    self.experiment.source_file,
                    self.experiment.split_name,
                    self.experiment.target,
                    self.experiment.featurizer,
                    json.dumps(self.experiment.featurizer_parameters),
                    self.experiment.model,
                    json.dumps(self.experiment.parameters),
                    json.dumps(self.experiment.training),
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

        featurizer = FEATURIZER_REGISTRY[experiment.featurizer](
            **experiment.featurizer_parameters
        )

        representation = featurizer.transform(dataset, target=experiment.target)

        training_options = experiment.training

        include_validation = training_options.get("include_validation", False)

        model_kwargs = {}

        model_kwargs.update(experiment.parameters)

        for key in MODEL_TRAINING_KEYS:
            if key in experiment.training:
                model_kwargs[key] = experiment.training[key]

        model_kwargs.pop("include_validation", False)

        model = MODEL_REGISTRY[experiment.model](**model_kwargs)

        model.fit(representation, include_validation=include_validation)

        metrics = model.evaluate(representation)
        metrics.update(model.get_complexity_metrics(representation))

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
