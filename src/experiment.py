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
from src.split_generator import SplitGenerator


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

    cv_strategy: str | None = None
    cv_repeats: int = 1

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

        cv_strategy = config.get("cv_strategy", None)
        cv_repeats = config.get("cv_repeats", 1)

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
            cv_strategy=cv_strategy,
            cv_repeats=cv_repeats,
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
            cv_strategy=row["cv_strategy"],
            cv_repeats=row["cv_repeats"],
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
            "cv_strategy": self.cv_strategy,
            "cv_repeats": self.cv_repeats,
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
            "cv_strategy": self.cv_strategy,
            "cv_repeats": self.cv_repeats,
            "training": self.training,
        }


@dataclass
class ExperimentRunResult:
    run_number: int
    random_seed: int

    split_name: str

    metrics: dict

    compound_ids: np.ndarray
    observed: np.ndarray
    predicted: np.ndarray

    training_loss: list[float] | None = None


@dataclass
class ExperimentResult:
    experiment: Experiment
    runs: list[ExperimentRunResult]

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
                    cv_strategy,
                    cv_repeats,
                    raw_toml,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    self.experiment.cv_strategy,
                    self.experiment.cv_repeats,
                    self.experiment.raw_toml,
                    datetime.now().isoformat(),
                ),
            )

            experiment_id = cursor.lastrowid

            for run in self.runs:
                cursor.execute(
                    """
                INSERT INTO experiment_runs (
                    experiment_id,
                    run_number,
                    split_name,
                    random_seed,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                    (
                        experiment_id,
                        run.run_number,
                        run.split_name,
                        run.random_seed,
                        datetime.now().isoformat(),
                    ),
                )

                run_id = cursor.lastrowid

                for metric, value in run.metrics.items():
                    cursor.execute(
                        """
                        INSERT INTO experiment_metrics (
                            run_id,
                            metric,
                            value
                        )
                        VALUES (?, ?, ?)
                        """,
                        (
                            run_id,
                            metric,
                            float(value),
                        ),
                    )

                for compound_id, obs, pred in zip(
                    run.compound_ids,
                    run.observed,
                    run.predicted,
                ):
                    cursor.execute(
                        """
                        INSERT INTO
                        experiment_predictions (
                            run_id,
                            compound_id,
                            observed,
                            predicted
                        )
                        VALUES (?, ?, ?, ?)
                        """,
                        (
                            run_id,
                            int(compound_id),
                            float(obs),
                            float(pred),
                        ),
                    )

            for metric, statistics in self.summary_metrics.items():
                for statistic, value in statistics.items():
                    cursor.execute(
                        """
                        INSERT INTO experiment_summary_metrics (
                            experiment_id,
                            metric,
                            statistic,
                            value
                        )
                        VALUES (?, ?, ?, ?)
                        """,
                        (
                            experiment_id,
                            metric,
                            statistic,
                            float(value),
                        ),
                    )

            conn.commit()

    def save_artifacts(self, output_dir):
        output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        for run in self.runs:
            if run.training_loss is None:
                continue
            pd.DataFrame(
                {
                    "epoch": range(len(run.training_loss)),
                    "loss": run.training_loss,
                }
            ).to_csv(output_dir / f"loss_run_{run.run_number}.csv", index=False)

    @property
    def summary_metrics(self):
        summary_metrics = {}

        if not self.runs:
            return {}

        metric_names = self.runs[0].metrics.keys()

        for metric in metric_names:
            values = [run.metrics[metric] for run in self.runs]

            summary_metrics[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

        return summary_metrics


class ExperimentRunner:
    def __init__(self, ml_database):
        self.ml_database = ml_database

    def run(self, experiment: Experiment):
        run_results = []

        for run_number in range(experiment.cv_repeats):
            run_result = self.run_single(
                experiment, run_number=run_number, seed=run_number
            )
            run_results.append(run_result)

        return ExperimentResult(
            experiment=experiment,
            runs=run_results,
        )

    def run_single(
        self, experiment: Experiment, run_number: int, seed: int = 42
    ) -> ExperimentRunResult:
        loader = SQLiteDataLoader()

        dataset = loader.load(
            self.ml_database,
            split_name=experiment.split_name,
            target=experiment.target,
        )

        split_generator = SplitGenerator()

        if experiment.cv_strategy is None:
            cv_dataset = dataset
        elif experiment.cv_strategy == "repeated_holdout":
            cv_dataset = split_generator.generate(
                dataset,
                seed=seed,
            )
        else:
            raise ValueError(f"Unknown cv_strategy: {experiment.cv_strategy}")

        featurizer = FEATURIZER_REGISTRY[experiment.featurizer](
            **experiment.featurizer_parameters
        )

        representation = featurizer.transform(cv_dataset, target=experiment.target)

        training_options = experiment.training

        include_validation = training_options.get("include_validation", False)

        model_kwargs = dict(experiment.parameters)

        model_kwargs.pop("include_validation", False)

        model = MODEL_REGISTRY[experiment.model](**model_kwargs)

        model.fit(representation, include_validation=include_validation)

        metrics = model.evaluate(representation)
        metrics.update(model.get_complexity_metrics(representation))

        predictions = model.predict(representation)

        return ExperimentRunResult(
            run_number=run_number,
            random_seed=seed,
            split_name=experiment.split_name,
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
