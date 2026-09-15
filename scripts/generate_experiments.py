import json
import toml
from itertools import product
from pathlib import Path

MODEL_FEATURIZERS = {
    "linear": {"descriptor", "ecfp"},
    "ridge": {"descriptor", "ecfp"},
    "lasso": {"descriptor", "ecfp"},
    "random_forest": {"descriptor", "ecfp"},
    "svr": {"descriptor", "ecfp"},
    "gcn": {"graph"},
    "gin": {"graph"},
    "gin_edge": {"graph"},
    "gat": {"graph"},
    "gat_edge": {"graph"},
    "attentivefp": {"graph"},
}

TRAINABLE_MODELS = {
    "gcn",
    "gin",
    "gin_edge",
    "gat",
    "gat_edge",
    "attentivefp",
}


def expand_grid(grid):
    keys = list(grid.keys())
    values = list(grid.values())

    if not keys:
        return [{}]

    return [dict(zip(keys, combo)) for combo in product(*values)]


def calculate_number_of_designs(config):
    n_experiments = 0

    for task in config["tasks"]:
        for split in config["splits"]:
            for representation_name, representation_grid in config[
                "representations"
            ].items():
                for _ in expand_grid(representation_grid):
                    for model_name, model_design in config["model_designs"].items():
                        if representation_name not in MODEL_FEATURIZERS[model_name]:
                            continue

                        active_training = (
                            expand_grid(config["training"])
                            if model_name in TRAINABLE_MODELS
                            else [{}]
                        )

                        n_experiments += len(
                            expand_grid(model_design["parameter_grid"])
                        ) * len(active_training)

    return n_experiments


config = json.loads(Path("experiments/EXPERIMENTS.json").read_text())

output_dir = Path("experiments")
output_dir.mkdir(exist_ok=True)

n_experiments = calculate_number_of_designs(config)
MAX_EXPERIMENTS = 10000

if n_experiments > MAX_EXPERIMENTS:
    raise RuntimeError(
        f"Refusing to generate {n_experiments:,} experiments "
        f"(limit={MAX_EXPERIMENTS:,})"
    )
else:
    print(f"Generating {n_experiments} experiment designs.")

counter = 1

for task in config["tasks"]:
    for split in config["splits"]:
        for representation_name, representation_grid in config[
            "representations"
        ].items():
            for featurizer_parameters in expand_grid(representation_grid):
                for model_name, model_design in config["model_designs"].items():
                    if representation_name not in MODEL_FEATURIZERS[model_name]:
                        continue

                    if model_name in TRAINABLE_MODELS:
                        active_training = expand_grid(config["training"])
                    else:
                        active_training = [{}]

                    for parameters in expand_grid(model_design["parameter_grid"]):
                        for training in active_training:
                            experiment_name = f"{model_name}_{counter:04d}"
                            experiment = {
                                "experiment_name": experiment_name,
                                "split_name": split,
                                "target": task,
                                "featurizer": representation_name,
                                "featurizer_parameters": featurizer_parameters,
                                "model": model_name,
                                "parameters": parameters,
                                "training": training,
                            }

                            output_file = output_dir / f"{experiment_name}.toml"

                            if output_file.exists():
                                raise RuntimeError(f"{output_file} already exists")

                            with open(output_file, "w") as f:
                                toml.dump(experiment, f)

                            counter += 1
