import random
import json
import toml
from itertools import product
from pathlib import Path

random.seed(42)

N_EXPERIMENTS_PER_MODEL = 10

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


config = json.loads(Path("experiments/EXPERIMENTS.json").read_text())

output_dir = Path("experiments")
output_dir.mkdir(exist_ok=True)


counter = 1

for task in config["tasks"]:
    for split in config["splits"]:
        for representation_name, representation_grid in config[
            "representations"
        ].items():
            for model_name, model_design in config["model_designs"].items():
                if representation_name not in MODEL_FEATURIZERS[model_name]:
                    continue

                rep_configs = expand_grid(representation_grid)
                param_configs = expand_grid(model_design["parameter_grid"])

                all_combinations = [
                    (rep, params) for rep in rep_configs for params in param_configs
                ]

                selected = random.sample(
                    all_combinations,
                    min(
                        len(all_combinations),
                        N_EXPERIMENTS_PER_MODEL,
                    ),
                )

                for featurizer_parameters, parameters in selected:
                    if model_name in TRAINABLE_MODELS:
                        training_configs = expand_grid(config["training"])
                    else:
                        training_configs = [{}]

                    for training in training_configs:
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
                            "cv_strategy": "repeated_holdout",
                            "cv_repeats": 5,
                        }

                        output_file = output_dir / f"{experiment_name}.toml"

                        if not output_file.exists():
                            with open(output_file, "w") as f:
                                toml.dump(experiment, f)

                        counter += 1
