import toml
from pathlib import Path
from src.experiment import Experiment, ExperimentRunner, experiment_exists


def run_experiment(experiment, ml_database):
    # Run experiment
    runner = ExperimentRunner(ml_database)
    result = runner.run(experiment)

    # Save results
    result.save_to_database(ml_database)
    output_dir = Path("outputs") / f"{result.experiment.experiment_hash}"
    output_dir.mkdir(exist_ok=True)
    output_config = output_dir / "experiment.toml"
    result.save_artifacts(output_dir)
    output_config.write_text(experiment.raw_toml)


def run_design(filename, ml_database):
    experiment = Experiment.from_toml(filename)

    if experiment_exists(ml_database, experiment):
        print(f"Skipping {experiment.experiment_name}.")
    else:
        print(f"Running {experiment.experiment_name}.")
        run_experiment(experiment, ml_database)


def main():
    config = toml.loads(Path("config.toml").read_text())
    ml_database = config["ML_DATABASE"]

    design_files = [
        file for file in Path("experiments").iterdir() if file.suffix == ".toml"
    ]

    design_files.sort()
    for des_file in design_files:
        print(des_file)
        run_design(des_file, ml_database)


if __name__ == "__main__":
    main()
