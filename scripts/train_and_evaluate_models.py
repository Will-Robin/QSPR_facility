import toml
from pathlib import Path
from src.experiment import Experiment, ExperimentRunner


def main():
    # config
    config = toml.loads(Path("config.toml").read_text())

    ml_database = config["ML_DATABASE"]

    # experiment
    experiment = Experiment.from_toml("experiments/attfp_head_graph_pcmc.toml")

    # run
    runner = ExperimentRunner(ml_database)

    result = runner.run(experiment)

    # persist
    result.save_to_database(ml_database)

    output_dir = Path("outputs") / f"{result.experiment.experiment_hash}"
    output_config = output_dir / "experiment.toml"
    result.save_artifacts(output_dir)
    output_config.write_text(experiment.raw_toml)


if __name__ == "__main__":
    main()
