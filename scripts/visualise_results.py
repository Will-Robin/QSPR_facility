import toml
import sqlite3
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

query = """
SELECT
    ep.observed,
    ep.predicted,
    e.experiment_hash,
    e.experiment_name
FROM experiment_predictions ep
JOIN experiments e USING(experiment_id)
"""


def main():
    config = toml.loads(Path("config.toml").read_text())

    ML_DATABASE = Path(config["ML_DATABASE"])

    with sqlite3.connect(ML_DATABASE) as conn:
        df = pd.read_sql_query(
            query,
            conn,
        )

        fig, ax = plt.subplots()
        for c, group in df.groupby("experiment_hash"):
            ax.scatter(
                group.observed, group.predicted, label=group.experiment_name.iloc[0]
            )
        ax.plot(group.observed, group.observed, "--", c="k")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
