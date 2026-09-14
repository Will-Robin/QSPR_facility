import toml
import sqlite3
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

query = """
SELECT
    ep.observed,
    ep.predicted,
    e.experiment_id,
    e.experiment_hash,
    e.experiment_name
FROM experiment_predictions ep
JOIN experiments e USING(experiment_id)
"""

top_rmse_ranking = """
SELECT
    e.model,
    e.parameters_json,
    e.experiment_hash,
    e.experiment_id,
    em.value as 'root mean squared error'
FROM experiment_metrics em
JOIN experiments e USING(experiment_id)
WHERE em.metric = "rmse"
ORDER BY em.value
LIMIT 5;
"""


def main():
    config = toml.loads(Path("config.toml").read_text())

    ML_DATABASE = Path(config["ML_DATABASE"])

    with sqlite3.connect(ML_DATABASE) as conn:
        df = pd.read_sql_query(query, conn)

        top_df = pd.read_sql_query(top_rmse_ranking, conn)
        top_n = top_df.experiment_id.to_list()

        fig, ax = plt.subplots(ncols=len(top_n))
        for c, n in enumerate(top_n):
            group = df[df.experiment_id == n]
            if group.experiment_id.iloc[0] in top_n:
                ax[c].scatter(
                    group.observed,
                    group.predicted,
                    label=f"{c + 1} " + group.experiment_name.iloc[0],
                    alpha=0.5,
                )
            else:
                pass

            ax[c].plot(group.observed, group.observed, "--", c="k")
            ax[c].legend()
        plt.show()


if __name__ == "__main__":
    main()
