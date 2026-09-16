import toml
import sqlite3
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

SPLIT_STRATEGY = "leave_B_Si_out"
METRIC = "rmse"

query = f"""
SELECT
    ep.observed,
    ep.predicted,
    r.run_number,
    e.experiment_id,
    esm.value,
    e.experiment_name
FROM experiment_predictions ep
JOIN experiment_runs r
    ON ep.run_id = r.run_id
JOIN experiments e
    ON r.experiment_id = e.experiment_id
JOIN experiment_summary_metrics esm
    ON esm.experiment_id = e.experiment_id
WHERE
    e.experiment_id IN (
        SELECT
            e.experiment_id
        FROM experiments e
        JOIN experiment_summary_metrics esm
            ON e.experiment_id = esm.experiment_id
        WHERE
            e.split_name = '{SPLIT_STRATEGY}'
            AND esm.metric = '{METRIC}'
            AND esm.statistic = 'mean'
        ORDER BY
            esm.value
        LIMIT 5
    )
    AND esm.statistic = 'mean'
    AND esm.metric = '{METRIC}'
ORDER BY
    esm.value ASC
"""


def main():
    config = toml.loads(Path("config.toml").read_text())

    ML_DATABASE = Path(config["ML_DATABASE"])

    with sqlite3.connect(ML_DATABASE) as conn:
        df = pd.read_sql_query(query, conn)

        top_ids = df["experiment_id"].unique()

        fig, ax = plt.subplots(ncols=len(top_ids))
        for ax, experiment_id in zip(ax, top_ids):
            group = df[df["experiment_id"] == experiment_id]
            ax.set_title(f"{group.experiment_name.iloc[0]}")
            for run, group in group.groupby("run_number"):
                ax.scatter(
                    group.observed, group.predicted, alpha=0.5, edgecolor="none"
                )

            ax.plot(group.observed, group.observed, "--", c="k")
            ax.legend()

        plt.show()


if __name__ == "__main__":
    main()
