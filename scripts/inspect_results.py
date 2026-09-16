import toml
import sqlite3
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

leader_board_query = """
SELECT
    e.model,
    e.parameters_json,
    COUNT(*) AS n_experiments,
    AVG(esm.value) AS mean_rmse,
    MIN(esm.value) AS best_rmse,
    MAX(esm.value) AS worst_rmse
FROM experiment_summary_metrics esm
JOIN experiments e
    USING (experiment_id)
WHERE
    esm.metric = 'rmse'
    AND esm.statistic = 'mean'
    AND e.split_name = 'leave_Si_out'
GROUP BY e.model
ORDER BY mean_rmse ASC;
"""


def main():
    config = toml.loads(Path("config.toml").read_text())

    ML_DATABASE = Path(config["ML_DATABASE"])

    with sqlite3.connect(ML_DATABASE) as conn:
        df = pd.read_sql_query(
            leader_board_query,
            conn,
        )

    print(df.head())
    fig, ax = plt.subplots()
    ax.hist(df.mean_rmse)
    plt.show()


if __name__ == "__main__":
    main()
