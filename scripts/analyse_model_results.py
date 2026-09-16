import toml
import sqlite3
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


def main():
    config = toml.loads(Path("config.toml").read_text())

    ML_DATABASE = Path(config["ML_DATABASE"])

    conn = sqlite3.connect(ML_DATABASE)
    analysis_1(conn)
    analysis_2(conn)


def analysis_1(conn):
    df = pd.read_sql_query(
        """
        SELECT
            e.experiment_id,
            e.split_name,
            e.model,
            er.run_number,
            em.value AS rmse
        FROM experiments e
        JOIN experiment_runs er
            ON e.experiment_id = er.experiment_id
        JOIN experiment_metrics em
            ON er.run_id = em.run_id
        WHERE
            em.metric = 'rmse'
            AND e.model != "linear"
        """,
        conn,
    )

    split_num = df.split_name.unique()
    fig, ax = plt.subplots(ncols=len(split_num), figsize=(10, 6), sharey=True)

    for a, (c, group) in zip(ax, df.groupby("split_name")):
        group.boxplot(column="rmse", by="model", ax=a)
        a.set_ylabel("RMSE")
        a.set_xlabel("Model")
        a.set_title(f"{c} RMSE")
    plt.tight_layout()
    plt.show()


def analysis_2(conn):
    df = pd.read_sql_query(
        """
            SELECT
                e.experiment_id,
                e.model,
                esm.value AS mean_rmse
            FROM experiments e
            JOIN experiment_summary_metrics esm
                ON e.experiment_id = esm.experiment_id
            WHERE
                esm.metric = 'rmse'
                AND esm.statistic = 'mean'
                AND e.model != "linear";
            """,
        conn,
    )

    fig, ax = plt.subplots()
    for model, group in df.groupby("model"):
        ax.bar(group.experiment_id, group.mean_rmse, label=model)
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
