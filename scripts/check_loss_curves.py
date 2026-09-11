import toml
import sqlite3
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

query = """
SELECT DISTINCT
    experiment_hash
FROM experiments
WHERE split_name = "surfactant_type_v1";
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

        for i, row in df.iterrows():
            loss_path = Path("outputs") / row.experiment_hash / "loss.csv"

            if loss_path.is_file():
                df_loss = pd.read_csv(loss_path)
                ax.plot(df_loss.epoch, df_loss.loss, label=row.experiment_hash[:5])
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
