import toml
import sqlite3
import pandas as pd
from pathlib import Path

leader_board_query = """
SELECT
	e.experiment_name,
	e.split_name,
	e.featurizer,
	e.parameters_json,
	em.metric,
	em.value
FROM experiments e
JOIN experiment_metrics em USING(experiment_id)
WHERE em.metric = 'r2'
ORDER BY em.value DESC;
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

if __name__ == "__main__":
    main()
