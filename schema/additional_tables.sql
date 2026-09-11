CREATE TABLE IF NOT EXISTS ml_splits (
    split_name TEXT NOT NULL,
    compound_id INTEGER NOT NULL,
    split_type TEXT NOT NULL,
    split_strategy TEXT NOT NULL,
    random_seed INTEGER,
    split_hash TEXT,

    PRIMARY KEY (split_name, compound_id),
    FOREIGN KEY (compound_id)
        REFERENCES compounds(compound_id)
);

CREATE TABLE IF NOT EXISTS experiments (
    experiment_id INTEGER PRIMARY KEY,
    experiment_name TEXT,
    experiment_hash TEXT UNIQUE,
    source_file TEXT,
    split_name TEXT,
    target TEXT,
    featurizer TEXT,
    featurizer_parameters_json TEXT,
    model TEXT,
    parameters_json TEXT,
    raw_toml TEXT,
    training_json TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS experiment_metrics (
    experiment_id INTEGER,
    metric TEXT,
    value REAL,

    FOREIGN KEY (experiment_id)
        REFERENCES experiments(experiment_id)
);

CREATE TABLE IF NOT EXISTS experiment_predictions (
    experiment_id INTEGER,
    compound_id INTEGER,
    observed REAL,
    predicted REAL,

    FOREIGN KEY (experiment_id)
       REFERENCES experiments(experiment_id)
);

CREATE TABLE IF NOT EXISTS split_definitions (
    split_id INTEGER,
    split_name TEXT PRIMARY KEY,
    split_hash TEXT,
    split_strategy TEXT,
    split_definition_json TEXT,
    created_at TEXT
);

CREATE UNIQUE INDEX idx_experiment_hash ON experiments(experiment_hash);
