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
    experiment_hash TEXT,
    source_file TEXT,
    split_name TEXT,
    target TEXT,
    featurizer TEXT,
    featurizer_parameters_json TEXT,
    model TEXT,
    parameters_json TEXT,
    cv_strategy TEXT,
    cv_repeats INTEGER DEFAULT 1,
    raw_toml TEXT,
    training_json TEXT,
    created_at TEXT,

    UNIQUE (experiment_hash)

);

CREATE TABLE IF NOT EXISTS experiment_metrics (
    run_id INTEGER,
    metric TEXT,
    value REAL,

    UNIQUE (run_id, metric),

    FOREIGN KEY (run_id)
        REFERENCES experiment_runs(run_id)
);

CREATE TABLE IF NOT EXISTS experiment_predictions (
    run_id INTEGER,
    compound_id INTEGER,
    observed REAL,
    predicted REAL,

    UNIQUE (run_id, compound_id),

    FOREIGN KEY (run_id)
       REFERENCES experiment_runs(run_id)
);

CREATE TABLE IF NOT EXISTS split_definitions (
    split_id INTEGER,
    split_name TEXT PRIMARY KEY,
    split_hash TEXT,
    split_strategy TEXT,
    split_definition_json TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS experiment_runs (
    run_id INTEGER PRIMARY KEY,

    experiment_id INTEGER NOT NULL,

    run_number INTEGER NOT NULL,

    split_name TEXT,
    random_seed INTEGER,

    created_at TEXT,

    UNIQUE (experiment_id, run_number),

    FOREIGN KEY (experiment_id)
        REFERENCES experiments(experiment_id)

);

CREATE TABLE IF NOT EXISTS experiment_summary_metrics (
    experiment_id INTEGER,
    metric TEXT,
    statistic TEXT,
    value REAL,

    UNIQUE (experiment_id, metric, statistic),

    FOREIGN KEY (experiment_id)
        REFERENCES experiments(experiment_id)
);

CREATE UNIQUE INDEX idx_experiment_hash ON experiments(experiment_hash);
