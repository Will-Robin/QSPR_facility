A repository for reproducibly and data managed QSPR experiments.
No experiment is lost!
Information on models and their performance is stored in a database alongside data split information and the original source of data.

Running these in order should work.

1. `prepare_data.py`: clones the SurfPro2 repository and compiles a new copy with new tables in.
2. `prepare_train_test_splits.py`: prepares the train/test splits for insertion into the database
3. `train_and_evaluate_models.py`: trains and evaluates models defined in `./experiments`

These can be run any time:

1. `check_loss_curves.py`
2. `inspect_results.py`
3. `visualise_results.py`

If you want to start modelling from scratch after trying some things, this wipes all previous runs.
4. `experiment_cleaner.py`
