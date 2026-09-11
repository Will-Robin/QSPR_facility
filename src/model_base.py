from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
)


class Model:
    input_type = None

    estimator = None

    TRAINING_PARAMETERS = set()

    def fit(self, dataset, include_validation=False):
        X_train, y_train = dataset.get_training_data(
            include_validation=include_validation
        )

        self.estimator.fit(
            X_train,
            y_train,
        )

        return self

    def predict(self, dataset):
        return self.estimator.predict(dataset.X_test)

    def evaluate(self, dataset):
        y_true = dataset.test_targets

        y_pred = self.predict(dataset)

        return {
            "mae": mean_absolute_error(y_true, y_pred),
            "rmse": root_mean_squared_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
        }
