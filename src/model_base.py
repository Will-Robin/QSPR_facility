import torch
from src.dataloader import GraphDataset
from torch_geometric.loader import DataLoader
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


class GraphModel(Model):
    input_type = GraphDataset

    TRAINING_PARAMETERS = {
        "learning_rate",
        "batch_size",
        "n_epochs",
    }

    def __init__(
        self,
        batch_size=32,
        n_epochs=100,
        learning_rate=1e-3,
        device=None,
    ):
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.training_loss = []

        self.device = (
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.loss_fn = torch.nn.MSELoss()

    def make_train_loader(self, dataset, include_validation=False):
        return DataLoader(
            dataset.get_training_graphs(include_validation=include_validation),
            batch_size=self.batch_size,
            shuffle=True,
        )

    def make_val_loader(self, dataset):
        return DataLoader(
            dataset.val_graphs,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def make_test_loader(self, dataset):
        return DataLoader(
            dataset.get_test_graphs(),
            batch_size=self.batch_size,
            shuffle=False,
        )

    def build_network(self, dataset):
        raise NotImplementedError

    def forward(self, batch):
        raise NotImplementedError

    def fit(self, dataset, include_validation=False):
        self.training_loss = []

        train_loader = self.make_train_loader(
            dataset, include_validation=include_validation
        )

        self.network = self.build_network(dataset).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.learning_rate,
        )

        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            self.network.train()

            for batch in train_loader:
                batch = batch.to(self.device)

                pred = self.forward(batch)

                loss = self.loss_fn(
                    pred.squeeze(),
                    batch.y.squeeze(),
                )

                epoch_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.training_loss.append(epoch_loss / len(train_loader))

        return self

    def predict(self, dataset):
        loader = self.make_test_loader(dataset)

        predictions = []

        self.network.eval()

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)

                pred = self.forward(batch)

                predictions.append(pred.squeeze().cpu())

        return torch.cat(predictions).numpy()
