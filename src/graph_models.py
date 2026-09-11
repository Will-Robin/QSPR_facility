import torch

from src.model_base import Model
from src.dataloader import GraphDataset

from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch import nn

from torch.nn import Linear

from torch_geometric.nn import GCNConv, global_mean_pool

from torch_geometric.nn.models import AttentiveFP


class GraphModel(Model):
    input_type = GraphDataset

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


class GCNNet(torch.nn.Module):
    def __init__(
        self,
        num_node_features,
        hidden_channels=64,
    ):
        super().__init__()

        self.conv1 = GCNConv(num_node_features, hidden_channels)

        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        self.lin = Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)

        x = F.relu(x)

        x = self.conv2(x, edge_index)

        x = F.relu(x)

        x = global_mean_pool(x, batch)

        return self.lin(x)


class GCNModel(GraphModel):
    def __init__(
        self,
        hidden_channels=64,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_channels = hidden_channels

    def build_network(self, dataset):
        return GCNNet(
            num_node_features=dataset.num_node_features,
            hidden_channels=self.hidden_channels,
        )

    def forward(self, batch):
        return self.network(
            batch.x,
            batch.edge_index,
            batch.batch,
        )


class AttentiveFPModel(GraphModel):
    def __init__(
        self,
        hidden_channels=64,
        out_channels=1,
        num_layers=2,
        num_timesteps=2,
        dropout=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        self.dropout = dropout

    def build_network(
        self,
        dataset,
    ):
        return AttentiveFP(
            in_channels=dataset.num_node_features,
            hidden_channels=self.hidden_channels,
            out_channels=self.out_channels,
            edge_dim=dataset.num_edge_features,
            num_layers=self.num_layers,
            num_timesteps=self.num_timesteps,
            dropout=self.dropout,
        )

    def forward(self, batch):
        return self.network(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
        )


class RegressionHead(torch.nn.Module):
    def __init__(
        self,
        n_outputs,
        hidden_dim,
        dropout,
        dim,
    ):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        self.norm = nn.LayerNorm(dim)

        self.fc1 = nn.Linear(dim, hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.fc3 = nn.Linear(hidden_dim, n_outputs)

    def forward(self, x):
        x = self.norm(x)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        return self.fc3(x)


class AttentiveFPHeadNet(torch.nn.Module):
    def __init__(
        self,
        num_node_features,
        num_edge_features,
        n_outputs=1,
        hidden_channels=64,
        head_hidden_dim=64,
        out_channels=128,
        num_layers=2,
        num_timesteps=2,
        dropout=0.1,
    ):
        super().__init__()

        self.encoder = AttentiveFP(
            in_channels=num_node_features,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            edge_dim=num_edge_features,
            num_layers=num_layers,
            num_timesteps=num_timesteps,
            dropout=dropout,
        )

        self.head = RegressionHead(
            n_outputs=n_outputs,
            hidden_dim=head_hidden_dim,
            dim=out_channels,
            dropout=dropout,
        )

    def forward(self, x, edge_index, edge_attr, batch):
        embeddings = self.encoder(x, edge_index, edge_attr, batch)

        return self.head(embeddings)

    def encode(self, x, edge_index, edge_attr, batch):
        """
        embeddings = model.network.encode(...)
        """
        return self.encoder(
            x,
            edge_index,
            edge_attr,
            batch,
        )


class AttentiveFPHeadModel(GraphModel):
    def __init__(
        self,
        hidden_channels=64,
        head_hidden_dim=64,
        out_channels=128,
        num_layers=2,
        num_timesteps=2,
        dropout=0.1,
        n_outputs=1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_channels = hidden_channels
        self.head_hidden_dim = head_hidden_dim
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        self.dropout = dropout
        self.n_outputs = n_outputs

    def build_network(self, dataset):
        return AttentiveFPHeadNet(
            num_node_features=dataset.num_node_features,
            num_edge_features=dataset.num_edge_features,
            n_outputs=self.n_outputs,
            hidden_channels=self.hidden_channels,
            head_hidden_dim=self.head_hidden_dim,
            out_channels=self.out_channels,
            num_layers=self.num_layers,
            num_timesteps=self.num_timesteps,
            dropout=self.dropout,
        )

    def forward(self, batch):
        return self.network(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
