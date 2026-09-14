import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool
from src.model_base import GraphModel
from torch import nn


class GINNet(torch.nn.Module):
    def __init__(
        self,
        num_node_features,
        hidden_channels=64,
        num_layers=2,
    ):
        super().__init__()

        self.convs = nn.ModuleList()

        in_dim = num_node_features

        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            )

            self.convs.append(GINConv(mlp))

            in_dim = hidden_channels

        self.lin = Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)

            x = F.relu(x)

        x = global_mean_pool(x, batch)

        return self.lin(x)


class GINModel(GraphModel):
    def __init__(
        self,
        hidden_channels=64,
        num_layers=2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

    def build_network(self, dataset):
        return GINNet(
            num_node_features=dataset.num_node_features,
            hidden_channels=self.hidden_channels,
            num_layers=self.num_layers,
        )

    def forward(self, batch):
        return self.network(batch.x, batch.edge_index, batch.batch)


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


class GINHeadNet(torch.nn.Module):
    def __init__(
        self,
        num_node_features,
        num_layers=1,
        hidden_channels=64,
        embedding_dim=128,
        head_hidden_dim=64,
        dropout=0.1,
        n_outputs=1,
    ):
        super().__init__()

        self.convs = nn.ModuleList()

        in_dim = num_node_features

        for layer in range(num_layers):
            out_dim = embedding_dim if layer == num_layers - 1 else hidden_channels

            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, out_dim),
            )

            self.convs.append(GINConv(mlp))

            in_dim = out_dim

        self.dropout = nn.Dropout(dropout)

        self.head = RegressionHead(
            n_outputs=n_outputs,
            hidden_dim=head_hidden_dim,
            dim=embedding_dim,
            dropout=dropout,
        )

    def encode(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)

            x = F.relu(x)

            x = self.dropout(x)

        return global_mean_pool(x, batch)

    def forward(self, x, edge_index, batch):
        embedding = self.encode(x, edge_index, batch)

        return self.head(embedding)


class GINHeadModel(GraphModel):
    def __init__(
        self,
        num_layers=1,
        hidden_channels=64,
        embedding_dim=128,
        head_hidden_dim=64,
        dropout=0.1,
        n_outputs=1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.embedding_dim = embedding_dim
        self.head_hidden_dim = head_hidden_dim
        self.dropout = dropout
        self.n_outputs = n_outputs

    def build_network(self, dataset):
        return GINHeadNet(
            num_node_features=dataset.num_node_features,
            num_layers=self.num_layers,
            hidden_channels=self.hidden_channels,
            embedding_dim=self.embedding_dim,
            head_hidden_dim=self.head_hidden_dim,
            dropout=self.dropout,
            n_outputs=self.n_outputs,
        )

    def forward(self, batch):
        return self.network(batch.x, batch.edge_index, batch.batch)
