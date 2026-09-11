import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from src.model_base import GraphModel
from torch import nn


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


class GCNHeadNet(torch.nn.Module):
    def __init__(
        self,
        num_node_features,
        hidden_channels=64,
        embedding_dim=128,
        head_hidden_dim=64,
        dropout=0.1,
        n_outputs=1,
    ):
        super().__init__()

        self.conv1 = GCNConv(num_node_features, hidden_channels)

        self.conv2 = GCNConv(hidden_channels, embedding_dim)

        self.dropout = nn.Dropout(dropout)

        self.head = RegressionHead(
            n_outputs=n_outputs,
            hidden_dim=head_hidden_dim,
            dim=embedding_dim,
            dropout=dropout,
        )

    def encode(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)

        return global_mean_pool(x, batch)

    def forward(self, x, edge_index, batch):
        embedding = self.encode(x, edge_index, batch)

        return self.head(embedding)


class GCNHeadModel(GraphModel):
    def __init__(
        self,
        hidden_channels=64,
        embedding_dim=128,
        head_hidden_dim=64,
        dropout=0.1,
        n_outputs=1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_channels = hidden_channels
        self.embedding_dim = embedding_dim
        self.head_hidden_dim = head_hidden_dim
        self.dropout = dropout
        self.n_outputs = n_outputs

    def build_network(self, dataset):
        return GCNHeadNet(
            num_node_features=dataset.num_node_features,
            hidden_channels=self.hidden_channels,
            embedding_dim=self.embedding_dim,
            head_hidden_dim=self.head_hidden_dim,
            dropout=self.dropout,
            n_outputs=self.n_outputs,
        )

    def forward(self, batch):
        return self.network(batch.x, batch.edge_index, batch.batch)
