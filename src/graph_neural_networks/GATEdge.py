import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from src.model_base import GraphModel
from torch import nn


class GATEdgeNet(torch.nn.Module):
    def __init__(
        self,
        num_node_features,
        num_edge_features,
        hidden_channels=64,
        num_layers=2,
        heads=4,
    ):
        super().__init__()

        self.convs = nn.ModuleList()

        in_dim = num_node_features

        for _ in range(num_layers):
            self.convs.append(
                GATConv(
                    in_dim, hidden_channels, heads=heads, edge_dim=num_edge_features
                )
            )

            in_dim = hidden_channels * heads

        self.lin = Linear(in_dim, 1)

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        batch,
    ):
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)

            x = F.relu(x)

        x = global_mean_pool(x, batch)

        return self.lin(x)


class GATEdgeModel(GraphModel):
    def __init__(
        self,
        hidden_channels=64,
        num_layers=2,
        heads=4,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.heads = heads

    def build_network(self, dataset):
        return GATEdgeNet(
            num_node_features=dataset.num_node_features,
            num_edge_features=dataset.num_edge_features,
            hidden_channels=self.hidden_channels,
            num_layers=self.num_layers,
            heads=self.heads,
        )

    def forward(self, batch):
        return self.network(batch.x, batch.edge_index, batch.edge_attr, batch.batch)


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


class GATEdgeHeadNet(torch.nn.Module):
    def __init__(
        self,
        num_node_features,
        num_edge_features,
        num_layers=1,
        hidden_channels=64,
        embedding_dim=128,
        head_hidden_dim=64,
        dropout=0.1,
        heads=2,
        n_outputs=1,
    ):
        super().__init__()

        self.convs = nn.ModuleList()

        in_dim = num_node_features

        for layer in range(num_layers):
            if layer == num_layers - 1:
                self.convs.append(
                    GATConv(
                        in_dim,
                        embedding_dim,
                        heads=heads,
                        edge_dim=num_edge_features,
                        concat=False,
                    )
                )

                in_dim = embedding_dim

            else:
                self.convs.append(
                    GATConv(
                        in_dim, hidden_channels, heads=heads, edge_dim=num_edge_features
                    )
                )

                in_dim = hidden_channels * heads

        self.dropout = nn.Dropout(dropout)

        self.head = RegressionHead(
            n_outputs=n_outputs,
            hidden_dim=head_hidden_dim,
            dim=embedding_dim,
            dropout=dropout,
        )

    def encode(self, x, edge_index, edge_attr, batch):
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)

            x = F.relu(x)

            x = self.dropout(x)

        return global_mean_pool(x, batch)

    def forward(self, x, edge_index, edge_attr, batch):
        embedding = self.encode(x, edge_index, edge_attr, batch)

        return self.head(embedding)


class GATEdgeHeadModel(GraphModel):
    def __init__(
        self,
        num_layers=1,
        hidden_channels=64,
        embedding_dim=128,
        head_hidden_dim=64,
        dropout=0.1,
        heads=2,
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
        self.heads = heads

    def build_network(self, dataset):
        return GATEdgeHeadNet(
            num_node_features=dataset.num_node_features,
            num_edge_features=dataset.num_edge_features,
            num_layers=self.num_layers,
            hidden_channels=self.hidden_channels,
            embedding_dim=self.embedding_dim,
            head_hidden_dim=self.head_hidden_dim,
            dropout=self.dropout,
            heads=self.heads,
            n_outputs=self.n_outputs,
        )

    def forward(self, batch):
        return self.network(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
