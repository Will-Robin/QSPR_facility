import torch
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool
from src.model_base import GraphModel
from src.head import RegressionHead
from src.graph_neural_networks.graph_regressor import GraphRegressorNet
from torch import nn


class GINEdgeEncoderNet(torch.nn.Module):
    def __init__(
        self,
        num_node_features,
        num_edge_features,
        hidden_channels=64,
        embedding_dim=128,
        num_layers=2,
        dropout=0.1,
    ):
        super().__init__()


        self.dropout = nn.Dropout(dropout)
        self.convs = nn.ModuleList()

        in_dim = num_node_features
        for layer in range(num_layers):
            out_dim = embedding_dim if layer == num_layers - 1 else hidden_channels

            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, out_dim),
            )

            self.convs.append(GINEConv(mlp, edge_dim=num_edge_features))

            in_dim = out_dim

        self.embedding_dim = embedding_dim

    def forward(self, x, edge_index, edge_attr, batch):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr)

            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = self.dropout(x)

        return global_mean_pool(x, batch)


class GINEdgeRegressorModel(GraphModel):
    def __init__(
        self,
        num_layers=1,
        hidden_channels=64,
        embedding_dim=128,
        head_hidden_dim=64,
        head_layers=0,
        dropout=0.1,
        num_outputs=1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.embedding_dim = embedding_dim
        self.head_hidden_dim = head_hidden_dim
        self.head_layers = head_layers
        self.dropout = dropout
        self.num_outputs = num_outputs

    def build_network(self, dataset):
        encoder = GINEdgeEncoderNet(
            num_node_features=dataset.num_node_features,
            num_edge_features=dataset.num_edge_features,
            hidden_channels=self.hidden_channels,
            embedding_dim=self.embedding_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )

        head = RegressionHead(
            dim=self.embedding_dim,
            num_outputs=self.num_outputs,
            hidden_dim=self.head_hidden_dim,
            n_layers=self.head_layers,
            dropout=self.dropout,
        )

        return GraphRegressorNet(encoder, head)

    def forward(self, batch):
        return self.network(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
