import torch
from torch_geometric.nn.models import AttentiveFP

from src.head import RegressionHead
from src.graph_neural_networks.graph_regressor import GraphRegressorNet
from src.model_base import GraphModel


class AttentiveFPEncoderNet(torch.nn.Module):
    def __init__(
        self,
        num_node_features,
        num_edge_features,
        hidden_channels=64,
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

    def forward(self, x, edge_index, edge_attr, batch):
        embeddings = self.encoder(x, edge_index, edge_attr, batch)

        return embeddings


class AttentiveFPRegressorModel(GraphModel):
    def __init__(
        self,
        hidden_channels=64,
        head_hidden_dim=64,
        out_channels=128,
        num_layers=2,
        num_timesteps=2,
        head_layers=0,
        dropout=0.1,
        num_outputs=1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_channels = hidden_channels
        self.head_hidden_dim = head_hidden_dim
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        self.dropout = dropout
        self.num_outputs = num_outputs
        self.out_channels = out_channels
        self.head_layers = head_layers

    def build_network(self, dataset):
        encoder = AttentiveFPEncoderNet(
            num_node_features=dataset.num_node_features,
            num_edge_features=dataset.num_edge_features,
            hidden_channels=self.hidden_channels,
            out_channels=self.out_channels,
            num_layers=self.num_layers,
            num_timesteps=self.num_timesteps,
            dropout=self.dropout,
        )

        head = RegressionHead(
            dim=self.out_channels,
            num_outputs=self.num_outputs,
            hidden_dim=self.head_hidden_dim,
            n_layers=self.head_layers,
            dropout=self.dropout,
        )

        return GraphRegressorNet(encoder, head)

    def forward(self, batch):
        return self.network(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
