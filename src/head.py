from torch import nn


class RegressionHead(nn.Module):
    def __init__(
        self, dim, num_outputs=1, hidden_dim=None, n_layers=0, dropout=0.1, use_norm=True
    ):
        super().__init__()

        if n_layers > 0 and hidden_dim is None:
            raise ValueError("hidden_dim must be specified when n_layers > 0")

        layers = []

        if use_norm:
            layers.append(nn.LayerNorm(dim))

        in_dim = dim

        for _ in range(n_layers):
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )

            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, num_outputs))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
