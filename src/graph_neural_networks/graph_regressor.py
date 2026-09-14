from torch import nn


class GraphRegressorNet(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()

        self.encoder = encoder
        self.head = head

    def forward(self, *encoder_args):
        embedding = self.encoder(*encoder_args)

        return self.head(embedding)
