from src.ml_models import (
    LinearRegressionModel,
    RidgeRegressionModel,
    LassoRegressionModel,
    RandomForestModel,
    SVRModel,
    MLPModel,
)

from src.graph_neural_networks.AttentiveFP import AttentiveFPRegressorModel
from src.graph_neural_networks.GCN import GCNRegressorModel
from src.graph_neural_networks.GIN import GINRegressorModel
from src.graph_neural_networks.GINE import GINEdgeRegressorModel
from src.graph_neural_networks.GAT import GATRegressorModel
from src.graph_neural_networks.GATEdge import GATEdgeRegressorModel

from src.dataloader import ECFPFeaturizer, DescriptorFeaturizer, GraphFeaturizer

MODEL_REGISTRY = {
    "linear": LinearRegressionModel,
    "ridge": RidgeRegressionModel,
    "lasso": LassoRegressionModel,
    "random_forest": RandomForestModel,
    "svr": SVRModel,
    "mlp": MLPModel,
    "attentivefp": AttentiveFPRegressorModel,
    "gcn": GCNRegressorModel,
    "gin": GINRegressorModel,
    "gine": GINEdgeRegressorModel,
    "gat": GATRegressorModel,
    "gat_edge": GATEdgeRegressorModel,
}


FEATURIZER_REGISTRY = {
    "ecfp": ECFPFeaturizer,
    "descriptor": DescriptorFeaturizer,
    "graph": GraphFeaturizer,
}
