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
from src.graph_neural_networks.GINEdge import GINEdgeRegressorModel
from src.graph_neural_networks.GAT import GATRegressorModel
from src.graph_neural_networks.GATEdge import GATEdgeRegressorModel

from src.split_strategies import RandomSplitStrategy
from src.split_strategies import StratifiedSplitStrategy
from src.split_strategies import MurckoScaffoldSplitStrategy
from src.split_strategies import ElementHoldoutSplitStrategy

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
    "gin_edge": GINEdgeRegressorModel,
    "gat": GATRegressorModel,
    "gat_edge": GATEdgeRegressorModel,
}


FEATURIZER_REGISTRY = {
    "ecfp": ECFPFeaturizer,
    "descriptor": DescriptorFeaturizer,
    "graph": GraphFeaturizer,
}

SPLIT_STRATEGIES = {
    "random": RandomSplitStrategy,
    "stratified": StratifiedSplitStrategy,
    "murcko_scaffold": MurckoScaffoldSplitStrategy,
    "element_holdout": ElementHoldoutSplitStrategy,
}
