from src.ml_models import (
    LinearRegressionModel,
    RidgeRegressionModel,
    LassoRegressionModel,
    RandomForestModel,
    SVRModel,
    MLPModel,
)

from src.graph_neural_networks.AttentiveFP import AttentiveFPModel, AttentiveFPHeadModel

from src.graph_neural_networks.GCN import GCNModel, GCNHeadModel

from src.graph_neural_networks.GIN import GINModel, GINHeadModel

from src.dataloader import ECFPFeaturizer, DescriptorFeaturizer, GraphFeaturizer

MODEL_REGISTRY = {
    "linear": LinearRegressionModel,
    "ridge": RidgeRegressionModel,
    "lasso": LassoRegressionModel,
    "random_forest": RandomForestModel,
    "svr": SVRModel,
    "mlp": MLPModel,
    "attentivefp": AttentiveFPModel,
    "attentivefp_head": AttentiveFPHeadModel,
    "gcn": GCNModel,
    "gcn_head": GCNHeadModel,
    "gin": GINModel,
    "gin_head": GINHeadModel,
}


FEATURIZER_REGISTRY = {
    "ecfp": ECFPFeaturizer,
    "descriptor": DescriptorFeaturizer,
    "graph": GraphFeaturizer,
}
