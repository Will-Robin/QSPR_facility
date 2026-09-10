from src.ml_models import (
    LinearRegressionModel,
    RidgeRegressionModel,
    LassoRegressionModel,
    RandomForestModel,
    SVRModel,
    MLPModel,
)

from src.graph_models import (
    AttentiveFPModel,
    AttentiveFPHeadModel,
    GCNModel,
)

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
}


FEATURIZER_REGISTRY = {
    "ecfp": ECFPFeaturizer,
    "descriptor": DescriptorFeaturizer,
    "graph": GraphFeaturizer,
}
