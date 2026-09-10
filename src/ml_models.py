from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
# from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from src.model_base import Model
from src.dataloader import VectorDataset

class TabularModel(Model):
    input_type = VectorDataset

    estimator_class = None

    def __init__(self, **kwargs):
        self.estimator = self.estimator_class(**kwargs)


class LinearRegressionModel(TabularModel):
    estimator_class = LinearRegression


class RidgeRegressionModel(TabularModel):
    estimator_class = Ridge


class LassoRegressionModel(TabularModel):
    estimator_class = Lasso


"""
class XGBoostModel(
    TabularModel
):
    estimator_class = XGBRegressor
"""


class RandomForestModel(TabularModel):
    estimator_class = RandomForestRegressor


class SVRModel(TabularModel):
    estimator_class = SVR


class MLPModel(TabularModel):
    estimator_class = MLPRegressor
