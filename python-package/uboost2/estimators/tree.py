import typing
import numpy as np

from .base import BaseEstimator
from ..core import _core

__DMatrix_dict = dict()


def maybe_numpyToDMatrix(x):
    if isinstance(x, np.ndarray):
        return _core.numpyToDMatrix(x.copy())
    return x


def maybe_numpyToDColumn(x):
    if isinstance(x, np.ndarray):
        return _core.numpyToDColumn(x.copy())
    return x


class AbstractTreeRegressor:
    pass


def predict_many(trees: typing.List[AbstractTreeRegressor], x: np.ndarray) -> typing.List[np.ndarray]:
    x_ = maybe_numpyToDMatrix(x)
    outs = []
    for tree in trees:
        out = np.zeros((x.shape[0],))
        out_ = tree._handle.predict_value(x_)
        _core.DColumntoNumpyInplace(out_, out)
        outs.append(out)
    return outs


class DecisionTreeRegressor(AbstractTreeRegressor, BaseEstimator):
    x_reference = None
    x_dmatrix = None

    def __init__(self, max_depth=10, min_samples_leaf: int = 1, min_samples_split: int = 2,
                 colsample_bytree: float = 1.0, colsample_bylevel: float = 1.0):
        self._handle = _core.Tree(max_depth)
        self._builder_class = _core.LayerWiseTreeBuilder
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        pass

    def fit(self, x: np.ndarray, y: np.ndarray, sample_weight: typing.Union[None, np.ndarray] = None, eval_set=None,
            eval_metric=None):
        if y.ndim == 2 and y.shape[-1] == 1:
            y = y.squeeze()

        x_ = maybe_numpyToDMatrix(x)
        y_ = maybe_numpyToDColumn(y)
        builder = self._builder_class(x_, y_,
                                      min_samples_leaf=self.min_samples_leaf,
                                      min_samples_split=self.min_samples_split,
                                      colsample_bytree=self.colsample_bytree,
                                      colsample_bylevel=self.colsample_bylevel)
        builder.update(self._handle)
        self._eval(eval_set, eval_metric)
        return self

    def get_node(self, idx: int):
        return self._handle.get_node(idx)

    @staticmethod
    def predict_many(trees: typing.List, x: np.ndarray) -> typing.List[np.ndarray]:
        x_ = maybe_numpyToDMatrix(x)
        outs = []
        for tree in trees:
            out = np.zeros((x.shape[0],))
            out_ = tree._handle.predict_value(x_)
            _core.DColumntoNumpyInplace(out_, out)
            outs.append(out)
        return outs

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_ = maybe_numpyToDMatrix(x)
        out = np.zeros((x.shape[0],))
        out_ = self._handle.predict_value(x_)
        _core.DColumntoNumpyInplace(out_, out)
        del x_
        return out

    def predict_leaf(self, x: np.ndarray) -> np.ndarray:
        x_ = maybe_numpyToDMatrix(x)
        out = np.array([self._handle.predict_leaf(x_, i) for i in range(x.shape[0])])
        del x_
        return out.astype(int)


class GHDecisionTreeRegressor(AbstractTreeRegressor):
    def __init__(self, max_depth: int = 10, min_samples_leaf: int = 1, min_samples_split: int = 2,
                 min_weight_leaf: float = 0.0, min_weight_split: float = 0.0,
                 colsample_bytree: float = 1.0, colsample_bylevel: float = 1.0,
                 reg_lambda: float = 1.0, reg_alpha: float = 0.0):
        self._builder_class = _core.GHLayerWiseTreeBuilder
        self._handle = _core.Tree(max_depth)
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.min_weight_leaf = min_weight_leaf
        self.min_weight_split = min_weight_split
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        pass

    def fit_gh(self, x: np.ndarray, g: np.ndarray, h: np.ndarray, sample_weight: typing.Union[None, np.ndarray] = None,
               eval_set=None,
               eval_metric=None):
        if g.ndim == 2 and g.shape[-1] == 1:
            g = g.squeeze()
        if h.ndim == 2 and h.shape[-1] == 1:
            h = h.squeeze()

        x_ = maybe_numpyToDMatrix(x)
        g_ = maybe_numpyToDColumn(g)
        h_ = maybe_numpyToDColumn(h)
        builder = self._builder_class(x_, g_, h_,
                                      min_samples_leaf=self.min_samples_leaf,
                                      min_samples_split=self.min_samples_split,
                                      min_weight_leaf=self.min_weight_leaf,
                                      min_weight_split=self.min_weight_split,
                                      colsample_bytree=self.colsample_bytree,
                                      colsample_bylevel=self.colsample_bylevel,
                                      reg_lambda=self.reg_lambda,
                                      reg_alpha=self.reg_alpha)
        builder.update(self._handle)
        del x_, g_, h_
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_ = maybe_numpyToDMatrix(x)
        out_ = self._handle.predict_value(x_)
        out = np.zeros((x.shape[0],))
        _core.DColumntoNumpyInplace(out_, out)
        del x_
        return out

    @staticmethod
    def predict_many(trees: typing.List, x: np.ndarray) -> typing.List[np.ndarray]:
        x_ = maybe_numpyToDMatrix(x)
        outs = []
        for tree in trees:
            out = np.zeros((x.shape[0],))
            out_ = tree._handle.predict_value(x_)
            _core.DColumntoNumpyInplace(out_, out)
            outs.append(out)
        del x_
        return outs

    pass
