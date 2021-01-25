import numpy as np
import pandas as pd
from sklearn import model_selection

from .base import GeneralBoosting
from ..transformers import ColumnSamplerTransformer, DummyTransformer
from ..estimators.tree import DecisionTreeRegressor


class RandomForestRegressor(GeneralBoosting):
    aggregation_method = 'mean'

    def __init__(self, n_estimators: int = 10, max_depth=10, colsample_bytree=0.5, subsample=1.0):
        super(RandomForestRegressor, self).__init__(n_estimators=n_estimators)
        self.transformers = list()
        self.estiamtor_builder = lambda: DecisionTreeRegressor(max_depth=max_depth)
        self.colsample_bytree = colsample_bytree
        self.subsample = subsample
        pass

    def _initialize(self, x, y, sample_weight):
        self.x = x
        self.y = y
        assert sample_weight is None
        self.oof_mse_by_leaf = []
        self.oof_mse = []
        self.list_idxV = []
        self.y_std = self.y.std()
        pass

    def _fit_learner(self):
        transformer = ColumnSamplerTransformer(frac=self.colsample_bytree)
        z: np.ndarray = transformer.fit_transform(self.x)

        idxT = np.arange(z.shape[0])
        idxV = None
        if self.subsample is not None:
            if self.subsample < 1.0:
                idxT, idxV = model_selection.train_test_split(idxT, test_size=1 - self.subsample)

        estimator = self.estiamtor_builder()
        estimator.fit(z[idxT], self.y[idxT])

        if idxV is not None:
            pred = estimator.predict(z[idxV])
            leaf = estimator.predict_leaf(z[idxV])
            res = pd.DataFrame({'leaf': leaf, 'pred': pred, 'target': self.y[idxV].squeeze()})
            self.list_idxV.append(idxV)
            self.oof_mse.append(np.mean((res['pred'] - res['target']) ** 2))
            self.oof_mse_by_leaf.append(res.groupby('leaf').apply(lambda x: np.mean((x['pred'] - x['target']) ** 2)))
            pass
        self.transformers.append(transformer)
        self.estimators.append(estimator)
        pass

    def predict_kth(self, x: np.ndarray, k) -> np.ndarray:
        return self.estimators[k].predict(self.transformers[k].transform(x))

    def predict_kth_oof_mse(self, x, k):
        e: DecisionTreeRegressor = self.estimators[k]
        z: np.ndarray = self.transformers[k].transform(x)
        leaves = e.predict_leaf(z)
        mse = np.array(
            [self.oof_mse_by_leaf[k].loc[l] if l in self.oof_mse_by_leaf[k].index else np.nan for l in leaves])
        mse[np.isnan(mse)] = self.y_std
        return mse

    def predict_v2(self, x, perc=0.9):
        preds = np.stack([self.predict_kth(x, k) for k, e in enumerate(self.estimators)], 1)
        mses = np.stack([self.predict_kth_oof_mse(x, k) for k, e in enumerate(self.estimators)], 1)

        idx = np.argsort(mses, 1)[:, :int(self.n_estimators * perc)]
        p = []
        for i, pi in enumerate(preds):
            p += [pi[idx[i]]]
        out = np.stack(p, 0).mean(1)
        return out
