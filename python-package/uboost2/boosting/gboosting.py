import numpy as np
import typing

from sklearn import model_selection

from .base import GeneralBoosting, log_time
from ..losses import Loss, get_loss
from ..optimizers import Optimizer, get_optimizer
from ..estimators.tree import DecisionTreeRegressor
from ..transformers import DummyTransformer
from ..utils import logit, sigmoid


class GradientBoosting(GeneralBoosting):
    aggregation_method = 'sum'
    _training_pred_method = 2

    def __init__(self, n_estimators=10, learning_rate: float = 0.1, subsample: float = 1.0,
                 max_depth: int = 10,
                 max_delta_step: float = 1e6,
                 dropout_rate: float = 0.0,
                 base_score: typing.Union[str, float] = 'auto',
                 optimizer: typing.Union[str, Optimizer] = 'gd', loss: str = 'mse'):
        super(GradientBoosting, self).__init__(n_estimators=n_estimators)
        self.build_estimator = lambda: DecisionTreeRegressor(max_depth=max_depth)
        self.build_transformer = lambda: DummyTransformer()
        self.learning_rate = learning_rate
        self.max_delta_step: float = max_delta_step
        self.dropout_rate: float = dropout_rate
        self.subsample: float = subsample
        self.base_score: typing.Union[str, float] = base_score
        self.optimizer: Optimizer = get_optimizer(optimizer)() if isinstance(optimizer, str) else optimizer
        self.loss: Loss = get_loss(loss)()
        #
        if self.dropout_rate <= 0.0:
            self._training_pred_method = 2
        pass

    def _initialize(self, x, y, sample_weight):
        assert sample_weight is None
        self.x = x
        self.y = y.reshape(x.shape[0], -1)
        if self.base_score == 'auto':
            self.base_score = 'mean'
        if self.base_score == 'mean':
            self.baseline = self.y.mean(0).reshape(1, -1)
        elif self.base_score == 'logit':
            self.baseline = logit(self.y.mean(0).reshape(1, -1))
        elif self.base_score == 'zeros':
            self.baseline = 0 * self.y.mean(0).reshape(1, -1)
        elif self.base_score == 'zero':
            self.baseline = 0 * self.y.mean(0).reshape(1, -1)
        else:
            self.baseline = np.zeros((1, 1)) + self.base_score
        self.predictions = list()
        self.total_prediction = self.baseline + np.zeros((self.x.shape[0], 1))
        pass

    def _fit_learner(self):
        transformer = self.build_transformer()
        z = transformer.fit_transform(self.x)
        y = self.y

        if self._training_pred_method == 0:
            assert self.dropout_rate >= 1.0
            # method 0: simply predict at every iteration
            p = self.predict(self.x)
        elif self._training_pred_method == 1:
            # method 1: aggregate cached predictions
            p = self.baseline + np.zeros((self.x.shape[0], 1))
            if self.current_iteration >= 0:
                if self.dropout_rate <= 0.0:
                    p += np.stack(self.predictions, -1).sum(-1).reshape(p.shape)
                else:
                    n = len(self.predictions)
                    n_sample = int(np.ceil(n * (1 - self.dropout_rate)))
                    preds_sample_idxs = np.random.choice(range(n), n_sample, replace=False)
                    preds_sample = [self.predictions[i] for i in preds_sample_idxs]
                    preds_sum = np.stack(preds_sample, -1).sum(-1).reshape(p.shape)
                    ratio = n / (n - n_sample)
                    p += preds_sum * ratio
        elif self._training_pred_method == 2:
            # method 2: use cached aggregation of sums
            if self.dropout_rate <= 0 or len(self.predictions) == 0:
                p = self.total_prediction
            else:
                pred_l = self.total_prediction - self.baseline
                n = len(self.predictions)
                n_sample = int(np.floor(n * self.dropout_rate))
                if n_sample > 0:
                    preds_sample_idxs = np.random.choice(range(n), n_sample, replace=False)
                    preds_sample_to_remove = [self.predictions[i] for i in preds_sample_idxs]
                    preds_sum_to_remove = np.stack(preds_sample_to_remove, -1).sum(-1).reshape(pred_l.shape)
                else:
                    preds_sum_to_remove = 0.0
                ratio = n / (n - n_sample)
                p = (pred_l - preds_sum_to_remove) * ratio + self.baseline
                pass

        # train
        estimator = self.build_estimator()
        lr = float(self.learning_rate)
        if hasattr(estimator, 'fit_gh'):
            g, h = self.optimizer.compute_grad_and_hess(self.loss, y, p)
            g *= -lr

            if self.subsample is not None and self.subsample < 1.0:
                idxT = np.arange(z.shape[0])
                idxT, idxV = model_selection.train_test_split(idxT, test_size=1 - self.subsample)
                estimator.fit_gh(z[idxT], g[idxT], h[idxT])
            else:
                estimator.fit_gh(z, g, h)
                pass
        else:
            g = self.optimizer.compute_step(self.loss, y, p)
            g *= lr

            if self.subsample is not None and self.subsample < 1.0:
                idxT = np.arange(z.shape[0])
                idxT, idxV = model_selection.train_test_split(idxT, test_size=1 - self.subsample)
                estimator.fit(z[idxT], g[idxT])
            else:
                estimator.fit(z, g)
                pass
            pass

        # pred
        pred = estimator.predict(z).reshape(y.shape)
        pred = np.clip(pred, -self.max_delta_step, +self.max_delta_step)
        self.predictions.append(pred)
        self.total_prediction += pred
        if hasattr(self.optimizer, 'update_last_step'):
            self.optimizer.update_last_step(pred / lr)

        self.estimators.append(estimator)
        self.transformers.append(transformer)
        pass

    def predict(self, x: np.ndarray) -> np.ndarray:
        # TODO: handle transformers
        out = np.zeros((x.shape[0], 1))
        out += self.baseline
        if self.current_iteration >= 0:
            if len(list(set([type(e) for e in self.estimators]))) == 1 and hasattr(type(self.estimators[0]),
                                                                                   'predict_many'):
                preds = type(self.estimators[0]).predict_many(self.estimators, x)
            else:
                preds = [self.predict_kth(x, k) for k in range(len(self.estimators))]
            p = np.stack([np.clip(pi, -self.max_delta_step, +self.max_delta_step) for pi in preds], -1).sum(-1)
            out += p.reshape(out.shape)
        return out

    def predict_kth(self, x: np.ndarray, k) -> np.ndarray:
        return self.estimators[k].predict(self.transformers[k].transform(x)).reshape(x.shape[0], -1)

    pass
