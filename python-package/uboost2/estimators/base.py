import typing
import numpy as np
from sklearn import base
from ..metrics import get_metric


class BaseEstimator(base.BaseEstimator):

    def fit(self, x: np.ndarray, y: np.ndarray, sample_weight: typing.Union[None, np.ndarray] = None, eval_set=None,
            eval_metric=None):
        raise NotImplementedError

    def predict(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _eval(self, eval_set=None, eval_metric=None):
        if eval_set is None or eval_metric is None:
            return
        eval_set = eval_set if isinstance(eval_set, list) else [eval_set]
        eval_metric = eval_metric if isinstance(eval_metric, list) else [eval_metric]

        if not hasattr(self, 'evals_results_'):
            self.evals_results_ = dict()

        for i_eval, (x_eval, y_eval) in enumerate(eval_set):
            set_name = "valid_%d" % i_eval
            p_eval = self.predict(x_eval).reshape(x_eval.shape[0], -1)
            if set_name not in self.evals_results_.keys():
                self.evals_results_[set_name] = dict()
            for i_metric, metric_name in enumerate(eval_metric):
                metric_fn = get_metric(metric_name)
                metric_value = metric_fn(y_eval, p_eval)
                if metric_name not in self.evals_results_[set_name].keys():
                    self.evals_results_[set_name][metric_name] = [metric_value]
                else:
                    self.evals_results_[set_name][metric_name] += [metric_value]
                pass
            pass
        pass

    pass
