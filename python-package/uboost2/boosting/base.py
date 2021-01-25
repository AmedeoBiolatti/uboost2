import typing
import numpy as np
import datetime

from ..estimators.base import BaseEstimator
from ..transformers import Transformer
from ..utils import eval_results_to_str
from ..metrics import get_metric


def log_time(message=''):
    date = "[%s]" % datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    print(date, message)
    pass


class GeneralBoosting(BaseEstimator):
    aggregation_method = 'sum'

    def __init__(self, n_estimators: int = 10):
        self.n_estimators: int = n_estimators
        self.current_iteration: int = -1
        #
        self.estimators: typing.List[BaseEstimator] = list()
        self.transformers: typing.List[Transformer] = list()
        #
        self.callbacks = list()
        pass

    def fit(self, x: np.ndarray, y: np.ndarray, sample_weight: typing.Union[None, np.ndarray] = None, eval_set=None,
            eval_metric=None, verbose=0):
        self._initialize(x, y, sample_weight)

        self._on_training_begin()
        self._call_callbacks('on_training_begin')
        for i in range(self.n_estimators):
            self.current_iteration = i
            self._on_iteration_begin()
            self._call_callbacks('on_iteration_start')

            self._fit_learner()
            self._eval(eval_set, eval_metric)
            self._log(verbose)

            self._on_iteration_end()
            self._call_callbacks('on_iteration_end')
            pass
        self._on_training_end()
        self._call_callbacks('on_training_end')
        return self

    def predict_kth(self, x: np.ndarray, k) -> np.ndarray:
        if len(self.transformers) > 0:
            out = self.estimators[k].predict(self.transformers[k].transform(x))
        else:
            out = self.estimators[k].predict(x)
        return out

    def predict(self, x: np.ndarray) -> np.ndarray:
        out = [self.predict_kth(x, k) for k in range(len(self.estimators))]
        if type(self).aggregation_method == "sum":
            return np.stack(out, -1).sum(-1)
        elif type(self).aggregation_method == "mean":
            return np.stack(out, -1).mean(-1)
        elif type(self).aggregation_method == 'median':
            return np.median(np.stack(out, -1), -1)

    def _log(self, verbose):
        if not verbose:
            return
        if self.current_iteration % int(verbose) > 0:
            return
        date = "[%s]" % datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        iteration = "[Iter %3d]" % self.current_iteration
        out = '\033[96m' + date + ' ' + iteration + '\033[0m'
        if hasattr(self, 'evals_results_'):
            res = eval_results_to_str(self.evals_results_)
            out += ' ' + res
        print(out)
        pass

    def _call_callbacks(self, fn: str):
        for cb in self.callbacks:
            getattr(cb, fn)(self)
        pass

    def _initialize(self, x, y, sample_weight):
        raise NotImplementedError

    def _on_training_begin(self):
        pass

    def _on_training_end(self):
        pass

    def _on_iteration_begin(self):
        pass

    def _on_iteration_end(self):
        pass

    def _fit_learner(self):
        raise NotImplementedError

    def _eval(self, eval_set=None, eval_metric=None):
        if eval_set is None or eval_metric is None:
            return
        eval_set = eval_set if isinstance(eval_set, list) else [eval_set]
        eval_metric = eval_metric if isinstance(eval_metric, list) else [eval_metric]

        if not hasattr(self, 'evals_results_'):
            self.evals_results_ = dict()

        if not hasattr(self, 'p_eval'):
            self.p_eval = dict()

        for i_eval, (x_eval, y_eval) in enumerate(eval_set):
            set_name = "valid_%d" % i_eval

            if not set_name in self.p_eval.keys():
                self.p_eval[set_name] = self.predict(x_eval)
            else:
                self.p_eval[set_name] += self.predict_kth(x_eval, len(self.estimators) - 1)

            p_eval = self.p_eval[set_name]
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
