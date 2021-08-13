import typing
import numpy as np


class Loss:
    name: str = 'Loss'

    def __call__(self, y: np.ndarray, p: np.ndarray) -> float:
        return self.value(y, p)

    def value(self, y: np.ndarray, p: np.ndarray) -> float:
        raise NotImplementedError

    def grad(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def hess(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def grad_and_hess(self, y: np.ndarray, p: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        return self.grad(y, p), self.hess(y, p)


class MSELoss(Loss):
    name: str = 'MSE'

    def value(self, y: np.ndarray, p: np.ndarray) -> float:
        return float(np.mean((y - p) ** 2.0))

    def grad(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        return p - y

    def hess(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        # it should be 2.0, but we normalize to have similar results with different optimizers
        # it's the same in xgboost
        return 1.0 + 0 * p


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1. / (1. + np.exp(-x))


class LogLoss(Loss):
    name: str = 'LogLoss'

    def value(self, y: np.ndarray, p: np.ndarray) -> float:
        pr = sigmoid(p)
        return np.mean(-y * np.log(pr) - (1 - y) * np.log(1 - pr))

    def grad(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        pr = sigmoid(p)
        return pr - y

    def hess(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        pr = sigmoid(p)
        return np.maximum(pr * (1 - pr), 1e-12)

    pass


_losses = dict()


def get_loss(name: str) -> type(Loss):
    if name.lower() in _losses.keys():
        return _losses[name.lower()]
    else:
        raise ValueError("Metric %s not found" % name)


def register_loss(name: str, fn: type(Loss)):
    _losses[name.lower()] = fn
    pass


register_loss('mse', MSELoss)
register_loss('logloss', LogLoss)
