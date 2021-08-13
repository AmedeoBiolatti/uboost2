import numpy as np
import typing

from .losses import Loss


class Optimizer:

    def compute_step(self, loss: Loss, y: np.ndarray, p: np.ndarray):
        g, h = loss.grad_and_hess(y, p)
        return self.get_step(g, h)

    def compute_grad_and_hess(self, loss: Loss, y: np.ndarray, p: np.ndarray):
        g, h = loss.grad_and_hess(y, p)
        return self.get_grad_and_hess(g, h)

    def get_step(self, g: np.ndarray, h: typing.Union[type(None), np.ndarray] = None) -> np.ndarray:
        raise NotImplementedError

    def get_grad_and_hess(self, g: np.ndarray, h: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        return g.copy(), h.copy()


class GradientDescentOptimizer(Optimizer):
    def get_step(self, g: np.ndarray, h: typing.Union[type(None), np.ndarray] = None) -> np.ndarray:
        assert g.shape == h.shape
        return -g

    def get_grad_and_hess(self, g: np.ndarray, h: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        return g.copy(), np.ones_like(h)


class NewtonOptimizer(Optimizer):
    def get_step(self, g: np.ndarray, h: typing.Union[type(None), np.ndarray] = None) -> np.ndarray:
        assert g.shape == h.shape
        return -g / h


class MomentumOptimizer(Optimizer):

    def __init__(self, base_optimizer: Optimizer = GradientDescentOptimizer(), alpha: float = 0.5,
                 warmup: int = 0):
        self.base_optimizer: Optimizer = base_optimizer
        self.alpha: float = alpha
        self.last_step = None
        self.warmup: int = warmup
        self.n: int = 0
        pass

    def update_last_step(self, last_step: np.ndarray):
        self.last_step = last_step
        pass

    def get_step(self, g: np.ndarray, h: typing.Union[type(None), np.ndarray] = None) -> np.ndarray:
        g, h = self.get_grad_and_hess(g, h)
        return self.base_optimizer.get_step(g, h)

    def get_grad_and_hess(self, g: np.ndarray, h: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        if self.last_step is None:
            self.last_step = g
        g_ = self.alpha * self.last_step + (1 - self.alpha) * g

        if self.n >= self.warmup:
            g, h = self.base_optimizer.get_grad_and_hess(g_, h)
        else:
            g, h = self.base_optimizer.get_grad_and_hess(g, h)

        self.last_step = g_.copy()
        self.n += 1
        return g, h


class AdaMOptimizer(Optimizer):
    def __init__(self, base_optimizer: Optimizer = GradientDescentOptimizer(), alpha: float = 0.5, beta: float = 0.5):
        self.base_optimizer: Optimizer = base_optimizer
        self.alpha: float = alpha
        self.beta: float = beta
        #
        self.m = 0.0
        self.v = 0.0
        self.last_step = 0.0
        pass

    def update_last_step(self, last_step: np.ndarray):
        self.last_step = last_step
        pass

    def get_step(self, g: np.ndarray, h: typing.Union[type(None), np.ndarray] = None) -> np.ndarray:
        g, h = self.get_grad_and_hess(g, h)
        return self.base_optimizer.get_step(g, h)

    def get_grad_and_hess(self, g: np.ndarray, h: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        self.m = self.alpha * self.m + (1 - self.alpha) * self.last_step
        self.v = self.beta * self.v + (1 - self.beta) * self.last_step ** 2.0

        m_ = (self.alpha * self.m + (1 - self.alpha) * g) / (1 - self.alpha)
        v_ = (self.beta * self.v + (1 - self.beta) * g ** 2.0) / (1 - self.beta)

        g = m_ / (np.sqrt(v_) + 1e-4)

        g, h = self.base_optimizer.get_grad_and_hess(g, h)
        self.update_last_step(g)
        return g, h

    pass


class AdaBeliefOptimizer(Optimizer):
    def __init__(self, base_optimizer: Optimizer = GradientDescentOptimizer(), alpha: float = 0.5, beta: float = 0.5):
        self.base_optimizer: Optimizer = base_optimizer
        self.alpha: float = alpha
        self.beta: float = beta
        #
        self.m = 0.0
        self.v = 0.0
        self.last_step = 0.0
        pass

    def update_last_step(self, last_step: np.ndarray):
        self.last_step = last_step
        pass

    def get_step(self, g: np.ndarray, h: typing.Union[type(None), np.ndarray] = None) -> np.ndarray:
        g, h = self.get_grad_and_hess(g, h)
        return self.base_optimizer.get_step(g, h)

    def get_grad_and_hess(self, g: np.ndarray, h: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        eps = 1e-6

        self.m = self.alpha * self.m + (1 - self.alpha) * self.last_step
        self.v = self.beta * self.v + (1 - self.beta) * (self.last_step - self.m) ** 2.0 + eps

        m_ = (self.alpha * self.m + (1 - self.alpha) * g) / (1 - self.alpha)
        v_ = (self.beta * self.v + (1 - self.beta) * (self.m - g) ** 2.0 + eps) / (1 - self.beta)

        g = m_ / (np.sqrt(v_) + eps)

        g, h = self.base_optimizer.get_grad_and_hess(g, h)
        self.update_last_step(g)

        return g, h

    pass


#
_optimizers = dict()


def get_optimizer(name: str) -> type(Optimizer):
    if name.lower() in _optimizers.keys():
        return _optimizers[name.lower()]
    else:
        raise ValueError("Metric %s not found" % name)


def register_optimizer(name: str, fn: type(Optimizer)):
    _optimizers[name.lower()] = fn
    pass


register_optimizer('gd', GradientDescentOptimizer)
register_optimizer('newton', GradientDescentOptimizer)
register_optimizer('momentum', MomentumOptimizer)
