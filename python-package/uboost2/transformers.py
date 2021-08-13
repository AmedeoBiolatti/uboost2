import typing
import numpy as np
from sklearn import base


class Transformer(base.TransformerMixin):

    def fit(self, x: np.ndarray, y: typing.Union[type(None), np.ndarray] = None):
        return self

    def transform(self, x) -> np.ndarray:
        raise NotImplementedError

    pass


class DummyTransformer(Transformer):
    def transform(self, x) -> np.ndarray:
        return x.copy()


class ColumnSamplerTransformer(Transformer):

    def __init__(self, frac=1.0):
        self.frac: float = frac

    def fit(self, x: np.ndarray, y: typing.Union[type(None), np.ndarray] = None):
        self.n_columns = x.shape[-1]
        self.n_sample = max(int(self.n_columns * self.frac), 1)
        self.sample = [i for i in np.random.choice(np.arange(self.n_columns),
                                                   int(self.n_columns * self.frac),
                                                   replace=False)]
        return self

    def transform(self, x) -> np.ndarray:
        out = x.copy()
        out = out[:, self.sample].copy()
        return out

    pass
