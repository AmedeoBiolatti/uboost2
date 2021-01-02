import shutil
import os

dir = "../out/build/x64-Release/include"
libpath = os.path.join(dir, '_core.pyd')

try:
    shutil.copy(libpath, '_core.pyd')
except PermissionError:
    pass

import _core
import numpy as np


class Tree:

    def __init__(self, max_depth=10):
        self._handle = _core.Tree(max_depth=max_depth)
        self._builder = None
        self._builder_class = _core.LayerWiseTreeBuilder
        pass

    def fit(self, x, y):
        x_ = _core.numpyToDMatrix(x)
        y_ = _core.numpyToDColumn(y)
        self._builder = self._builder_class(x_, y_)
        self._builder.update(self._handle)
        return self

    def predict(self, x):
        x_ = _core.numpyToDMatrix(x)
        out = np.array([self._handle.predict_value(x_, i) for i in range(x.shape[0])])
        return out

    def predict_leaf(self, x):
        x_ = _core.numpyToDMatrix(x)
        out = np.array([self._handle.predict_leaf(x_, i) for i in range(x.shape[0])])
        return out