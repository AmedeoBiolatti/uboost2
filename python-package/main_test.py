import numpy as np
import sklearn.tree as sktree
from sklearn import metrics
import xgboost as xgb
import matplotlib.pyplot as plt
from time import time

import uboost2 as u

x = np.random.uniform(0, 1, (2000, 1000))
y = x.sum(1)

params = dict(max_depth=10)

tree = u.Tree(**params)
t0 = time()
tree.fit(x, y)
t1 = time()
print("Time %.4f\t\tMSE %.4f" % (t1 - t0, metrics.mean_squared_error(y, tree.predict(x))))

tree = sktree.DecisionTreeRegressor(**params)
t0 = time()
tree.fit(x, y)
t1 = time()
print("Time %.4f\t\tMSE %.4f" % (t1 - t0, metrics.mean_squared_error(y, tree.predict(x))))

tree = xgb.XGBRegressor(n_estimators=1, learning_rate=1.0, **params)
t0 = time()
tree.fit(x, y)
t1 = time()
print("Time %.4f\t\tMSE %.4f" % (t1 - t0, metrics.mean_squared_error(y, tree.predict(x))))
