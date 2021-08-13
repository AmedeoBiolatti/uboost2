from sklearn import metrics as sk_metrics
from .utils import sigmoid

_metrics = dict()


def get_metric(name: str):
    if name.lower() in _metrics.keys():
        return _metrics[name.lower()]
    else:
        raise ValueError("Metric %s not found" % name)


def register_metric(name: str, fn):
    _metrics[name.lower()] = fn
    pass


register_metric('mse', sk_metrics.mean_squared_error)
register_metric('rmse', lambda y, p: (sk_metrics.mean_squared_error(y, p) ** 0.5))
register_metric('mae', sk_metrics.mean_absolute_error)
register_metric('logloss', sk_metrics.log_loss)
register_metric('logloss_logit', lambda y, p: sk_metrics.log_loss(y, sigmoid(p)))
# ...
