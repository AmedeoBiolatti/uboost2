import numpy as np


def eval_results_to_str(eval_results: dict):
    out = ""
    for set_name in eval_results.keys():
        for metric_name in eval_results[set_name].keys():
            out += "%s's %s %.6f " % (set_name, metric_name, eval_results[set_name][metric_name][-1])
            pass
        pass
    return out.strip()


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1. / (1. + np.exp(-x))


def logit(x: np.ndarray) -> np.ndarray:
    eps = 1e-6
    x = np.clip(x, eps, 1 - eps)
    return np.log(x / (1 - x))
