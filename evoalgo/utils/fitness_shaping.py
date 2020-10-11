import numpy as np


def compute_ranks(x):
    """
    Returns int ranks in range [0, len(x) - 1].
    Note: scipy.stats.rankdata returns ranks in range [1, len(x)].
    taken from: https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py
    """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks


def compute_zero_centered_ranks(x):
    """
    Returns float ranks in range [-0.5, 0.5].
    taken from: https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py
    """
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float64)
    y /= (x.size - 1)  # Normalization - scaling to [0, 1].
    y -= .5  # shifting to [-0.5, 0.5].
    return y