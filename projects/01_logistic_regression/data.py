from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True) 
class Data:
    x: np.ndarray
    y: np.ndarray

def make_data(n, seed) -> tuple[Data, Data, Data]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    noise = rng.normal(scale=0.35, size=n)
    score = x[:, 0] + 0.8 * x[:, 1] + noise
    y = (score > 0).astype(np.int64)
    order = rng.permutation(n)
    x, y = x[order], y[order]
    a, b = int(n * 0.6), int(n * 0.8)
    return Data(x[:a], y[:a]), Data(x[a:b], y[a:b]), Data(x[b:], y[b:])