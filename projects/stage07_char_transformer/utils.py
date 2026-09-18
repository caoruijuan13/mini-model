import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    """Return a numerically stable softmax over the last axis."""
    # subtract each row maximum before exponentiation.
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp_values = np.exp(shifted)
    result = exp_values / exp_values.sum(axis=-1, keepdims=True)
    return result