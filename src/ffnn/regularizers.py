import numpy as np


class L1Regularizer:
    def __init__(self, lambda_: float):
        self.lambda_ = lambda_

    def loss(self, weights: np.ndarray) -> float:
        return self.lambda_ * np.sum(np.abs(weights))

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        return self.lambda_ * np.sign(weights)


class L2Regularizer:
    def __init__(self, lambda_: float):
        self.lambda_ = lambda_

    def loss(self, weights: np.ndarray) -> float:
        return self.lambda_ * np.sum(weights**2)

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        return 2 * self.lambda_ * weights


class NoRegularizer:
    def loss(self, weights: np.ndarray) -> float:
        return 0.0

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        return np.zeros_like(weights)


def get_regularizer(name: str, **kwargs):
    name = name.lower()
    if name == "l1":
        return L1Regularizer(kwargs.get("lambda_", 0.01))
    elif name == "l2":
        return L2Regularizer(kwargs.get("lambda_", 0.01))
    elif name in ["none", "no"]:
        return NoRegularizer()
    else:
        raise ValueError(f"Unknown regularizer: {name}")
