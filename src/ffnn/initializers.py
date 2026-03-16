import numpy as np


class ZeroInitializer:
    def initialize(self, shape):
        return np.zeros(shape)


class UniformInitializer:
    def __init__(self, lower, upper, seed=None):
        self.lower = lower
        self.upper = upper
        self.seed = seed

    def initialize(self, shape):
        if self.seed is not None:
            np.random.seed(self.seed)
        return np.random.uniform(low=self.lower, high=self.upper, size=shape)


class NormalInitializer:
    def __init__(self, mean, variance, seed=None):
        self.mean = mean
        self.std = np.sqrt(variance)
        self.seed = seed

    def initialize(self, shape):
        if self.seed is not None:
            np.random.seed(self.seed)
        return np.random.normal(loc=self.mean, scale=self.std, size=shape)


def get_initializer(name: str, **kwargs):
    name = name.lower()

    initializer = {
        "zero": ZeroInitializer(),
        "uniform": UniformInitializer(
            lower=kwargs.get("lower", -0.1),
            upper=kwargs.get("upper", 0.1),
            seed=kwargs.get("seed", None),
        ),
        "normal": NormalInitializer(
            mean=kwargs.get("mean", 0.0),
            variance=kwargs.get("variance", 1.0),
            seed=kwargs.get("seed", None),
        ),
    }
    if name in initializer:
        return initializer[name]
    else:
        raise ValueError(
            f"Fungsi aktivasi '{name} tidak didukung. "
            f"Pilihan: {list(initializer.keys())}"
        )
