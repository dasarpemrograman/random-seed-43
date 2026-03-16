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
        return np.random.uniform(loc=self.mean, scale=self.std, size=shape)

def get_initializers(name: str):
    name = name.lower()
    
    initializer = {
        "zero" : ZeroInitializer(),
        "uniform" : UniformInitializer(),
        "normal" : NormalInitializer() 
    }
    if name in initializer:
        return initializer[name]
    else:
        raise ValueError(f"Fungsi aktivasi '{name} tidak didukung. "
                        f"Pilihan: {list(initializer.keys())}")