import numpy as np


class Linear:
    def forward(self, x):
        self.input = x
        return x

    def backward(self):
        return np.ones_like(self.input)


class ReLu:
    def forward(self, x):
        self.input = x
        return np.maximum(0, x)

    def backward(self):
        return (self.input > 0).astype(float)


class Sigmoid:
    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self):
        return self.out * (1 - self.out)


class Tanh:
    def forward(self, x):
        self.out = np.tanh(x)
        return self.out

    def backward(self):
        return 1 - self.out**2


class Softmax:
    def forward(self, x):
        exp = np.exp(x - np.max(x, axis=-1, keepdims=True))
        self.out = exp / np.sum(exp, axis=-1, keepdims=True)
        return self.out

    def backward(self):
        batch_size, n = self.out.shape
        J = np.zeros((batch_size, n, n))
        for i, out_i in enumerate(self.out):
            J[i] = np.diag(out_i) - np.outer(out_i, out_i)
        return J


def get_activation(name: str):
    name = name.lower()

    activation = {
        "linear": Linear(),
        "relu": ReLu(),
        "sigmoid": Sigmoid(),
        "tanh": Tanh(),
        "softmax": Softmax(),
    }

    if name in activation:
        return activation[name]
    else:
        raise ValueError(
            f"Fungsi aktivasi '{name}' tidak didukung. "
            f"Pilihan: {list(activation.keys())}"
        )
