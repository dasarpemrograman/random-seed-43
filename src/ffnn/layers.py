import numpy as np


class DenseLayer:
    def __init__(self, n_in: int, n_out: int, activation, initializer, regularizer):
        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation
        self.regularizer = regularizer

        self.W = initializer.initialize((n_in, n_out))
        self.b = np.zeros((1, n_out))

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        self.input_cache = None
        self.z_cache = None
        self.a_cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input_cache = x
        z = x @ self.W + self.b
        self.z_cache = z
        a = self.activation.forward(z)
        self.a_cache = a
        return a

    def backward(self, upstream_grad: np.ndarray) -> np.ndarray:
        batch_size = self.input_cache.shape[0]

        if self.activation.__class__.__name__.lower() == "softmax":
            J = self.activation.backward(self.z_cache)
            dz = np.einsum("bi,bij->bj", upstream_grad, J)
        else:
            dz = upstream_grad * self.activation.backward(self.z_cache)

        self.dW = (self.input_cache.T @ dz) / batch_size
        self.dW += self.regularizer.gradient(self.W)

        self.db = np.sum(dz, axis=0, keepdims=True) / batch_size

        dx = dz @ self.W.T
        return dx
