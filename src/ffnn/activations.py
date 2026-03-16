import numpy as np

class Linear:
    def forward(self, x):
        self.input = x
        return x
    
    def backward(self, delta):
        return delta * np.ones_like(self.input)

class ReLu:
    def forward(self, x):
        self.input = x
        return np.maximum(0, x)
    
    def backward(self, delta):
        gradien = (self.input > 0).astype(float)
        return delta * gradien

class Sigmoid:
    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out
    
    def backward(self, delta):
        return delta * (self.out * (1-self.out))

class Tanh:
    def forward(self, x):
        self.out = np.tanh(x)
        return self.out
    
    def backward(self, delta):
        # Turunan dari tanh(x)
        return delta * (1 - self.out**2)

class Softmax:
    def forward(self, x):
        exp = np.exp(x - np.max(x, axis=-1, keepdims=True))
        self.out = exp / np.sum(exp, axis=-1, keepdims=True)
        return self.out
    
    def backward(self, delta):
        res = np.empty_like(delta)
        for i, (out_i, delta_i) in enumerate(zip(self.out, delta)):
            jacobian = np.diag(out_i) - np.outer(out_i, out_i)
            res[i] = jacobian @ delta_i
        return res

def get_activation(name: str):
    name = name.lower()
    
    activation = {
        "linear" : Linear(),
        "relu" : ReLu(),
        "sigmoid" : Sigmoid(),
        "tanh" : Tanh(),
        "softmax" : Softmax()
    }
    
    if name in activation:
        return activation[name]
    else:
        raise ValueError(f"Fungsi aktivasi '{name} tidak didukung. "
                        f"Pilihan: {list(activation.keys())}")