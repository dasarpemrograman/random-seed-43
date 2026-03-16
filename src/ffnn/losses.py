import numpy as np

class MSE:
    def forward(self, y_true, y_pred):
        assert y_true.shape == y_pred.shape
        
        return np.mean((y_true - y_pred)**2)
    
    def backward(self, y_true, y_pred):
        n = y_true.shape[0]
        
        return (-2/n) * (y_true - y_pred)

class BinaryCrossEntropy:
    def forward(self, y_true, y_pred):
        n = y_true.shape[0]
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        
        return -(1/n) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def backward(self, y_true, y_pred):
        n = y_true.shape[0]
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        
        return -(1/n) * ((y_true / y_pred) - (1- y_true) / (1 - y_pred))

class CategoricalCrossEntropy:
    def forward(self, y_true, y_pred):
        n = y_true.shape[0]
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        
        return -(1/n) * np.sum(y_true * np.log(y_pred))
    
    def backward(self, y_true, y_pred):
        n = y_true.shape[0]
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        
        return -(1/n) * (y_true / y_pred)

def get_loss(name: str):
    name = name.lower()
    
    losses = {
        "mse" : MSE(),
        "bce" : BinaryCrossEntropy(),
        "cce" : CategoricalCrossEntropy()
    }
    
    if name in losses:
        return losses[name]
    else:
        raise ValueError(f"Fungsi aktivasi '{name} tidak didukung. "
                        f"Pilihan: {list(losses.keys())}")