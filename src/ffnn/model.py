import math
import pickle

import matplotlib.pyplot as plt
import numpy as np

from ffnn.activations import get_activation
from ffnn.initializers import get_initializer
from ffnn.layers import DenseLayer
from ffnn.losses import get_loss
from ffnn.regularizers import get_regularizer


class FFNN:
    def __init__(
        self,
        layer_sizes: list[int],
        activations: list[str],
        loss: str,
        initializer: str,
        regularizer: str,
        init_kwargs=None,
        reg_kwargs=None,
    ):
        if init_kwargs is None:
            init_kwargs = {}
        if reg_kwargs is None:
            reg_kwargs = {}

        self.layers = []
        self.loss_fn = get_loss(loss)
        self.history = {"train_loss": [], "val_loss": []}

        reg_obj = get_regularizer(regularizer, **reg_kwargs)

        for i in range(len(layer_sizes) - 1):
            init_obj = get_initializer(initializer, **init_kwargs)
            act_obj = get_activation(activations[i])

            layer = DenseLayer(
                n_in=layer_sizes[i],
                n_out=layer_sizes[i + 1],
                activation=act_obj,
                initializer=init_obj,
                regularizer=reg_obj,
            )
            self.layers.append(layer)

    def forward(self, X: np.ndarray) -> np.ndarray:
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        out = self.forward(X)
        if (
            out.shape[1] > 1
            or self.loss_fn.__class__.__name__.lower() == "categoricalcrossentropy"
        ):
            return np.argmax(out, axis=1)
        else:
            return (out >= 0.5).astype(int)

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray):
        grad = self.loss_fn.backward(y_true, y_pred)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update_weights(self, learning_rate: float):
        for layer in self.layers:
            layer.W -= learning_rate * layer.dW
            layer.b -= learning_rate * layer.db

    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.forward(X)
        total_loss = self.loss_fn.forward(y, y_pred)

        for layer in self.layers:
            total_loss += layer.regularizer.loss(layer.W)

        return total_loss

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        batch_size: int,
        learning_rate: float,
        epochs: int,
        verbose: int = 1,
        validation_data=None,
    ):

        n_samples = X_train.shape[0]
        n_batches = math.ceil(n_samples / batch_size)

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            X_batches = np.array_split(X_shuffled, n_batches)
            y_batches = np.array_split(y_shuffled, n_batches)

            for X_batch, y_batch in zip(X_batches, y_batches):
                if X_batch.shape[0] == 0:
                    continue
                y_pred = self.forward(X_batch)
                self.backward(y_batch, y_pred)
                self.update_weights(learning_rate)

            train_loss = self.compute_loss(X_train, y_train)
            self.history["train_loss"].append(train_loss)

            val_loss = None
            if validation_data is not None:
                X_val, y_val = validation_data
                val_loss = self.compute_loss(X_val, y_val)
                self.history["val_loss"].append(val_loss)

            if verbose == 1:
                val_str = f" - val_loss: {val_loss:.4f}" if val_loss is not None else ""
                print(f"Epoch {epoch + 1}/{epochs} - loss: {train_loss:.4f}{val_str}")

        return self.history

    def save(self, filepath: str):
        data = {
            "weights": [layer.W for layer in self.layers],
            "biases": [layer.b for layer in self.layers],
            "history": self.history,
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    def load(self, filepath: str):
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        for i, layer in enumerate(self.layers):
            layer.W = data["weights"][i]
            layer.b = data["biases"][i]

        self.history = data["history"]

    def plot_weight_distribution(self, layer_indices: list[int]):
        for i in layer_indices:
            weights_flat = self.layers[i].W.flatten()

            plt.figure(figsize=(6, 4))
            plt.hist(weights_flat, bins=50, alpha=0.7, color="blue")
            plt.title(f"Layer {i} Weight Distribution")
            plt.xlabel("Weight value")
            plt.ylabel("Count")
            plt.show()

    def plot_gradient_distribution(self, layer_indices: list[int]):
        for i in layer_indices:
            grad_flat = self.layers[i].dW.flatten()

            plt.figure(figsize=(6, 4))
            plt.hist(grad_flat, bins=50, alpha=0.7, color="red")
            plt.title(f"Layer {i} Gradient Distribution")
            plt.xlabel("Gradient value")
            plt.ylabel("Count")
            plt.show()
