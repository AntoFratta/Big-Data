from typing import List

import torch
import torch.nn as nn


def _activation(name: str) -> nn.Module:
    """Return the activation module associated with the configured name."""
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01)
    raise ValueError(f"Unsupported activation: '{name}'. Use 'relu' or 'leaky_relu'.")


class Autoencoder(nn.Module):
    """
    Fully connected autoencoder with configurable depth and activation.

    Parameters
    ----------
    layer_dims:
        Complete sequence of layer dimensions, including encoder, bottleneck,
        and decoder. The first and last dimensions must match.
    activation:
        Hidden-layer activation, either 'relu' or 'leaky_relu'.
    use_dropout:
        Whether to insert dropout after hidden activations.
    dropout_rate:
        Dropout probability used when use_dropout is enabled.
    """

    def __init__(
        self,
        layer_dims: List[int],
        activation: str = "relu",
        use_dropout: bool = False,
        dropout_rate: float = 0.15,
    ):
        super().__init__()

        if len(layer_dims) < 3:
            raise ValueError("layer_dims must contain at least [input, hidden, output].")
        if layer_dims[0] != layer_dims[-1]:
            raise ValueError("The first and last layer dimensions must match.")

        layers: List[nn.Module] = []

        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))

            is_last = i == len(layer_dims) - 2
            if not is_last:
                layers.append(_activation(activation))
                if use_dropout:
                    layers.append(nn.Dropout(p=dropout_rate))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
