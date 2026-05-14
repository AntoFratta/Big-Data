import torch
import torch.nn as nn
from typing import List


def _activation(name: str) -> nn.Module:
    """Restituisce il modulo di attivazione dato il nome (stringa)."""
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    elif name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01)
    else:
        raise ValueError(f"Attivazione non supportata: '{name}'. Usa 'relu' o 'leaky_relu'.")


class Autoencoder(nn.Module):
    """
    Autoencoder simmetrico a profondità variabile.

    Parameters
    ----------
    layer_dims : List[int]
        Lista completa delle dimensioni, encoder + bottleneck + decoder.
        Esempio: [41, 30, 10, 3, 10, 30, 41]
        Il primo elemento è input_dim, l'ultimo è output_dim (uguale a input_dim).
    activation : str
        Tipo di attivazione: 'relu' oppure 'leaky_relu'.
    use_dropout : bool
        Se True aggiunge un Dropout(0.5) dopo ogni attivazione tranne l'ultima.
    dropout_rate : float
        Probabilità di dropout (usata solo se use_dropout=True). Default 0.5.
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
            raise ValueError("layer_dims deve avere almeno 3 elementi [input, ..., output].")
        if layer_dims[0] != layer_dims[-1]:
            raise ValueError("Il primo e l'ultimo elemento di layer_dims devono essere uguali (input_dim == output_dim).")

        layers: List[nn.Module] = []

        # Costruisce tutti i blocchi Linear → Activation → (Dropout)
        # L'ultimo strato non ha attivazione né dropout (output lineare)
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))

            is_last = (i == len(layer_dims) - 2)
            if not is_last:
                layers.append(_activation(activation))
                if use_dropout:
                    layers.append(nn.Dropout(p=dropout_rate))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)