import numpy as np
import torch.nn as nn


def stackWeights(network):
    weights = np.array([])
    for layer in network.layers:
        if type(layer) == nn.Linear:
            weights = np.concatenate(
                [
                    weights,
                    layer.weight.detach().numpy().flatten(),
                    layer.bias.detach().numpy().flatten(),
                ]
            )
    return weights
