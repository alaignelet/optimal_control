import numpy as np
import torch.nn as nn
import logging
import sys


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


def setLogger(name):
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Create console handler and set level to INFO
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Add formatter to handler
    handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(handler)
