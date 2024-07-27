import torch.nn as nn
import torch
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import logging

# Set up logger
logger = logging.getLogger(__name__)


class BaseNeuralNet(nn.Module, ABC):
    """
    Base class for a neural network model.

    Args:
        layers (list): A list of integers representing the number of neurons in each layer.

    Attributes:
        device (str): The device on which the model will be trained (either "cuda:0" if CUDA is available, or "cpu").
        layers (list): A list of integers representing the number of neurons in each layer.
        model: The neural network model built using the specified layers.
    """

    def __init__(self, layers):
        super(BaseNeuralNet, self).__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.layers = layers
        self.model = self._buildModel(layers)

    def _buildModel(self, layers):
        """
        Builds a neural network model based on the given layers.

        Args:
            layers (list): A list of layer sizes for the neural network.

        Returns:
            nn.Sequential: The built neural network model.
        """
        neuralNetLayers = self._buildLayers(layers)
        neuralNetModel = nn.Sequential(*neuralNetLayers)
        neuralNetModel = neuralNetModel.apply(self._weightInitialisation)
        return neuralNetModel

    def partialDerivative(self, tensorToDerive, x):
        """
        Compute the partial derivative of `tensorToDerive` with respect to `x`.

        Args:
            tensorToDerive (torch.Tensor): The tensor to compute the derivative of.
            x (torch.Tensor): The input tensor with respect to which the derivative is computed.

        Returns:
            torch.Tensor: The computed partial derivative.
        """
        grad = torch.autograd.grad(
            outputs=tensorToDerive,
            inputs=x,
            grad_outputs=torch.ones_like(tensorToDerive),
            create_graph=True,
        )[0]

        return grad

    def train(self, feedDict, lrs, iterations):
        """
        Trains the neural network model.

        Args:
            feedDict (dict): A dictionary containing the necessary input data for training.
            lrs (list): A list of learning rates to be used during training.
            iterations (list): A list of the number of iterations to be performed for each learning rate.

        Returns:
            pandas.DataFrame: A DataFrame containing information about each epoch of training.
        """
        # Implementation details omitted for brevity
        pass

    def computeValueFunctionDerivative(self, x):
        """
        Computes the derivative of the value function with respect to the input tensor x.

        Parameters:
        - x: The input tensor.

        Returns:
        - The derivative of the value function with respect to x.
        """
        valueFunction = self.computeValueFunction(x)
        return self.partialDerivative(tensorToDerive=valueFunction, x=x)

    def _directValueFunction(self, x):
        """
        The output of the network is the value function directly.

        Parameters:
        - x: Input to the network.

        Returns:
        - The value function output by the network.
        """
        return self.model(x)

    def _buildLayers(self, layers):
        """
        Builds the layers of the neural network.

        Args:
            layers (list): List of integers representing the number of units in each layer.

        Returns:
            list: List of neural network layers.
        """
        neuralNetLayers = []
        for i in range(len(layers) - 2):
            neuralNetLayers.append(
                nn.Linear(in_features=layers[i], out_features=layers[i + 1])
            )
            neuralNetLayers.append(nn.Sigmoid())
        neuralNetLayers.append(
            nn.Linear(in_features=layers[-2], out_features=layers[-1])
        )
        return neuralNetLayers

    def _weightInitialisation(self, layer):
        """
        Initializes the weights of the linear layers.

        Args:
            layer (torch.nn.Module): Linear layer to be initialized.
        """
        torch.manual_seed(1)
        if type(layer) == nn.Linear:
            torch.nn.init.xavier_normal_(layer.weight)

    @abstractmethod
    def computeValueFunction(self, x):
        """
        Abstract method to compute the value function.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The computed value function.
        """
        pass


class MatrixNeuralNet(BaseNeuralNet):
    """
    A neural network model for computing the value function using matrix operations.

    Args:
        layers (list): List of integers representing the number of units in each layer.

    Attributes:
        device (str): The device (CPU or GPU) on which the model is being trained.
        layers (list): List of integers representing the number of units in each layer.
        model (torch.nn.Module): The neural network model.

    Methods:
        computeValueFunction: Computes the value function using the neural network model.
    """

    def __init__(self, layers):
        super(MatrixNeuralNet, self).__init__(layers)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.layers = layers
        self.model = self._buildModel(layers).to(self.device)

    def computeValueFunction(self, x):
        """
        Computes the value function using the neural network model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Computed value function.
        """
        dim = x.shape[1]

        stackedMatrices = torch.zeros((x.shape[0], dim, dim)).to(self.device)
        outputModel = self.model(x)

        inds = np.triu_indices(dim)
        k = 0
        for i, j in zip(inds[0], inds[1]):
            stackedMatrices[:, i, j] = outputModel[:, k]
            stackedMatrices[:, j, i] = outputModel[:, k]
            k += 1

        valueFunction = 0.5 * torch.einsum(
            "ni, nij, nj -> n", x, stackedMatrices, x
        ).reshape(-1, 1).to(self.device)

        return valueFunction
