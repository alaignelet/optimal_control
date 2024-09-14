import torch.nn as nn
import torch
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from icnn import ConvexLinear
from enums import ActivationFunctionEnum, PositivityFunctionEnum, InitFunctionEnum
import logging

# Set up logger
logger = logging.getLogger("training")


class BaseNeuralNet(nn.Module, ABC):
    @abstractmethod
    def __init__(self, layers):
        """
        Abstract initializer for BaseNeuralNet.
        Args:
            layers (list): A list of integers representing the number of neurons in each layer.
        """
        super(BaseNeuralNet, self).__init__()
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

    @abstractmethod
    def _buildLayers(self, layers):
        pass

    @abstractmethod
    def _buildModel(self, layers):
        pass

    @abstractmethod
    def computeValueFunction(self, x):
        pass

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

        # Extract data from feedDict
        gamma = feedDict["gamma"]
        lossFunction = feedDict["lossFunction"]
        xInt = feedDict["xInt"].to(self.device)
        xData = feedDict["xData"].to(self.device)

        # Ensure xInt and xData are leaf tensors with requires_grad=True
        xInt = xInt.clone().detach().requires_grad_(True)
        xData = xData.clone().detach().requires_grad_(True)

        # Placeholder for gradients of interior points
        gradInt = torch.zeros_like(xInt, device=self.device)

        # Placeholder for data and gradients of data points
        yData = torch.zeros((xData.shape[0], 1), device=self.device)
        gradData = torch.zeros_like(xData, device=self.device)

        epochTotal = 0
        info = []

        # Iterate over learning rates and iterations
        for lr, iteration in zip(lrs, iterations):

            self.optimizer = torch.optim.Adam(
                params=self.model.parameters(), lr=lr, weight_decay=0.0
            )

            # Iterate over epochs
            for epoch in range(iteration):

                # Compute data and gradients of data points
                if gamma["data"] > 0.0:
                    yData = self.computeValueFunction(xData)

                if gamma["gradient"] > 0.0:
                    gradData = self.computeValueFunctionDerivative(xData)

                # Compute residuals for interior points
                if gamma["residual"] > 0.0:
                    gradInt = self.computeValueFunctionDerivative(xInt)

                # Compute loss and backpropagate
                lossData, lossGrad, lossResidual = lossFunction(
                    xInt, gradInt, yData, gradData
                )
                loss = (
                    gamma["data"] * lossData
                    + gamma["gradient"] * lossGrad
                    + gamma["residual"] * lossResidual
                )
                self.optimizer.zero_grad()

                # Retain graph if needed
                retain_graph = True if epoch < iteration - 1 else False
                loss.backward(retain_graph=retain_graph)
                self.optimizer.step()

                # Print training logs
                if epochTotal % 1000 == 0:
                    logger.info(
                        f"{epochTotal} / {sum(iterations)} ({epoch} / {iteration}), "
                        f"lr:{lr:.1e}, loss:{loss.item():.2e} (data: {lossData.item():.2e}, "
                        f"grad: {lossGrad.item():.2e}, res: {lossResidual.item():.2e})"
                    )

                epochTotal += 1

                # Save training information
                info_dict = {
                    "epoch": epochTotal,
                    "loss": loss.detach().cpu().numpy().item(),
                }
                info.append(info_dict)

        return pd.DataFrame(info)

    def _partialDerivative(self, tensorToDerive, x):
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
            grad_outputs=torch.ones_like(tensorToDerive, device=self.device),
            create_graph=True,
            retain_graph=True,
        )[0]

        return grad

    def computeValueFunctionDerivative(self, x):
        """
        Computes the derivative of the value function with respect to the input tensor x.

        Parameters:
        - x: The input tensor.

        Returns:
        - The derivative of the value function with respect to x.
        """
        valueFunction = self.computeValueFunction(x)
        return self._partialDerivative(tensorToDerive=valueFunction, x=x)

    def _directValueFunction(self, x):
        """
        The output of the network is the value function directly.

        Parameters:
        - x: Input to the network.

        Returns:
        - The value function output by the network.
        """
        return self.model(x)


class LinearNeuralNet(BaseNeuralNet):
    def __init__(self, layers):
        super(LinearNeuralNet, self).__init__(layers)
        self.model = self._buildModel(layers).to(self.device)

    def computeValueFunction(self, x):
        """
        Computes the value function using the neural network model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Computed value function.
        """
        return self._directValueFunction(x)

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
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)


class ConvexNeuralNet(BaseNeuralNet):
    def __init__(self, layers, activation, positivity, init):
        super(ConvexNeuralNet, self).__init__(layers)
        self.model = self._buildModel(layers, activation, positivity, init).to(
            self.device
        )

    def computeValueFunction(self, x):
        """
        Computes the value function using the neural network model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Computed value function.
        """
        return self._directValueFunction(x)

    def _buildModel(self, layers, activation, positivity, init):
        """
        Builds a neural network model based on the given layers.

        Args:
            layers (list): A list of layer sizes for the neural network.

        Returns:
            nn.Sequential: The built neural network model.
        """

        model = nn.Sequential(*self._buildLayers(layers, activation, positivity))

        for idx in range(len(model)):
            if isinstance(model[idx], ConvexLinear):
                init.value(model[idx].weight, model[idx].bias)

        return model

    def _buildLayers(self, layers, activation, positivity):
        """
        Builds the layers of the neural network.

        Args:
            layers (list): List of integers representing the number of units in each layer.

        Returns:
            list: List of neural network layers.
        """

        # input layer: Linear
        neuralNetLayers = [
            nn.Linear(in_features=layers[0], out_features=layers[1]),
            activation.value,
        ]

        # all convex layers
        for i in range(1, len(layers) - 2):
            neuralNetLayers.append(
                ConvexLinear(
                    in_features=layers[i],
                    out_features=layers[i + 1],
                    positivity=positivity.value,
                )
            )
            neuralNetLayers.append(activation.value)

        # output convex layer
        neuralNetLayers.append(
            ConvexLinear(
                in_features=layers[-2],
                out_features=layers[-1],
                positivity=positivity.value,
            )
        )
        return neuralNetLayers


class MatrixLinearNeuralNet(LinearNeuralNet):
    """
    A neural network model for computing the value function and evaluating matrices.

    Args:
        layers (list): List of integers representing the number of units in each layer.

    Attributes:
        device (str): The device to be used for computation (e.g., "cuda:0" or "cpu").
        layers (list): List of integers representing the number of units in each layer.
        model (torch.nn.Module): The neural network model.

    Methods:
        computeValueFunction: Computes the value function using the neural network model.
    """

    def __init__(self, layers):
        super(MatrixLinearNeuralNet, self).__init__(layers)

    def computeValueFunction(self, x):
        """
        Computes the value function using the neural network model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Computed value function.
        """
        dim = x.shape[1]
        batch_size = x.shape[0]

        # Get the output from the model
        outputModel = self.model(x)  # Shape: (batch_size, num_features)

        # Calculate the number of upper triangular elements including diagonal
        num_tri_elems = (dim * (dim + 1)) // 2

        # Ensure outputModel has the correct number of features
        assert (
            outputModel.shape[1] == num_tri_elems
        ), f"Expected outputModel to have {num_tri_elems} features, got {outputModel.shape[1]}"

        # Create indices for the upper triangular part
        inds = torch.triu_indices(dim, dim)

        # Initialize the stacked matrices
        stackedMatrices = torch.zeros((batch_size, dim, dim), device=self.device)

        # Assign the upper triangular elements
        stackedMatrices[:, inds[0], inds[1]] = outputModel

        # Mirror the upper triangle to the lower triangle
        stackedMatrices_transpose = stackedMatrices.transpose(1, 2)
        stackedMatrices = stackedMatrices + stackedMatrices_transpose
        # Subtract the diagonal elements once because they were added twice
        diagonal_indices = torch.arange(dim)
        stackedMatrices[:, diagonal_indices, diagonal_indices] /= 2

        # Compute the value function
        valueFunction = 0.5 * torch.einsum(
            "ni, nij, nj -> n", x, stackedMatrices, x
        ).reshape(-1, 1).to(self.device)

        return valueFunction


class MatrixConvexNeuralNet(ConvexNeuralNet):
    def __init__(self, layers, activation, positivity, init):
        super(MatrixConvexNeuralNet, self).__init__(
            layers, activation, positivity, init
        )

    def computeValueFunction(self, x):
        """
        Computes the value function using the neural network model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Computed value function.
        """
        dim = x.shape[1]
        batch_size = x.shape[0]

        # Get the output from the model
        outputModel = self.model(x)  # Shape: (batch_size, num_features)

        # Calculate the number of upper triangular elements including diagonal
        num_tri_elems = (dim * (dim + 1)) // 2

        # Ensure outputModel has the correct number of features
        assert (
            outputModel.shape[1] == num_tri_elems
        ), f"Expected outputModel to have {num_tri_elems} features, got {outputModel.shape[1]}"

        # Create indices for the upper triangular part
        inds = torch.triu_indices(dim, dim)

        # Initialize the stacked matrices
        stackedMatrices = torch.zeros((batch_size, dim, dim), device=self.device)

        # Assign the upper triangular elements
        stackedMatrices[:, inds[0], inds[1]] = outputModel

        # Mirror the upper triangle to the lower triangle
        stackedMatrices_transpose = stackedMatrices.transpose(1, 2)
        stackedMatrices = stackedMatrices + stackedMatrices_transpose
        # Subtract the diagonal elements once because they were added twice
        diagonal_indices = torch.arange(dim)
        stackedMatrices[:, diagonal_indices, diagonal_indices] /= 2

        # Compute the value function
        valueFunction = 0.5 * torch.einsum(
            "ni, nij, nj -> n", x, stackedMatrices, x
        ).reshape(-1, 1).to(self.device)

        return valueFunction
