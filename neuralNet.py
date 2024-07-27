import torch.nn as nn
import torch
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import logging

# Set up logger
logger = logging.getLogger(__name__)


class BaseNeuralNet(nn.Module, ABC):

    def __init__(self, layers):
        """
        Initializes a BaseNeuralNet object.

        Args:
            layers (list): A list of integers representing the number of neurons in each layer.

        Attributes:
            device (str): The device on which the model will be trained (either "cuda:0" if CUDA is available, or "cpu").
            layers (list): A list of integers representing the number of neurons in each layer.
            model: The neural network model built using the specified layers.

        """
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
                - gamma (dict): A dictionary containing the weights for different loss components.
                    - matrix (float): Weight for the matrix loss component.
                    - data (float): Weight for the data loss component.
                    - gradient (float): Weight for the gradient loss component.
                    - residual (float): Weight for the residual loss component.
                - lossFunction (function): A function that computes the loss given the input data.
                - xInt (torch.Tensor): Tensor containing the interior points data for residual loss.
                - xData (torch.Tensor): Tensor containing the data points for supervised learning.

            lrs (list): A list of learning rates to be used during training.
            iterations (list): A list of the number of iterations to be performed for each learning rate.
            verbose (bool, optional): Whether to print training logs. Defaults to False.

        Returns:
            pandas.DataFrame: A DataFrame containing information about each epoch of training.

        """

        # Extract data from feedDict
        gamma = feedDict["gamma"]
        lossFunction = feedDict["lossFunction"]
        xInt = feedDict["xInt"].to(self.device)
        xData = feedDict["xData"].to(self.device)

        # Placholder for gradients of interior points
        gradInt = torch.zeros(xInt.shape).to(self.device)

        # Placeholder for data and gradients of data points
        yData = torch.zeros((xData.shape[0], 1)).to(self.device)
        gradData = torch.zeros(xData.shape).to(self.device)

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
                loss.backward()
                self.optimizer.step()

                # Print training logs
                if epochTotal % 100 == 0:
                    logger.info(
                        f"{epochTotal} / {sum(iterations)} ({epoch} / {iteration}), "
                        f"lr:{lr:.1e}, loss:{loss:.2e} (data: {lossData:.2e}, "
                        f"grad: {lossGrad:.2e}, res: {lossResidual:.2e}"
                    )

                epochTotal += 1

                # Save training information
                info_dict = {
                    "xData": xData,
                    "epoch": epochTotal,
                    "loss": loss.detach().cpu().numpy().item(),
                }
                info.append(info_dict)

        return pd.DataFrame(info)

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
        pass


class MatrixNeuralNet(BaseNeuralNet):
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
