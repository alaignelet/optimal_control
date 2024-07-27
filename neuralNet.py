import torch.nn as nn
import torch
import numpy as np
import spdlayers
import pandas as pd
from abc import ABC, abstractmethod


class BaseNeuralNet(nn.Module, ABC):
    def __init__(self, layers):
        super(BaseNeuralNet, self).__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.layers = layers
        self.model = self._buildModel(layers)

    def _buildModel(self, layers):
        neuralNetLayers = self._buildLayers(layers)
        neuralNetModel = nn.Sequential(*neuralNetLayers)
        neuralNetModel = neuralNetModel.apply(self._weightInitialisation)
        return neuralNetModel

    def partialDerivative(self, tensorToDerive, x):
        grad = torch.autograd.grad(
            outputs=tensorToDerive,
            inputs=x,
            grad_outputs=torch.ones_like(tensorToDerive),
            create_graph=True,
        )[0]

        return grad

    def train(self, feedDict, lrs, iterations, verbose=False):
        """Training function."""
        gamma = feedDict["gamma"]
        lossFunction = feedDict["lossFunction"]
        evaluationFunction = feedDict["evaluationFunction"]

        xInt = feedDict["xInt"].to(self.device)
        xData = feedDict["xData"].to(self.device)

        # network dependend quantities
        matrixData = torch.zeros((xInt.shape[0], self.layers[-1])).to(self.device)
        gradInt = torch.zeros(xInt.shape).to(self.device)
        yData = torch.zeros((xData.shape[0], 1)).to(self.device)
        gradData = torch.zeros(xData.shape).to(self.device)
        errorDerivative = torch.zeros(xData.shape).to(self.device)

        epochTotal = 0
        info = []

        for lr, iteration in zip(lrs, iterations):

            self.optimizer = torch.optim.Adam(
                params=self.model.parameters(), lr=lr, weight_decay=0.0
            )

            for epoch in range(iteration):

                # xInt = dataSampler.samplePoints(interiorPointCount).to(self.device)

                # compute model dependent quantities
                if gamma["matrix"] > 0:
                    matrixData = self.model(xData)

                if gamma["data"] > 0.0:
                    yData = self.computeValueFunction(xData)

                if gamma["gradient"] > 0.0:
                    gradData = self.computeValueFunctionDerivative(xData)

                if gamma["residual"] > 0.0:
                    gradInt = self.computeValueFunctionDerivative(xInt)

                # compute loss and backpropagate
                lossData, lossGrad, lossResidual, lossMatrix = lossFunction(
                    xInt, gradInt, yData, gradData, matrixData, errorDerivative
                )
                loss = (
                    gamma["data"] * lossData
                    + gamma["gradient"] * lossGrad
                    + gamma["residual"] * lossResidual
                    + gamma["matrix"] * lossMatrix
                )
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # print logs
                if epochTotal % 100 == 0:
                    if verbose:
                        print(
                            "%d / %d (%d / %d), lr:%.1e, loss:%.2e (data: %.2e, grad: %.2e, res: %.2e, mat: %.2e)"
                            % (
                                epochTotal,
                                sum(iterations),
                                epoch,
                                iteration,
                                lr,
                                loss,
                                lossData,
                                lossGrad,
                                lossResidual,
                                lossMatrix,
                            )
                        )

                epochTotal += 1
                info_dict = {
                    "xData": xData,
                    "epoch": epochTotal,
                    "mse": mse.detach().cpu().numpy().item(),
                    "loss": loss.detach().cpu().numpy().item(),
                }
                info.append(info_dict)

        return pd.DataFrame(info)

    def computeValueFunctionDerivative(self, x):
        valueFunction = self.computeValueFunction(x)
        return self.partialDerivative(tensorToDerive=valueFunction, x=x)

    def directValueFunction(self, x):
        """The output of the network is the value function directly."""
        return self.model(x)

    @abstractmethod
    def computeValueFunction(self, x):
        pass

    @abstractmethod
    def _buildLayers(self, layers):
        pass

    @abstractmethod
    def _weightInitialisation(self, layer):
        pass


class MatrixNeuralNet(BaseNeuralNet):
    """Neural Network used as a mapping function.
    Glorot initialisation.
    """

    def __init__(self, layers):
        super(BaseNeuralNet, self).__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.layers = layers
        self.model = self._buildModel(layers).to(self.device)

    def _buildLayers(self, layers):
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

    def _buildModel(self, layers, imposePsd, spd):
        neuralNetLayers = self._buildLayers(layers, imposePsd, spd)
        neuralNetModel = nn.Sequential(*neuralNetLayers)
        neuralNetModel = neuralNetModel.apply(self._normalInit)
        return neuralNetModel

    def _normalInit(self, layer):
        torch.manual_seed(1)
        if type(layer) == nn.Linear:
            torch.nn.init.xavier_normal_(layer.weight)

    def computeValueFunction(self, x):
        dim = x.shape[1]

        # the below is SUPER fast
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

    def _matrixEvaluated(self, x):
        """The output of the network is the symetric matrix P."""
