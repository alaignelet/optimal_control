from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
from generateData import GenerateData


class HamiltonJacobiBellman(ABC):
    """Build the PDE solver for Hamilton Jacobi.
    The problem is:
    y' = f(y) + g(y)u
    J = int l(y) + beta ||u||^2

    The boundary condition is V(T,x).
    """

    def __init__(self, network, domain, beta, gamma, correctShift):
        self.network = network
        self.domain = domain
        self.beta = beta
        self.gamma = gamma
        self.dataSampler = GenerateData(domain=self.domain)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.correctShift = correctShift

    def train(self, interiorPointCount, dataPointCount, lrs, iterations, sampling):
        """
        Generate data and train the network.

        Args:
            interiorPointCount (int): The number of interior points to generate.
            dataPointCount (int): The number of data points to generate.
            lrs (list): A list of learning rates to use during training.
            iterations (int): The number of training iterations.
            sampling (str): The sampling method to use for generating interior points.
                Can be either "random" or "grid". Defaults to "random".

        Returns:
            list: A list of loss values during training.

        """
        # Sample interior points
        if sampling == "random":
            xInt = self.dataSampler.samplePoints(interiorPointCount).to(self.device)
        elif sampling == "grid":
            xInt = self.dataSampler.sampleGrid(interiorPointCount).to(self.device)

        # Sample evaluation points for out of sample performance
        xEvaluation = self.getEvaluationPoints().to(self.device)
        self.yEvaluationTrue = self.groundTruthSolution(xEvaluation.detach())

        # Sample data points
        xData = self._getDataPoints(dataPointCount).to(self.device)

        # Compute the true values for the data points and their derivatives
        self.yTrue = self.dataValueFunction(xData.detach()).to(self.device)
        self.gradTrue = self.dataValueFunctionDerivative(xData.detach()).to(self.device)

        # Build feed dictionary to train the network
        feedDict = {
            "xEvaluation": xEvaluation,
            "xInt": xInt,
            "xData": xData,
            "gamma": self.gamma,
            "lossFunction": self.lossFunction,
            "evaluationFunction": self.evaluationFunction,
            "groundTruthSolution": self.groundTruthSolution,
            "dataSampler": self.dataSampler,
        }

        # Train the network and return the loss values
        return self.network.train(feedDict, lrs, iterations)

    def lossFunction(self, xInt, gradInt, yData, gradData):
        """
        Compute the loss function.

        Args:
            xInt (torch.Tensor): Input tensor for xInt.
            gradInt (torch.Tensor): Input tensor for gradInt.
            yData (torch.Tensor): Input tensor for yData.
            gradData (torch.Tensor): Input tensor for gradData.

        Returns:
            tuple: A tuple containing the following loss values:
                - lossData (torch.Tensor): Loss value for data.
                - lossGradient (torch.Tensor): Loss value for gradient.
                - residualInt (torch.Tensor): Residual value for the Hamilton-Jacobi equation.
        """
        residualInt = torch.tensor([0]).float().to(self.device)
        lossData = torch.tensor([0]).float().to(self.device)
        lossGradient = torch.tensor([0]).float().to(self.device)

        # Compute the residual on the interior points
        if self.gamma["residual"] > 0.0:
            equation = self._computeHamiltonJacobiEquation(xInt, gradInt)
            residualInt = torch.mean(equation**2).to(self.device)

        # Compute the loss on the data points
        if self.gamma["data"] > 0.0:
            lossData = (
                torch.mean((yData.double() - self.yTrue.double()) ** 2)
                .float()
                .to(self.device)
            )

        # Compute the loss on the gradient of the data points
        if self.gamma["gradient"] > 0.0:
            lossGradient = (
                torch.mean((gradData.double() - self.gradTrue.double()) ** 2)
                .float()
                .to(self.device)
            )

        return lossData, lossGradient, residualInt

    def evaluationFunction(self, yEvaluation):
        """
        Evaluate the performance on out of sample points.

        Parameters:
        - yEvaluation: Tensor containing the predicted values for the out of sample points.

        Returns:
        - meanSquaredError: Mean squared error between the predicted values and the true values.
        """
        # For two steps training, we need to remove the z-shift
        if self.correctShift:
            yEvaluation -= yEvaluation.min()

        meanSquaredError = (
            torch.mean((yEvaluation.double() - self.yEvaluationTrue.double()) ** 2)
            .float()
            .to(self.device)
        )
        return meanSquaredError

    def _computeHamiltonJacobiEquation(self, x, gradV):
        """
        Computes the Hamilton-Jacobi equation for a given state `x` and gradient `gradV`.

        Parameters:
            x (float): The state value.
            gradV (float): The gradient of the value function.

        Returns:
            float: The computed Hamilton-Jacobi equation value.
        """
        return (
            self.computeGxTerm(gradV)
            + self.computeFxTerm(x, gradV)
            + self.computeLxTerm(x)
        )

    def _getDataPoints(self, dataPointCount):
        """
        Returns a list of data points sampled from the data sampler.

        Parameters:
        - dataPointCount (int): The number of data points to sample.

        Returns:
        - list: A list of data points sampled from the data sampler.
        """
        return self.dataSampler.samplePoints(dataPointCount)

    @abstractmethod
    def computeFxTerm(self, x, gradV):
        pass

    @abstractmethod
    def computeGxTerm(self, gradV):
        pass

    @abstractmethod
    def computeLxTerm(self, x):
        pass

    @abstractmethod
    def dataValueFunction(self, x):
        pass

    @abstractmethod
    def dataValueFunctionDerivative(self, x):
        pass

    @abstractmethod
    def getEvaluationPoints(self):
        pass

    @abstractmethod
    def groundTruthSolution(self):
        pass


class LinearQuadraticRegulator(HamiltonJacobiBellman):
    """Represents a linear quadratic regulator problem.

    The problem is defined by the following dynamics:
    y' = Ay + Bu

    The cost functional is defined as:
    J = 1/2 y^T Q y + 1/2 u^T R u

    The correspondance with the Hamilton-Jacobi-Bellman (HJB) problem is:
    f(y) = Ay
    g(y) = B
    l(y) = 1/2 y^T Q y
    beta ||u||^2 = 1/2 u^T R u

    Attributes:
        network (object): The neural network used for solving the HJB equation.
        gamma (float): The discount factor.
        dim (int): The dimension of the system.
        correctShift (bool): Flag indicating whether to apply a correction shift.

    """

    def __init__(self, network, gamma, dim, correctShift=False):
        """Initializes the LinearQuadraticRegulator class.

        Args:
            network (object): The neural network used for solving the HJB equation.
            gamma (float): The discount factor.
            dim (int): The dimension of the system.
            correctShift (bool, optional): Flag indicating whether to apply a correction shift.
                Defaults to False.

        """
        self.dim = dim
        domain = [(-1, 1)] * self.dim
        HamiltonJacobiBellman.__init__(
            self,
            network=network,
            domain=domain,
            beta=0.1,
            gamma=gamma,
            correctShift=correctShift,
        )
        self.A = torch.eye(dim).to(self.device)
        self.B = torch.eye(dim).to(self.device)
        self.Q = torch.eye(dim).to(self.device)

    def computeFxTerm(self, x, gradV):
        """Computes the f(x) term in the HJB equation.

        Args:
            x (torch.Tensor): The state variable.
            gradV (torch.Tensor): The gradient of the value function.

        Returns:
            torch.Tensor: The computed f(x) term.

        """
        productFx = (
            torch.einsum("ni, ij, nj -> n", gradV, self.A, x)
            .reshape(-1, 1)
            .to(self.device)
        )
        return productFx

    def computeGxTerm(self, gradV):
        """Computes the g(x) term in the HJB equation.

        Args:
            gradV (torch.Tensor): The gradient of the value function.

        Returns:
            torch.Tensor: The computed g(x) term.

        """
        productGx = (
            -1.0
            / (4 * self.beta)
            * torch.einsum("ni, ij, nj -> n", gradV, self.B, gradV)
            .reshape(-1, 1)
            .to(self.device)
        )
        return productGx

    def computeLxTerm(self, x):
        """Computes the l(x) term in the HJB equation.

        Args:
            x (torch.Tensor): The state variable.

        Returns:
            torch.Tensor: The computed l(x) term.

        """
        productLx = 0.5 * torch.einsum("ni, ij, nj -> n", x, self.Q, x).reshape(
            -1, 1
        ).to(self.device)
        return productLx

    def dataValueFunction(self, x):
        """Computes the value function for a given state variable.

        Args:
            x (torch.Tensor): The state variable.

        Returns:
            torch.Tensor: The computed value function.

        """
        alpha1 = 1.0 / 5 * (1 + np.sqrt(6))
        P = alpha1 * torch.eye(n=self.dim).to(self.device)
        productValueFunction = 0.5 * torch.einsum("ni, ij, nj -> n", x, P, x).reshape(
            -1, 1
        ).to(self.device)
        return productValueFunction

    def dataValueFunctionDerivative(self, x):
        """Computes the derivative of the value function for a given state variable.

        Args:
            x (torch.Tensor): The state variable.

        Returns:
            torch.Tensor: The computed derivative of the value function.

        """
        alpha1 = 1.0 / 5 * (1 + np.sqrt(6))
        P = alpha1 * torch.eye(n=self.dim).to(self.device)
        productValueFunctionDerivative = torch.einsum("ij, nj -> ni", P, x).to(
            self.device
        )
        return productValueFunctionDerivative

    def groundTruthSolution(self, xEvaluation):
        """Computes the ground truth solution for a given state variable.

        Args:
            xEvaluation (torch.Tensor): The state variable.

        Returns:
            torch.Tensor: The computed ground truth solution.

        """
        groundTruth = self.dataValueFunction(xEvaluation).reshape(-1, 1)
        return groundTruth

    def getEvaluationPoints(self):
        raise NotImplementedError


class LinearQuadraticRegulator2D(LinearQuadraticRegulator):
    """
    A class representing a 2D linear quadratic regulator.

    Args:
        network (object): The network object used for the regulator.
        gamma (float): The discount factor.
        correctShift (bool, optional): Whether to correct the shift. Defaults to False.
    """

    def __init__(self, network, gamma, correctShift=False):
        LinearQuadraticRegulator.__init__(
            self, network, gamma, dim=2, correctShift=correctShift
        )

    def getEvaluationPoints(self):
        """
        Returns the evaluation points for the regulator.

        Returns:
            list: The evaluation points.
        """
        return self.dataSampler.sampleGrid(nPoint=100)


class LinearQuadraticRegulator10D(LinearQuadraticRegulator):
    """
    A class representing a linear quadratic regulator for a 10-dimensional system.

    Args:
        network (object): The network object used for the regulator.
        gamma (float): The discount factor.
        correctShift (bool, optional): Whether to correct the shift. Defaults to False.
    """

    def __init__(self, network, gamma, correctShift=False):
        LinearQuadraticRegulator.__init__(
            self, network, gamma, dim=10, correctShift=correctShift
        )

    def getEvaluationPoints(self):
        """
        Returns the evaluation points for the regulator.

        Returns:
            list: A list of evaluation points.
        """
        return self.dataSampler.samplePoints(pointCount=10000)


class LinearQuadraticRegulatorND(LinearQuadraticRegulator):
    """
    A class representing a linear quadratic regulator for a system with multiple dimensions.

    Args:
        network (object): The network object used for the regulator.
        gamma (float): The discount factor.
        dim (int): The number of dimensions in the system.
        correctShift (bool, optional): Whether to correct the shift. Defaults to False.
    """

    def __init__(self, network, gamma, dim, correctShift=False):
        LinearQuadraticRegulator.__init__(
            self, network, gamma, dim, correctShift=correctShift
        )

    def getEvaluationPoints(self):
        """
        Returns the evaluation points for the regulator.

        Returns:
            list: The evaluation points.
        """
        return self.dataSampler.samplePoints(pointCount=10000)


class NonLinear(HamiltonJacobiBellman):
    """The problem is:
    y' = Ay + Bu
    J = int 1/2 y^T Q y  + 1/2 u^T R u

    The correspondance with the HJB problem is:
    f(y) = Ay
    g(y) = B
    l(y) = 1/2 y^T Q y
    beta ||u||^2 = 1/2 u^T R u

    R is not a parameter, instead beta=0.5

    A = [[0, 1], [eps * x0**2, 0]]
    B = [[0], [1]]
    Q = Id

    """

    def __init__(self, network, gamma, correctShift=False, dim=2, eps=1.0):
        domain = [(-1, 1)] * dim
        HamiltonJacobiBellman.__init__(
            self,
            network=network,
            domain=domain,
            beta=0.5,
            gamma=gamma,
            correctShift=correctShift,
        )
        self.B = torch.tensor([[0, 0], [0, 1]]).float().to(self.device)
        self.Q = torch.eye(dim).to(self.device)
        self.eps = eps
        self.true_solution = self._loadTrueSolution()

    def computeFxTerm(self, x, gradV):
        # we need to build a stacked matrix because A depends on x
        stackedMatrices = torch.zeros((x.shape[0], x.shape[1], x.shape[1])).to(
            self.device
        )
        stackedMatrices[:, 0, 1] = 1
        stackedMatrices[:, 1, 0] = self.eps * x[:, 0] ** 2
        productFx = (
            torch.einsum("ni, nij, nj -> n", gradV, stackedMatrices, x)
            .reshape(-1, 1)
            .to(self.device)
        )
        return productFx

    def computeGxTerm(self, gradV):
        productGx = (
            -1.0
            / (4 * self.beta)
            * torch.einsum("ni, ij, nj -> n", gradV, self.B, gradV)
            .reshape(-1, 1)
            .to(self.device)
        )
        return productGx

    def computeLxTerm(self, x):
        productLx = 0.5 * torch.einsum("ni, ij, nj -> n", x, self.Q, x).reshape(
            -1, 1
        ).to(self.device)
        return productLx

    def dataMatrixFunction(self, x):
        stackedVectors = torch.zeros((x.shape[0], 3)).to(self.device)

        p12 = lambda x: self.eps * x**2 + torch.sqrt((self.eps**2) * x**4 + 1)
        p22 = lambda x: torch.sqrt(1 + 2 * p12(x))
        p11 = lambda x: torch.sqrt((self.eps**2) * x**4 + 1) * p22(x)

        stackedVectors[:, 0] = p11(x[:, 0])
        stackedVectors[:, 1] = p12(x[:, 0])
        stackedVectors[:, 2] = p22(x[:, 0])

        return stackedVectors

    def dataValueFunction(self, x):
        stackedMatrices = torch.zeros((x.shape[0], x.shape[1], x.shape[1])).to(
            self.device
        )

        p12 = lambda x: self.eps * x**2 + torch.sqrt((self.eps**2) * x**4 + 1)
        p22 = lambda x: torch.sqrt(1 + 2 * p12(x))
        p11 = lambda x: torch.sqrt((self.eps**2) * x**4 + 1) * p22(x)

        stackedMatrices[:, 0, 0] = p11(x[:, 0])
        stackedMatrices[:, 0, 1] = p12(x[:, 0])
        stackedMatrices[:, 1, 0] = p12(x[:, 0])
        stackedMatrices[:, 1, 1] = p22(x[:, 0])

        productValueFunction = 0.5 * torch.einsum(
            "ni, nij, nj -> n", x, stackedMatrices, x
        ).reshape(-1, 1).to(self.device)

        return productValueFunction

    def dataValueFunctionDerivative(self, x):
        stackedVectors = torch.zeros((x.shape[0], x.shape[1])).to(self.device)

        p12 = lambda x: self.eps * x**2 + torch.sqrt(((self.eps) ** 2) * x**4 + 1)
        p22 = lambda x: torch.sqrt(1 + 2 * p12(x))
        p11 = lambda x: torch.sqrt(((self.eps) ** 2) * x**4 + 1) * p22(x)

        p12_deriv = lambda x: 2 * self.eps * x + (self.eps**2) * x**3 / torch.sqrt(
            ((self.eps) ** 2) * x**4 + 1
        )
        p22_deriv = lambda x: p12_deriv(x) / p22(x)
        p11_deriv = lambda x: 2 * self.eps * x**3 / torch.sqrt(
            ((self.eps) ** 2) * x**4 + 1
        ) * p22(x) + torch.sqrt(((self.eps) ** 2) * x**4 + 1) * p22_deriv(x)

        v2_deriv = lambda x, y: p12(x) * x + p22(x) * y
        v1_deriv = lambda x, y: 0.5 * (
            p11_deriv(x) * x**2
            + 2 * x * p11(x)
            + 2 * p12_deriv(x) * x * y
            + 2 * p12(x) * y
            + y**2 * p22_deriv(x)
        )

        stackedVectors[:, 0] = v1_deriv(x[:, 0], x[:, 1])
        stackedVectors[:, 1] = v2_deriv(x[:, 0], x[:, 1])

        return stackedVectors

        # stackedMatrices = torch.zeros((x.shape[0], x.shape[1], x.shape[1])).to(self.device)

        # p12 = lambda x: x**2 + torch.sqrt(x**4 + 1)
        # p22 = lambda x: torch.sqrt( 1 + 2*p12(x) )
        # p11 = lambda x: (torch.sqrt(x**4 + 1)) * p22(x)

        # stackedMatrices[:, 0, 0] = p11( x[:, 0] )
        # stackedMatrices[:, 0, 1] = p12( x[:, 0] )
        # stackedMatrices[:, 1, 0] = p12( x[:, 0] )
        # stackedMatrices[:, 1, 1] = p22( x[:, 0] )

        # gradV = torch.einsum('nij, nj -> ni', stackedMatrices, x)

        # return gradV

    def _loadTrueSolution(
        self, solutionFilename="non_linear_true_solution/neural_net/non_linear_true.csv"
    ):
        trueSolution = pd.read_csv(solutionFilename).drop(columns="Unnamed: 0")
        trueSolution = (
            torch.tensor(trueSolution.to_numpy()).reshape(-1, 1).to(self.device)
        )
        return trueSolution

    def groundTruthSolution(self, xEvaluation):
        return self.true_solution

    def getEvaluationPoints(self):
        return self.dataSampler.sampleGrid(nPoint=100)


class CuckerSmale(HamiltonJacobiBellman):
    """The problem is:
    x' = Ax + Bu
    J = int 1/N x^T Q x  + 1/N u^T R u
    """

    def __init__(self, network, gamma, correctShift=False, dim=20):
        self.dim = dim
        self.domain = [(-3, 3)] * (2 * self.dim)
        HamiltonJacobiBellman.__init__(
            self,
            network=network,
            domain=self.domain,
            beta=0.5,
            gamma=gamma,
            correctShift=correctShift,
        )

        # its actually BBT
        # self.B = torch.vstack( [torch.zeros((self.dim, self.dim)), torch.eye(self.dim)] ).float()
        self.B = (
            torch.diag(torch.tensor([0] * self.dim + [1] * self.dim))
            .float()
            .to(self.device)
        )
        self.Q = 1.0 / self.dim * torch.eye(2 * self.dim).to(self.device)

    def computeFxTerm(self, x, gradV):
        # we need to build a stacked matrix because A depends on x
        stackedMatrices = torch.zeros((x.shape[0], x.shape[1], x.shape[1])).to(
            self.device
        )

        # fill small A matrix
        interationCoefficient = lambda yi, yj: 1.0 / (1.0 + (yi - yj) ** 2)

        for i in range(self.dim):
            for j in range(self.dim):
                if i == j:
                    stackedMatrices[:, self.dim + i, self.dim + j] = (
                        -1
                        / self.dim
                        * sum(
                            [
                                interationCoefficient(yi=x[:, i], yj=x[:, k])
                                for k in range(self.dim)
                            ]
                        )
                    )
                else:
                    stackedMatrices[:, self.dim + i, self.dim + j] = (
                        1 / self.dim * interationCoefficient(x[:, i], x[:, j])
                    )

        # fill the rest of the matrix, namely identity on the upper right corner
        for i in range(self.dim):
            stackedMatrices[:, i, self.dim + i] = 1

        productFx = (
            torch.einsum("ni, nij, nj -> n", gradV, stackedMatrices, x)
            .reshape(-1, 1)
            .to(self.device)
        )
        return productFx

    def computeGxTerm(self, gradV):
        productGx = (
            -self.dim
            * 1.0
            / (4 * self.beta)
            * torch.einsum("ni, ij, nj -> n", gradV, self.B, gradV)
            .reshape(-1, 1)
            .to(self.device)
        )
        return productGx

    def computeLxTerm(self, x):
        productLx = 0.5 * torch.einsum("ni, ij, nj -> n", x, self.Q, x).reshape(
            -1, 1
        ).to(self.device)
        return productLx

    def dataMatrixFunction(self, x):
        dataMatrixFunction = np.loadtxt("matrixFunction.csv")
        dataMatrixFunction = torch.tensor(dataMatrixFunction).to(self.device)
        return dataMatrixFunction

    def dataValueFunction(self, x):
        dataValueFunction = np.loadtxt("valueFunction.csv")
        dataValueFunction = (
            torch.tensor(dataValueFunction).reshape(-1, 1).to(self.device)
        )
        return dataValueFunction

    def dataValueFunctionDerivative(self, x):
        dataValueFunctionDerivative = np.loadtxt("valueFunctionDerivative.csv")
        dataValueFunctionDerivative = (
            torch.tensor(dataValueFunctionDerivative)
            .reshape(-1, 2 * self.dim)
            .to(self.device)
        )
        return dataValueFunctionDerivative

    def getDataPoints(self, dataPointCount):
        sampledPoints = np.loadtxt("sampledPoints.csv")
        sampledPoints = (
            torch.tensor(sampledPoints, requires_grad=True)
            .reshape(-1, 2 * self.dim)
            .float()
            .to(self.device)
        )
        return sampledPoints[:dataPointCount]

    def groundTruthSolution(self):
        pass

    def getEvaluationPoints(self):
        pass
