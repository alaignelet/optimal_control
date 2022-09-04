import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import time
from abc import ABC, abstractmethod
from scipy.special import expit, logit
from scipy.stats import ortho_group

def stackWeights(network):
    weights = np.array([])
    for layer in network.layers:
        if type(layer) == nn.Linear:
            weights = np.concatenate([
                weights,
                layer.weight.detach().numpy().flatten(),
                layer.bias.detach().numpy().flatten()
            ])
    return weights

class GenerateData(object):
    ''' Generate data points.
    '''

    def __init__(self, domain):
        self.domain = domain
        self.dim = len(self.domain)

    def samplePoints(self, pointCount):
        ''' Sample the interior of the domain.

            Parameters:
                domain: tuple of lists
                    eg: ([0, 1], [0, 1])
                pointCount: int
                    eg: 100

            Returns:
                xPoint: list of numpy.array
        '''

        xPoint = []
        for i in range( self.dim ):
            xPoint.append(np.random.uniform( low=self.domain[i][0], high=self.domain[i][1], size=(pointCount, 1)) )
            
        xPoint = torch.tensor( xPoint, requires_grad=True )[:, :, 0].T.float()
        return xPoint
    

    def sampleGrid(self, nPoint=100):
        gridPoints = np.meshgrid( *[np.linspace(-1, 1, nPoint) for i in range(self.dim)] )
        gridPoints = torch.tensor( list(gridPoints) ).T.float().reshape(-1, self.dim)
        return gridPoints



class NeuralNet(nn.Module):
    ''' Neural Network used as a mapping function.
        Glorot initialisation.
    '''

    def __init__(self, layers, quadraticForm=False, useAdditionalModel=False, imposePsd=False, imposeCholesky=False, **parameters):
        super(NeuralNet, self).__init__()
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.layers = layers
        self.model = self._buildModel(layers, imposePsd).to(self.device)
        self.quadraticForm = quadraticForm
        self.useAdditionalModel = useAdditionalModel
        if self.useAdditionalModel:
            self.additionalModel = self._buildModel(layers=[ layers[0], 20, layers[0] ], imposePsd=False).to(self.device)
        #np.random.seed(0)
        self.imposePsd = imposePsd
        if self.imposePsd:
            self.countMatrices = parameters.get('countMatrices', 1)
            # self.orthogonalMatrix = [torch.tensor( ortho_group.rvs(layers[0]) ).float() for i in range(self.countMatrices)]
            self.orthogonalMatrix = [ torch.tensor( [[-1, -1], [-1, 1]]).float()  / np.sqrt(2) ]
        self.imposeCholesky = imposeCholesky
            
    def _buildLayers(self, layers, imposePsd):
        neuralNetLayers = []
        for i in range(len(layers) - 2):
            neuralNetLayers.append(nn.Linear(in_features=layers[i], out_features=layers[i + 1]))
            neuralNetLayers.append( nn.Sigmoid() )
        neuralNetLayers.append(nn.Linear(in_features=layers[-2], out_features=layers[-1]))
        if imposePsd:
            neuralNetLayers.append( nn.Softplus() )     
        return neuralNetLayers
         
        
    def _buildModel(self, layers, imposePsd):
        neuralNetLayers = self._buildLayers(layers, imposePsd)
        neuralNetModel = nn.Sequential( *neuralNetLayers )
        neuralNetModel = neuralNetModel.apply(self._normalInit)
        return neuralNetModel

    
    def _normalInit(self, layer):
        if type(layer) == nn.Linear:
            torch.nn.init.xavier_normal_(layer.weight)
            
            
    def partialDerivative(self, tensorToDerive, x):
        grad = torch.autograd.grad(
                outputs=tensorToDerive, inputs=x, grad_outputs=torch.ones_like(tensorToDerive), create_graph=True)[0]
        
        return grad
            
            
    def computeValueFunctionDerivative(self, x):
        valueFunction = self.computeValueFunction(x)
        return self.partialDerivative( tensorToDerive=valueFunction, x=x)


    def computeValueFunction(self, x):
        if self.quadraticForm:
            return self._matrixEvaluated(x)

        elif self.imposePsd:
            return self._matrixEvaluatedPsd(x)

        elif self.imposeCholesky:
            return self._choleskyEvaluated(x)

        else:
            return self._directValueFunction(x)
        
    
    def _directValueFunction(self, x):
        ''' The output of the network is the value function directly.
        '''
        return self.model(x)
    

    def _choleskyEvaluated(self, x):
        ''' The output of the network is the symetric matrix P.
        '''        
        
        dim = x.shape[1]

        # the below is SUPER fast
        stackedMatrices = torch.zeros((x.shape[0], dim, dim)).to(self.device)
        outputModel = self.model(x)

        inds = np.triu_indices( dim, k=1 )
        k = 0
        for i, j in zip( inds[0], inds[1] ):
            stackedMatrices[:, i, j] = outputModel[:, k]
            k += 1

        # we need the diagonal terms to be positive
        for i in range(dim):
            stackedMatrices[:, i, i] = torch.exp( outputModel[:, k] )
            k += 1

        stackedP = torch.einsum( 'nji, njk -> nik', stackedMatrices, stackedMatrices)

        valueFunction = 0.5 * torch.einsum('ni, nij, nj -> n', x, stackedP, x).reshape(-1, 1).to(self.device)

        return valueFunction  

    def _matrixEvaluatedPsd(self, x):
        ''' The output of the network is the eigenvalues of PSD matrix P.
        '''

        dim = x.shape[1]

        # the below is SUPER fast
        stackedMatrices = torch.zeros((x.shape[0], dim, dim)).to(self.device)
        outputModel = self.model(x)

        valueFunction = 0
        for idx, Q in zip( [i*dim for i in range(self.countMatrices)], self.orthogonalMatrix):
            stackedMatrices = torch.zeros((x.shape[0], dim, dim))
            
            for i in range(dim):
                stackedMatrices[:, i, i] = outputModel[:, idx + i]

            stackedMatricesPsd = Q @ stackedMatrices @ Q.T

            valueFunction += 0.5 * torch.einsum('ni, nij, nj -> n', x, stackedMatricesPsd, x).reshape(-1, 1)

        return valueFunction  

    
    def _matrixEvaluated(self, x):
        ''' The output of the network is the symetric matrix P.
        '''        
        
        dim = x.shape[1]

        # the below is SUPER fast
        stackedMatrices = torch.zeros((x.shape[0], dim, dim)).to(self.device)
        outputModel = self.model(x)

        inds = np.triu_indices( dim )
        k = 0
        for i, j in zip( inds[0], inds[1] ):
            stackedMatrices[:, i, j] = outputModel[:, k]
            stackedMatrices[:, j, i] = outputModel[:, k]
            k += 1

        valueFunction = 0.5 * torch.einsum('ni, nij, nj -> n', x, stackedMatrices, x).reshape(-1, 1).to(self.device)

        return valueFunction        
        
        
    def train(self, feedDict, lrs, iterations, useTestData, verbose=False):
        ''' Training function.
        '''
        gamma = feedDict['gamma']
        lossFunction = feedDict['lossFunction']
        evaluationFunction = feedDict['evaluationFunction']

        xInt = feedDict['xInt'].to(self.device)
        xData = feedDict['xData'].to(self.device)

        interiorPointCount = xInt.shape[0]
        dataSampler = feedDict['dataSampler']

        # test
        if useTestData:
            xDataTest = feedDict['xDataTest'].to(self.device)
            yTrueTest = feedDict['yTrueTest'].to(self.device)
            gradTrueTest = feedDict['gradTrueTest'].to(self.device)

        else:
            xEvaluation = feedDict['xEvaluation'].to(self.device)

        # network dependend quantities
        matrixData = torch.zeros( (xInt.shape[0], self.layers[-1]) ).to(self.device)
        gradInt = torch.zeros( xInt.shape ).to(self.device)
        yData = torch.zeros( (xData.shape[0], 1) ).to(self.device)
        gradData = torch.zeros( xData.shape ).to(self.device)
        errorDerivative = torch.zeros( xData.shape ).to(self.device)

        epochTotal = 0
        info = []

        for lr, iteration in zip(lrs, iterations):
            self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr)
            
            if self.useAdditionalModel:
                additionalOptimizer = torch.optim.Adam(params=self.additionalModel.parameters(), lr=lr)
            
            for epoch in range(iteration):
                # xInt = dataSampler.samplePoints(interiorPointCount).to(self.device)

                # compute model dependent quantities
                if gamma['matrix'] > 0:
                    matrixData = self.model(xData)

                if gamma['data'] > 0.:
                    yData = self.computeValueFunction(xData)

                if gamma['gradient'] > 0.:
                    gradData = self.computeValueFunctionDerivative(xData)

                if gamma['residual'] > 0.:
                    gradInt = self.computeValueFunctionDerivative(xInt)
                    
                if self.useAdditionalModel:
                    errorDerivative = self.additionalModel(xData)
                    
                # compute loss and backpropagate
                lossData, lossGrad, lossResidual, lossMatrix = lossFunction(xInt, gradInt, yData, gradData, matrixData, errorDerivative)
                loss = (
                    gamma['data'] * lossData + 
                    gamma['gradient'] * lossGrad + 
                    gamma['residual'] * lossResidual +
                    gamma['matrix'] * lossMatrix
                     )
                
                if self.useAdditionalModel:
                    self.optimizer.zero_grad()
                    additionalOptimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    additionalOptimizer.step()
                    
                else:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()


                # print logs
                if (epochTotal % 100 == 0):
                    if verbose:
                        print('%d / %d (%d / %d), lr:%.1e, loss:%.2e (data: %.2e, grad: %.2e, res: %.2e, mat: %.2e)' % (
                            epochTotal, sum(iterations), epoch, iteration, lr, loss, lossData, lossGrad, lossResidual, lossMatrix
                            )
                        )

                    mse = torch.tensor(0.).to(self.device)

                    if useTestData:
                    # check on test set
                        # if gamma['data'] > 0.:
                        yDataTest = self.computeValueFunction(xDataTest)
                        print('yDataTest', yDataTest[:5])
                        print('yTrueTest', yTrueTest[:5])
                        lossDataTest = torch.mean( (yDataTest.double() - yTrueTest.double())**2 ).float().to(self.device)
                        print('lossDataTest: %.2e'%lossDataTest.detach().cpu().numpy().item())
                        mse = lossDataTest

                        # if  gamma['gradient'] > 0.:
                        gradDataTest = self.computeValueFunctionDerivative(xDataTest)
                        lossGradientTest = torch.mean( (gradDataTest.double() - gradTrueTest.double())**2 ).float().to(self.device)
                        print('lossGradientTest: %.2e'%lossGradientTest.detach().cpu().numpy().item())
                        
                    else:
                        yEvaluation = self.computeValueFunction(xEvaluation)
                        mse = evaluationFunction(yEvaluation)
                        # print('mse: %.2e' %mse)

                epochTotal += 1
                info_dict = {
                    'xData': xData,
                    'epoch': epochTotal,
                    'mse': mse.detach().cpu().numpy().item(),
                    'loss': loss.detach().cpu().numpy().item()
                    }
                info.append(info_dict)

        return pd.DataFrame( info )



class HamiltonJacobiBellman(ABC):
    ''' Build the PDE solver for Hamilton Jacobi.
        The problem is:
        y' = f(y) + g(y)u
        J = int l(y) + beta ||u||^2

        The boundary condition is V(T,x).
    '''

    def __init__(self, network, domain, beta, gamma, correctShift):
        self.network = network
        self.domain = domain
        self.beta = beta
        self.gamma = gamma
        self.dataSampler = GenerateData(domain=self.domain)
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.correctShift = correctShift


    def train(self, interiorPointCount, dataPointCount, lrs, iterations, useTestData=False, verbose=False):
        ''' Generate data and train.
        '''

        # interior points
        xInt = self.dataSampler.samplePoints(interiorPointCount).to(self.device)
        print('xInt: ', xInt.shape)
        
        if useTestData:

            dataPointCountTrain = int( 0.8 * dataPointCount )

            # train
            xData = self.getDataPoints(dataPointCount).to(self.device)[:dataPointCountTrain]
            self.matrixTrue = self.dataMatrixFunction(xData.detach()).to(self.device)[:dataPointCountTrain]
            self.yTrue = self.dataValueFunction(xData.detach()).to(self.device)[:dataPointCountTrain]
            self.gradTrue = self.dataValueFunctionDerivative(xData.detach()).to(self.device)[:dataPointCountTrain]
            print('xData: ', xData.shape)

            # test
            xDataTest = self.getDataPoints(dataPointCount).to(self.device)[dataPointCountTrain:]
            yTrueTest = self.dataValueFunction(xData.detach()).to(self.device)[dataPointCountTrain:]
            gradTrueTest = self.dataValueFunctionDerivative(xData.detach()).to(self.device)[dataPointCountTrain:]
            print('xDataTest: ', xDataTest.shape)

            feedDict = {
                'xInt': xInt,
                'xData': xData,
                'gamma': self.gamma,
                'lossFunction': self.lossFunction,
                'evaluationFunction': self.evaluationFunction,
                'groundTruthSolution': self.groundTruthSolution,
                'xDataTest': xDataTest,
                'yTrueTest': yTrueTest,
                'gradTrueTest': gradTrueTest,
                'dataSampler': self.dataSampler
            }
        
        else:
            # space
            xEvaluation = self.getEvaluationPoints().to(self.device)
            self.yEvaluationTrue = self.groundTruthSolution(xEvaluation.detach())

            # train
            xData = self.getDataPoints(dataPointCount).to(self.device)
            self.matrixTrue = self.dataMatrixFunction(xData.detach()).to(self.device)
            self.yTrue = self.dataValueFunction(xData.detach()).to(self.device)
            self.gradTrue = self.dataValueFunctionDerivative(xData.detach()).to(self.device)
            print('xData: ', xData.shape)


            feedDict = {
                'xEvaluation': xEvaluation,
                'xInt': xInt,
                'xData': xData,
                'gamma': self.gamma,
                'lossFunction': self.lossFunction,
                'evaluationFunction': self.evaluationFunction,
                'groundTruthSolution': self.groundTruthSolution,
                'dataSampler': self.dataSampler
            }
        
        lossValues = self.network.train(feedDict, lrs, iterations, useTestData, verbose)

        return lossValues


    def lossFunction(self, xInt, gradInt, yData, gradData, matrixData, errorDerivative):
        ''' Compute the loss function.
        '''

        lossMatrix = torch.tensor([0]).float().to(self.device)
        residualInt = torch.tensor([0]).float().to(self.device)
        lossData = torch.tensor([0]).float().to(self.device)
        lossGradient = torch.tensor([0]).float().to(self.device)

        if self.gamma['matrix'] > 0.:
            lossMatrix = torch.mean( (matrixData.double() - self.matrixTrue.double())**2 ).float().to(self.device)


        if self.gamma['residual'] > 0.:
            equation = self.computeHamiltonJacobiEquation(xInt, gradInt)
            residualInt = torch.mean( equation**2 ).to(self.device)


        if self.gamma['data'] > 0.:
            lossData = torch.mean( (yData.double() - self.yTrue.double())**2 ).float().to(self.device)


        if self.gamma['gradient'] > 0.:
            lossGradient = torch.mean( (gradData.double() - self.gradTrue.double() - errorDerivative.double())**2 ).float().to(self.device)

        return lossData, lossGradient, residualInt, lossMatrix
    
    
    def evaluationFunction(self, yEvaluation):
        ''' Evaluate the performance on out of sample points.
        '''

        # for two steps training, we need to remove the z-shift
        if self.correctShift:
            yEvaluation -= yEvaluation.min()

        meanSquaredError = torch.mean( (yEvaluation.double() - self.yEvaluationTrue.double())**2 ).float().to(self.device)
        return meanSquaredError



    def computeHamiltonJacobiEquation(self, x, gradV):
        return self.computeGxTerm(gradV) + self.computeFxTerm(x, gradV) + self.computeLxTerm(x) 
    
    
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
    def dataMatrixFunction(self, x):
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
    ''' The problem is:
        y' = Ay + Bu
        J = int 1/2 y^T Q y  + 1/2 u^T R u

        The correspondance with the HJB problem is:
        f(y) = Ay
        g(y) = B
        l(y) = 1/2 y^T Q y
        beta ||u||^2 = 1/2 u^T R u

        R is not a parameter, instead beta=0.5
    '''

    def __init__(self, network, gamma, dim, correctShift=False):
        self.dim = dim
        domain = [(-1, 1)] * self.dim
        HamiltonJacobiBellman.__init__(self, network=network, domain=domain, beta=0.1, gamma=gamma, correctShift=correctShift)
        self.A = torch.eye(dim).to(self.device)
        self.B = torch.eye(dim).to(self.device)
        self.Q = torch.eye(dim).to(self.device)

    
    def computeFxTerm(self, x, gradV):
        productFx = torch.einsum('ni, ij, nj -> n', gradV, self.A, x).reshape(-1, 1).to(self.device)
        return productFx

        
    def computeGxTerm(self, gradV):
        productGx = -1. / (4 * self.beta) * torch.einsum('ni, ij, nj -> n', gradV, self.B, gradV).reshape(-1, 1).to(self.device)
        return productGx 
    
    
    def computeLxTerm(self, x):
        productLx = 0.5 * torch.einsum('ni, ij, nj -> n', x, self.Q, x).reshape(-1, 1).to(self.device)
        return productLx
        

    def dataValueFunction(self, x):
        alpha1 = 1./5 * ( 1 + np.sqrt(6) )
        P = alpha1 * torch.eye(n=self.dim).to(self.device)
        productValueFunction = 0.5 * torch.einsum('ni, ij, nj -> n', x, P, x).reshape(-1, 1).to(self.device)
        return productValueFunction
    
    
    def dataValueFunctionDerivative(self, x):
        alpha1 = 1./5 * ( 1 + np.sqrt(6) )
        P = alpha1 * torch.eye(n=self.dim).to(self.device)
        productValueFunctionDerivative = torch.einsum('ij, nj -> ni', P, x).to(self.device)
        return productValueFunctionDerivative
    

    def dataMatrixFunction(self, x):
        alpha1 = 1./5 * ( 1 + np.sqrt(6) )
        P = alpha1 * torch.eye(n=self.dim)
        inds = np.triu_indices(self.dim)
        return P[inds].repeat(x.shape[0], 1)


    def getDataPoints(self, dataPointCount):
        sampledPoints = self.dataSampler.samplePoints(dataPointCount)
        return sampledPoints


    def groundTruthSolution(self, xEvaluation):
        groundTruth = self.dataValueFunction(xEvaluation).reshape(-1, 1)
        return groundTruth



class LinearQuadraticRegulator2D(LinearQuadraticRegulator):
    def __init__(self, network, gamma, correctShift=False):
        LinearQuadraticRegulator.__init__(self, network, gamma, dim=2, correctShift=correctShift)

    def getEvaluationPoints(self):
        return self.dataSampler.sampleGrid(nPoint=100)


class LinearQuadraticRegulator10D(LinearQuadraticRegulator):
    def __init__(self, network, gamma, correctShift=False):
        LinearQuadraticRegulator.__init__(self, network, gamma, dim=10, correctShift=correctShift)

    def getEvaluationPoints(self):
        return self.dataSampler.samplePoints(pointCount=10000)
        
    
class LinearQuadraticRegulatorND(LinearQuadraticRegulator):
    def __init__(self, network, gamma, dim, correctShift=False):
        LinearQuadraticRegulator.__init__(self, network, gamma, dim, correctShift=correctShift)

    def getEvaluationPoints(self):
        return self.dataSampler.samplePoints(pointCount=10000)

    



class NonLinear(HamiltonJacobiBellman):
    ''' The problem is:
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
        
    '''

    def __init__(self, network, gamma, correctShift=False, dim=2, eps=1.):
        domain = [(-1, 1)] * dim
        HamiltonJacobiBellman.__init__(self, network=network, domain=domain, beta=0.5, gamma=gamma, correctShift=correctShift)
        self.B = torch.tensor([[0, 0],[0, 1]]).float().to(self.device)
        self.Q = torch.eye(dim).to(self.device)
        self.eps = eps
        self.true_solution = self._loadTrueSolution()

        
    def computeFxTerm(self, x, gradV):
        # we need to build a stacked matrix because A depends on x
        stackedMatrices = torch.zeros((x.shape[0], x.shape[1], x.shape[1])).to(self.device)
        stackedMatrices[:, 0, 1] = 1 
        stackedMatrices[:, 1, 0] = self.eps * x[:, 0]**2
        productFx = torch.einsum('ni, nij, nj -> n', gradV, stackedMatrices, x).reshape(-1, 1).to(self.device)
        return productFx
        
         
    def computeGxTerm(self, gradV):
        productGx = -1. / (4 * self.beta) * torch.einsum('ni, ij, nj -> n', gradV, self.B, gradV).reshape(-1, 1).to(self.device)
        return productGx 

    
    def computeLxTerm(self, x):
        productLx = 0.5 * torch.einsum('ni, ij, nj -> n', x, self.Q, x).reshape(-1, 1).to(self.device)
        return productLx


    def dataMatrixFunction(self, x):
        stackedVectors = torch.zeros((x.shape[0], 3)).to(self.device)

        p12 = lambda x: self.eps * x**2 + torch.sqrt( (self.eps**2) * x**4 + 1)
        p22 = lambda x: torch.sqrt( 1 + 2*p12(x) ) 
        p11 = lambda x: torch.sqrt( (self.eps**2) * x**4 + 1) * p22(x)

        stackedVectors[:, 0] = p11( x[:, 0] )
        stackedVectors[:, 1] = p12( x[:, 0] )
        stackedVectors[:, 2] = p22( x[:, 0] )

        return stackedVectors


    def dataValueFunction(self, x):
        stackedMatrices = torch.zeros((x.shape[0], x.shape[1], x.shape[1])).to(self.device)

        p12 = lambda x: self.eps * x**2 + torch.sqrt( (self.eps**2) * x**4 + 1)
        p22 = lambda x: torch.sqrt( 1 + 2*p12(x) ) 
        p11 = lambda x: torch.sqrt( (self.eps**2) * x**4 + 1) * p22(x)

        stackedMatrices[:, 0, 0] = p11( x[:, 0] )
        stackedMatrices[:, 0, 1] = p12( x[:, 0] )
        stackedMatrices[:, 1, 0] = p12( x[:, 0] )
        stackedMatrices[:, 1, 1] = p22( x[:, 0] )

        productValueFunction = 0.5 * torch.einsum('ni, nij, nj -> n', x, stackedMatrices, x).reshape(-1, 1).to(self.device)

        return productValueFunction
    
    
    def dataValueFunctionDerivative(self, x):
        stackedVectors = torch.zeros((x.shape[0], x.shape[1])).to(self.device)

        p12 = lambda x: self.eps * x**2 + torch.sqrt( ((self.eps)**2) * x**4 + 1)
        p22 = lambda x: torch.sqrt( 1 + 2*p12(x) ) 
        p11 = lambda x: torch.sqrt( ((self.eps)**2) * x**4 + 1) * p22(x)

        p12_deriv = lambda x: 2 * self.eps * x + (self.eps**2) * x**3 / torch.sqrt( ((self.eps)**2) * x**4 + 1)
        p22_deriv = lambda x: p12_deriv(x) / p22(x)
        p11_deriv = lambda x: 2 * self.eps * x**3 / torch.sqrt( ((self.eps)**2) * x**4 + 1) * p22(x) + torch.sqrt( ((self.eps)**2) * x**4 + 1) * p22_deriv(x)

        v2_deriv = lambda x, y: p12(x) * x + p22(x) * y
        v1_deriv = lambda x, y: 0.5 * ( p11_deriv(x) * x**2 + 2 * x * p11(x) + 2*p12_deriv(x) * x * y + 2*p12(x) * y + y**2 * p22_deriv(x) )

        stackedVectors[:, 0] = v1_deriv( x[:, 0], x[:, 1] )
        stackedVectors[:, 1] = v2_deriv( x[:, 0], x[:, 1] )

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

    
    def _loadTrueSolution(self, solutionFilename='non_linear_true_solution/neural_net/non_linear_true.csv'):
        trueSolution = pd.read_csv(solutionFilename).drop(columns='Unnamed: 0')
        trueSolution = torch.tensor( trueSolution.to_numpy() ).reshape(-1, 1).to(self.device)
        return trueSolution
    
    
    def getDataPoints(self, dataPointCount):
        sampledPoints = self.dataSampler.samplePoints(dataPointCount)
        return sampledPoints

    def groundTruthSolution(self, xEvaluation):
        return self.true_solution

    def getEvaluationPoints(self):
        return self.dataSampler.sampleGrid(nPoint=100)


class CuckerSmale(HamiltonJacobiBellman):
    ''' The problem is:
        x' = Ax + Bu
        J = int 1/N x^T Q x  + 1/N u^T R u        
    '''

    def __init__(self, network, gamma, correctShift=False, dim=20):
        self.dim = dim
        self.domain = [(-3, 3)] * (2 * self.dim)
        HamiltonJacobiBellman.__init__(self, network=network, domain=self.domain, beta=0.5, gamma=gamma, correctShift=correctShift)

        # its actually BBT
        #self.B = torch.vstack( [torch.zeros((self.dim, self.dim)), torch.eye(self.dim)] ).float()
        self.B = torch.diag( torch.tensor([0]*self.dim + [1]*self.dim) ).float().to(self.device)
        self.Q = 1. / self.dim * torch.eye(2 * self.dim).to(self.device)
        
    def computeFxTerm(self, x, gradV):
        # we need to build a stacked matrix because A depends on x
        stackedMatrices = torch.zeros((x.shape[0], x.shape[1], x.shape[1])).to(self.device)

        # fill small A matrix
        interationCoefficient = lambda yi, yj:  1. / ( 1. + (yi - yj)**2 )

        for i in range(self.dim):
            for j in range(self.dim):
                if i == j:
                    stackedMatrices[:, self.dim + i, self.dim + j] = - 1 / self.dim * sum( [interationCoefficient(yi=x[:, i], yj=x[:, k]) for k in range(self.dim)] )
                else:
                    stackedMatrices[:, self.dim + i, self.dim + j] = 1 / self.dim * interationCoefficient( x[:, i], x[:, j] )

        # fill the rest of the matrix, namely identity on the upper right corner
        for i in range(self.dim):
            stackedMatrices[:, i, self.dim + i] = 1

        productFx = torch.einsum('ni, nij, nj -> n', gradV, stackedMatrices, x).reshape(-1, 1).to(self.device)
        return productFx
        
        
    def computeGxTerm(self, gradV):
        productGx = -self.dim * 1. / (4 * self.beta) * torch.einsum('ni, ij, nj -> n', gradV, self.B, gradV).reshape(-1, 1).to(self.device)
        return productGx 

    
    def computeLxTerm(self, x):
        productLx = 0.5 * torch.einsum('ni, ij, nj -> n', x, self.Q, x).reshape(-1, 1).to(self.device)
        return productLx


    def dataMatrixFunction(self, x):
        dataMatrixFunction = np.loadtxt( 'matrixFunction.csv' )
        dataMatrixFunction = torch.tensor( dataMatrixFunction ).to(self.device)
        return dataMatrixFunction


    def dataValueFunction(self, x):  
        dataValueFunction = np.loadtxt( 'valueFunction.csv' )
        dataValueFunction = torch.tensor( dataValueFunction ).reshape(-1, 1).to(self.device)
        return dataValueFunction
    
    
    def dataValueFunctionDerivative(self, x):
        dataValueFunctionDerivative = np.loadtxt( 'valueFunctionDerivative.csv' )
        dataValueFunctionDerivative = torch.tensor( dataValueFunctionDerivative ).reshape(-1, 2 * self.dim).to(self.device)
        return dataValueFunctionDerivative


    def getDataPoints(self, dataPointCount):
        sampledPoints = np.loadtxt( 'sampledPoints.csv' )
        sampledPoints = torch.tensor( sampledPoints, requires_grad=True ).reshape(-1, 2 * self.dim).float().to(self.device)
        return sampledPoints[:dataPointCount]


    def groundTruthSolution(self):
        pass


    def getEvaluationPoints(self):
        pass