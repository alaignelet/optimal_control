import torch.nn as nn
from enum import Enum
from icnn import (
    ExponentialPositivity,
    LazyClippedPositivity,
    NoPositivity,
    ConvexInitialiser,
    TraditionalInitialiser,
)


class ActivationFunctionEnum(Enum):
    ELU = nn.ELU()
    HARDSHRINK = nn.Hardshrink()
    HARDTANH = nn.Hardtanh()
    LEAKYRELU = nn.LeakyReLU()
    LOGSIGMOID = nn.LogSigmoid()
    PRELU = nn.PReLU()
    RELU = nn.ReLU()
    RELU6 = nn.ReLU6()
    RRELU = nn.RReLU()
    SELU = nn.SELU()
    CELU = nn.CELU()
    GELU = nn.GELU()
    SIGMOID = nn.Sigmoid()
    SILU = nn.SiLU()
    MISH = nn.Mish()
    SOFTPLUS = nn.Softplus()
    SOFTSHRINK = nn.Softshrink()
    SOFTSIGN = nn.Softsign()
    TANH = nn.Tanh()
    TANHSHRINK = nn.Tanhshrink()
    SOFTMIN = nn.Softmin()
    SOFTMAX = nn.Softmax()
    LOGSOFTMAX = nn.LogSoftmax()


class PositivityFunctionEnum(Enum):
    FC = NoPositivity()
    CONVEX = ExponentialPositivity()
    ICNN = LazyClippedPositivity()


class InitFunctionEnum(Enum):
    TRADITIONAL = TraditionalInitialiser(gain=2.0)
    CONVEX = ConvexInitialiser()
