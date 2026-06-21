"""LSE-of-quadratics network: globally strongly convex by construction.

    V(x) = (m/2)||x||^2 + (1/tau) log sum_i exp( tau * q_i(x) ),
    q_i(x) = 0.5 x^T L_i L_i^T x + b_i^T x + c_i.

Convexity: each q_i is convex (L_i L_i^T is PSD for ANY unconstrained L_i),
log-sum-exp of convex functions is convex, and

    Hess V  >=  m I + sum_i p_i L_i L_i^T  >=  m I        (p = softmax weights)

so V is m-strongly convex on all of R^d --- a property of the architecture,
not a penalty. The parameter space (L_i, b_i, c_i) is unconstrained
Euclidean: no clipping, no positivity projections, no dead faces.

This is the smooth (log-sum-exp) version of a max-of-quadratics expansion,
i.e. the function class of max-plus methods for HJB equations (McEneaney);
the learned controller is a softmax-gated mixture of linear feedback laws:
grad V = m x + sum_i p_i(x) (A_i x + b_i).
"""

import torch
import torch.nn as nn

from neuralNet import BaseNeuralNet


class _LSEQuadCore(nn.Module):
    def __init__(self, dim, K, init_scale=0.3):
        super().__init__()
        self.L = nn.Parameter(init_scale * torch.randn(K, dim, dim) / dim**0.5)
        self.b = nn.Parameter(0.01 * torch.randn(K, dim))
        self.c = nn.Parameter(torch.zeros(K))


class LSEQuadNet(BaseNeuralNet):
    """Drop-in replacement for the MLPs: same computeValueFunction /
    computeValueFunctionDerivative / train interface (via BaseNeuralNet)."""

    def __init__(self, dim, K=8, m=0.1, tau=1.0):
        super().__init__(layers=None)
        self.dim, self.K, self.m, self.tau = dim, K, m, tau
        self.model = _LSEQuadCore(dim, K).to(self.device)

    def _buildLayers(self, layers):
        return None

    def _buildModel(self, layers):
        return None

    def computeValueFunction(self, x):
        core = self.model
        # (L_k^T x): einsum over input dim -> (n, K, d)
        xL = torch.einsum("nd,kde->nke", x, core.L)
        q = 0.5 * (xL**2).sum(-1) + x @ core.b.T + core.c          # (n, K)
        lse = torch.logsumexp(self.tau * q, dim=1, keepdim=True) / self.tau
        return 0.5 * self.m * (x**2).sum(dim=1, keepdim=True) + lse
