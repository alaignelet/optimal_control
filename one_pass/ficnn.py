"""FICNN: a fully-input-convex network fixed for residual training.

This starts from the Amos et al. (2017) input-convex architecture and applies
the three fixes our diagnosis (Section 3 of the paper) pointed to, all at the
architecture level --- NO penalty term:

  1. INPUT SKIP CONNECTIONS at every layer (the W^x x passthrough). These are
     what make ICNNs universal over convex functions; the 2024 skip-free
     variant dropped them. W^x is unconstrained; only the z->z weights W^z
     must be non-negative.
  2. SMOOTH POSITIVITY by reparameterization: the effective non-negative
     weight is softplus(rho) for an unconstrained parameter rho. No clipping,
     so no dead faces and gradients always flow (fixes the dying-unit death
     spiral).
  3. SMOOTH CONVEX activation (softplus), convex and non-decreasing with
     nowhere-zero derivative, so units cannot die the way ReLU units do.

Optionally a baked-in (m/2)||x||^2 term makes the whole network m-strongly
convex by construction (architectural, not a penalty).

Architecture (width h, L hidden layers, scalar output):
    z_1     = g( Wx_0 x + b_0 )                         Wx_0 free
    z_{l+1} = g( softplus(rho_l) z_l + Wx_l x + b_l )   l = 1..L-1
    V(x)    = softplus(rho_L) . z_L + wx_L x + b_L  + (m/2)||x||^2

Convexity: each z_l is convex in x (induction: non-negative combo of convex
z plus affine in x, through a convex non-decreasing g); V is a non-negative
combo of convex z_L plus affine plus a convex quadratic. Holds for ANY values
of the unconstrained parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralNet import BaseNeuralNet


def _pos(rho, kind):
    if kind == "softplus":
        return F.softplus(rho)
    if kind == "square":
        return rho ** 2
    if kind == "exp":
        return torch.exp(rho)
    if kind == "clip":
        return torch.clamp(rho, min=0.0)
    raise ValueError(kind)


class _FICNNCore(nn.Module):
    def __init__(self, dim, width, depth, use_skip=True, positivity="softplus",
                 init_scale=1.0):
        super().__init__()
        self.dim, self.width, self.depth = dim, width, depth
        self.use_skip = use_skip
        self.positivity = positivity
        g = init_scale

        # x -> layer skip weights (unconstrained); always present for layer 0,
        # present for every layer iff use_skip
        self.Wx = nn.ParameterList()
        self.bz = nn.ParameterList()
        self.rho = nn.ParameterList()  # z->z weights (made non-negative)

        # layer 0: x -> z1
        self.Wx.append(nn.Parameter(g * torch.randn(width, dim) / dim ** 0.5))
        self.bz.append(nn.Parameter(torch.zeros(width)))
        # hidden layers 1..L-1
        for _ in range(depth - 1):
            self.rho.append(nn.Parameter(_inv_softplus(g / width ** 0.5
                                                       * torch.rand(width, width),
                                                       positivity)))
            self.Wx.append(nn.Parameter(g * torch.randn(width, dim) / dim ** 0.5)
                           if use_skip else None)
            self.bz.append(nn.Parameter(torch.zeros(width)))
        # output layer
        self.rho.append(nn.Parameter(_inv_softplus(g / width ** 0.5
                                                   * torch.rand(1, width),
                                                   positivity)))
        self.Wx.append(nn.Parameter(g * torch.randn(1, dim) / dim ** 0.5)
                       if use_skip else None)
        self.bz.append(nn.Parameter(torch.zeros(1)))

    def forward(self, x):
        g = F.softplus
        z = g(F.linear(x, self.Wx[0], self.bz[0]))
        for l in range(self.depth - 1):
            Wz = _pos(self.rho[l], self.positivity)
            pre = F.linear(z, Wz, self.bz[l + 1])
            if self.use_skip:
                pre = pre + F.linear(x, self.Wx[l + 1])
            z = g(pre)
        Wz = _pos(self.rho[-1], self.positivity)
        out = F.linear(z, Wz, self.bz[-1])
        if self.use_skip:
            out = out + F.linear(x, self.Wx[-1])
        return out


def _inv_softplus(y, positivity):
    """Initialize rho so that _pos(rho) ~= y (>0)."""
    y = y.clamp(min=1e-4)
    if positivity == "softplus":
        return torch.log(torch.expm1(y).clamp(min=1e-6))
    if positivity == "square":
        return torch.sqrt(y)
    if positivity == "exp":
        return torch.log(y)
    return y  # clip


class FICNNet(BaseNeuralNet):
    """Fully-input-convex net; same interface as the other one_pass nets."""

    def __init__(self, dim, width=64, depth=3, m=0.0, use_skip=True,
                 positivity="softplus"):
        super().__init__(layers=None)
        self.dim, self.m = dim, m
        self.model = _FICNNCore(dim, width, depth, use_skip, positivity).to(self.device)

    def _buildLayers(self, layers):
        return None

    def _buildModel(self, layers):
        return None

    def computeValueFunction(self, x):
        out = self.model(x)
        if self.m > 0:
            out = out + 0.5 * self.m * (x ** 2).sum(dim=1, keepdim=True)
        return out
