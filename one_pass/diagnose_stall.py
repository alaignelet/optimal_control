"""Diagnose why a convex network stalls during one-pass residual training.

Loads a saved model from results/<tag>/model.pt and reports, per layer:
- fraction of weights at exactly 0 (LazyClipped casualties — gradient is dead
  through them if the unit never reactivates),
- pre-activation statistics over the training domain (dead ReLU units:
  always-negative pre-activations),
- gradient norm of the residual loss at a fresh batch.

Usage: python diagnose_stall.py --tag full_relu_icnn --activation relu
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enums import ActivationFunctionEnum, InitFunctionEnum, PositivityFunctionEnum
from icnn import ConvexLinear
from neuralNet import ConvexNeuralNet
from pdes import LinearQuadraticRegulator2D


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--activation", default="relu")
    parser.add_argument("--positivity", default="icnn")
    parser.add_argument("--init", default="convex")
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--depth", type=int, default=3)
    args = parser.parse_args()

    layers = [2] + [args.hidden] * args.depth + [1]
    network = ConvexNeuralNet(
        layers,
        activation=ActivationFunctionEnum[args.activation.upper()],
        positivity=PositivityFunctionEnum[args.positivity.upper()],
        init=InitFunctionEnum[args.init.upper()],
    )
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "results", args.tag, "model.pt")
    network.load_state_dict(torch.load(path, map_location=network.device))
    model = network.model

    print(f"=== {args.tag} ===")

    # --- weight statistics per layer ---------------------------------------
    for i, layer in enumerate(model):
        if isinstance(layer, (ConvexLinear, torch.nn.Linear)):
            w = layer.weight.detach()
            kind = "ConvexLinear" if isinstance(layer, ConvexLinear) else "Linear"
            print(
                f"layer {i:2d} {kind:12s} shape {tuple(w.shape)}  "
                f"zeros: {(w == 0).float().mean():.1%}  neg: {(w < 0).float().mean():.1%}  "
                f"|w| mean: {w.abs().mean():.3e}  max: {w.abs().max():.3e}"
            )

    # --- dead units over the domain -----------------------------------------
    x = torch.rand(5000, 2, device=network.device) * 2 - 1  # LQR2D domain [-1,1]^2
    h = x
    for i, layer in enumerate(model):
        h_prev = h
        h = layer(h)
        if isinstance(layer, (ConvexLinear, torch.nn.Linear)) and h.shape[1] > 1:
            dead = (h <= 0).all(dim=0)  # unit never positive on the domain
            print(f"layer {i:2d} pre-act: dead units {dead.sum().item()}/{h.shape[1]}  "
                  f"mean {h.mean():.3e}  std {h.std():.3e}")

    # --- gradient norm of the residual loss ---------------------------------
    gamma = {"data": 0.0, "gradient": 0.0, "residual": 1.0}
    pde = LinearQuadraticRegulator2D(network=network, gamma=gamma, correctShift=True)
    xInt = torch.rand(500, 2, device=network.device, requires_grad=True) * 2 - 1
    gradInt = network.computeValueFunctionDerivative(xInt)
    _, _, residual = pde.lossFunction(xInt, gradInt, None, None)
    residual.backward()
    total, frozen = 0.0, []
    for name, p in model.named_parameters():
        g = 0.0 if p.grad is None else p.grad.norm().item()
        total += g**2
        if g < 1e-10:
            frozen.append(name)
    print(f"residual loss: {residual.item():.3e}  grad norm: {total**0.5:.3e}")
    if frozen:
        print(f"parameters with ~zero gradient: {frozen}")


if __name__ == "__main__":
    main()
