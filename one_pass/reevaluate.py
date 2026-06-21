"""Re-evaluate saved NonLinear2D models against the (fixed) ground truth.

The original _loadTrueSolution flattened the CSV without transposing, so all
NonLinear2D eval MSEs computed before the fix are wrong. This reloads each
saved model and recomputes eval MSE with the corrected truth.

Usage: python reevaluate.py r5_nl_unconstrained r6_nl_pen100_m01 ...
"""

import json
import os
import sys

import torch
import wandb

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from enums import ActivationFunctionEnum, InitFunctionEnum, PositivityFunctionEnum
from neuralNet import ConvexNeuralNet
from pdes import NonLinear2D

wandb.init(mode="disabled")

results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

for tag in sys.argv[1:]:
    cfg = json.load(open(os.path.join(results_dir, tag, "metrics.json")))["config"]
    layers = [2] + [cfg["hidden"]] * cfg["depth"] + [1]
    network = ConvexNeuralNet(
        layers,
        activation=ActivationFunctionEnum[cfg["activation"].upper()],
        positivity=PositivityFunctionEnum[cfg["positivity"].upper()],
        init=InitFunctionEnum[cfg["init"].upper()],
    )
    network.load_state_dict(
        torch.load(os.path.join(results_dir, tag, "model.pt"), map_location=network.device)
    )
    one_pass = cfg["gamma_data"] == 0.0 and cfg["gamma_gradient"] == 0.0
    pde = NonLinear2D(network=network, gamma={"data": 0, "gradient": 0, "residual": 1},
                      correctShift=one_pass)
    x_eval = pde.getEvaluationPoints().to(network.device)
    y_eval = network.computeValueFunction(x_eval)
    pde.yEvaluationTrue = pde.groundTruthSolution(x_eval.detach())
    mse = pde.evaluationFunction(y_eval).item()
    print(f"{tag:30s} corrected eval MSE: {mse:.4e}  (was {json.load(open(os.path.join(results_dir, tag, 'metrics.json')))['eval_mse']:.4e})")
