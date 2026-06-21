"""3D surface plots of every trained value function vs the ground truth.

For each results/<tag>/ containing model.pt + metrics.json, renders a
two-panel figure (learned V left, ground truth right, shared z-scale,
notebook-style viridis plot_surface) and saves it as results/<tag>/surface.png.
Also writes standalone ground-truth surfaces to results/ground_truth_<problem>.png.

Usage:
    python plot_surfaces.py            # all result folders (skips smoke_*)
    python plot_surfaces.py r11_nl_continuation_pen ...   # specific tags
"""

import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from enums import ActivationFunctionEnum, InitFunctionEnum, PositivityFunctionEnum
from neuralNet import ConvexNeuralNet, LinearNeuralNet
from pdes import LinearQuadraticRegulator2D, NonLinear2D
from lseq import LSEQuadNet
from ficnn import FICNNet

wandb.init(mode="disabled")

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
N = 100  # grid resolution per axis


def build_network_from_config(cfg):
    if cfg["arch"] == "lseq":
        return LSEQuadNet(2, K=cfg["lseq_K"], m=cfg["lseq_m"], tau=cfg["lseq_tau"])
    if cfg["arch"] == "ficnn":
        return FICNNet(2, width=cfg["hidden"], depth=cfg["depth"],
                       m=cfg["ficnn_m"], use_skip=cfg["ficnn_skip"],
                       positivity=cfg["ficnn_positivity"])
    layers = [2] + [cfg["hidden"]] * cfg["depth"] + [1]
    if cfg["arch"] == "linear":
        return LinearNeuralNet(layers)
    return ConvexNeuralNet(
        layers,
        activation=ActivationFunctionEnum[cfg["activation"].upper()],
        positivity=PositivityFunctionEnum[cfg["positivity"].upper()],
        init=InitFunctionEnum[cfg["init"].upper()],
    )


def ground_truth_grid(problem):
    """Return (X0, X1, Vtrue) on the NxN grid, matching sampleGrid ordering."""
    # sampleGrid uses np.meshgrid default 'xy' indexing then reshape(-1, 2)
    g = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N))
    pts = np.stack(g, axis=-1).reshape(-1, 2)
    if problem == "lqr2d":
        p = (1 / 5) * (1 + np.sqrt(6))
        v = 0.5 * p * (pts[:, 0] ** 2 + pts[:, 1] ** 2)
    else:
        # use the (fixed) loader so orientation matches the eval pipeline
        dummy = ConvexNeuralNet(
            [2, 4, 1],
            activation=ActivationFunctionEnum.GELU,
            positivity=PositivityFunctionEnum.FC,
            init=InitFunctionEnum.TRADITIONAL,
        )
        pde = NonLinear2D(network=dummy, gamma={"data": 0, "gradient": 0, "residual": 1})
        v = pde.true_solution.cpu().numpy().reshape(-1)
    return g[0], g[1], v.reshape(N, N)


def surface_pair(ax_pred, ax_true, X0, X1, Vp, Vt, title_pred, title_true):
    zmin = min(Vp.min(), Vt.min())
    zmax = max(Vp.max(), Vt.max())
    for ax, Z, title in [(ax_pred, Vp, title_pred), (ax_true, Vt, title_true)]:
        ax.plot_surface(X0, X1, Z, cmap="viridis", linewidth=0, antialiased=True)
        ax.set_zlim(zmin, zmax)
        ax.set_xlabel("$x_0$")
        ax.set_ylabel("$x_1$")
        ax.set_title(title, fontsize=10)


def main():
    tags = sys.argv[1:] or sorted(
        os.path.basename(d)
        for d in glob.glob(os.path.join(RESULTS, "*"))
        if os.path.isdir(d)
        and not os.path.basename(d).startswith("smoke")
        and os.path.exists(os.path.join(d, "model.pt"))
    )

    truths = {p: ground_truth_grid(p) for p in ("lqr2d", "nonlinear2d")}

    # standalone ground-truth figures
    for problem, (X0, X1, Vt) in truths.items():
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X0, X1, Vt, cmap="viridis", linewidth=0, antialiased=True)
        ax.set_xlabel("$x_0$")
        ax.set_ylabel("$x_1$")
        ax.set_title(f"Ground truth V — {problem}")
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS, f"ground_truth_{problem}.png"), dpi=150)
        plt.close(fig)
        print(f"ground_truth_{problem}.png")

    for tag in tags:
        folder = os.path.join(RESULTS, tag)
        try:
            metrics = json.load(open(os.path.join(folder, "metrics.json")))
        except FileNotFoundError:
            print(f"{tag}: no metrics.json, skipped")
            continue
        cfg = metrics["config"]
        problem = cfg.get("problem", "lqr2d")
        X0, X1, Vt = truths[problem]

        network = build_network_from_config(cfg)
        network.load_state_dict(
            torch.load(os.path.join(folder, "model.pt"), map_location=network.device)
        )
        pts = np.stack(np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N)),
                       axis=-1).reshape(-1, 2)
        x = torch.tensor(pts, dtype=torch.float32, device=network.device)
        with torch.no_grad():
            v = network.computeValueFunction(x).cpu().numpy().reshape(N, N)

        one_pass = cfg["gamma_data"] == 0.0 and cfg["gamma_gradient"] == 0.0
        if one_pass:
            v = v - v.min()  # same shift correction as the evaluation

        fig = plt.figure(figsize=(11, 5))
        ax1 = fig.add_subplot(121, projection="3d")
        ax2 = fig.add_subplot(122, projection="3d")
        surface_pair(
            ax1, ax2, X0, X1, v, Vt,
            f"learned V ({'shift-corrected' if one_pass else 'raw'})",
            "ground truth",
        )
        fig.suptitle(f"{tag} — eval MSE {metrics['eval_mse']:.2e}, "
                     f"min eig {metrics['hessian_global_min_eigenvalue']:+.2e}",
                     fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(folder, "surface.png"), dpi=150)
        plt.close(fig)
        print(f"{tag}/surface.png")


if __name__ == "__main__":
    main()
