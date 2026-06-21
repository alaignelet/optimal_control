"""Figures for the 10D run: a 2D slice V(x0, x1, 0, ..., 0) learned-vs-truth
surface pair, and a predicted-vs-true scatter over random evaluation points.

Usage: python plot_nd_slice.py nd10_full [nd10_unconstrained ...]
"""

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
from neuralNet import ConvexNeuralNet
from lseq import LSEQuadNet
from ficnn import FICNNet
from run_nd import ground_truth

wandb.init(mode="disabled")
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
N = 80

for tag in sys.argv[1:]:
    folder = os.path.join(RESULTS, tag)
    metrics = json.load(open(os.path.join(folder, "metrics.json")))
    cfg = metrics["config"]
    d, eps = cfg["dim"], cfg["eps"]
    if cfg.get("arch") == "lseq":
        net = LSEQuadNet(d, K=cfg["lseq_K"], m=cfg["lseq_m"], tau=cfg["lseq_tau"])
    elif cfg.get("arch") == "ficnn":
        net = FICNNet(d, width=cfg["hidden"], depth=cfg["depth"], m=cfg["ficnn_m"],
                      use_skip=cfg["ficnn_skip"], positivity=cfg["ficnn_positivity"])
    else:
        net = ConvexNeuralNet(
            [d] + [cfg["hidden"]] * cfg["depth"] + [1],
            activation=ActivationFunctionEnum.GELU,
            positivity=PositivityFunctionEnum.FC,
            init=InitFunctionEnum.TRADITIONAL,
        )
    net.load_state_dict(torch.load(os.path.join(folder, "model.pt"),
                                   map_location=net.device))

    # --- slice surface ------------------------------------------------------
    g0, g1 = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N))
    pts = np.zeros((N * N, d))
    pts[:, 0] = g0.ravel()
    pts[:, 1] = g1.ravel()
    with torch.no_grad():
        vp = net.computeValueFunction(
            torch.tensor(pts, dtype=torch.float32, device=net.device)
        ).cpu().numpy().reshape(N, N)
    vt = ground_truth(pts, eps).reshape(N, N)
    vp = vp - vp.min() + vt.min()  # shift correction

    fig = plt.figure(figsize=(11, 5))
    zmin, zmax = min(vp.min(), vt.min()), max(vp.max(), vt.max())
    for k, (Z, title) in enumerate([(vp, "learned V (shift-corrected)"),
                                    (vt, "ground truth")]):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        ax.plot_surface(g0, g1, Z, cmap="viridis", linewidth=0, antialiased=True)
        ax.set_zlim(zmin, zmax)
        ax.set_xlabel("$x_0$")
        ax.set_ylabel("$x_1$")
        ax.set_title(title, fontsize=10)
    fig.suptitle(f"{tag} — slice $V(x_0, x_1, 0, \\dots, 0)$ — "
                 f"eval MSE {metrics['eval_mse']:.2e}, "
                 f"min eig {metrics['hessian_min_eig']:+.2e}", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(folder, "slice.png"), dpi=150)
    plt.close(fig)

    # --- scatter ------------------------------------------------------------
    rng = np.random.default_rng(1)
    xs = rng.uniform(-1, 1, size=(3000, d))
    yt = ground_truth(xs, eps).ravel()
    with torch.no_grad():
        yp = net.computeValueFunction(
            torch.tensor(xs, dtype=torch.float32, device=net.device)
        ).cpu().numpy().ravel()
    yp = yp - yp.min() + yt.min()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(yt, yp, s=2, alpha=0.3)
    lims = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlabel("true $V$")
    ax.set_ylabel("learned $V$")
    ax.set_title(f"{tag}: 3000 random points in $[-1,1]^{{{d}}}$")
    fig.tight_layout()
    fig.savefig(os.path.join(folder, "scatter.png"), dpi=150)
    plt.close(fig)
    print(f"{tag}: slice.png + scatter.png")
