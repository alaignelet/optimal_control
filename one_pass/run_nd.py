"""One-pass solver for a 10D nonlinear HJB with provably convex value function.

Problem (decoupled cubic, coupled only through the network's ignorance):
    dy_i = (eps * y_i^3 + u_i) dt,   J = int 1/2 ||y||^2 + beta ||u||^2,
    beta = 1/2, domain [-1,1]^d.

Scalar HJB per coordinate: -1/(4b) v'^2 + eps s^3 v' + s^2/2 = 0, stabilizing
branch  v'(s) = 2b*eps*s^3 + s*sqrt(4 b^2 eps^2 s^4 + 2b),  which gives
v''(s) = 6b*eps*s^2 + sqrt(.) + 8 b^2 eps^2 s^4 / sqrt(.) > 0:
the value function V(x) = sum_i v(x_i) is STRICTLY convex with curvature
floor sqrt(2b) = 1 on the whole space. Ground truth by 1D quadrature.

Method (the one-pass recipe, scaled to d dimensions):
    resampled residual
  + directional-curvature hinge  E_v[ relu(margin - v^T H v) ],  v ~ unit sphere
  + equilibrium anchor |V(0)|^2 + |grad V(0)|^2
  + homotopy continuation in eps (eps=0 is decoupled LQR).

Final verification: full autograd Hessian eigenvalues at random points,
eval MSE against the quadrature ground truth.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import wandb

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from enums import ActivationFunctionEnum, InitFunctionEnum, PositivityFunctionEnum
from neuralNet import ConvexNeuralNet

sys.path.insert(0, os.path.join(REPO_ROOT, "one_pass"))
from lseq import LSEQuadNet
from ficnn import FICNNet

BETA = 0.5


def v_prime(s, eps):
    return (2 * BETA * eps * s**3
            + s * np.sqrt(4 * BETA**2 * eps**2 * s**4 + 2 * BETA))


def v_scalar(s_grid, eps):
    """v(s) on a grid by cumulative trapezoid integration of v' from 0."""
    out = np.empty_like(s_grid)
    for k, s in enumerate(s_grid):
        ts = np.linspace(0.0, s, 200)
        out[k] = np.trapezoid(v_prime(ts, eps), ts)
    return out


def ground_truth(x, eps):
    """V(x) = sum_i v(x_i) via an interpolation table (x: numpy (n,d))."""
    grid = np.linspace(-1.5, 1.5, 2001)
    table = v_scalar(grid, eps)
    return np.interp(x, grid, table).sum(axis=1, keepdims=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="mlp", choices=["mlp", "lseq", "ficnn"])
    parser.add_argument("--lseq-K", type=int, default=8)
    parser.add_argument("--lseq-m", type=float, default=0.1)
    parser.add_argument("--lseq-tau", type=float, default=1.0)
    parser.add_argument("--ficnn-m", type=float, default=0.0)
    parser.add_argument("--ficnn-no-skip", dest="ficnn_skip", action="store_false",
                        default=True)
    parser.add_argument("--ficnn-positivity", default="softplus")
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--eps", type=float, default=1.0)
    parser.add_argument("--interior-points", type=int, default=1000)
    parser.add_argument("--dirs-per-point", type=int, default=4)
    parser.add_argument("--penalty", type=float, default=10.0)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--anchor", type=float, default=10.0)
    parser.add_argument("--eps-continuation", type=int, default=12000)
    parser.add_argument("--lrs", type=float, nargs="+", default=[1e-2, 1e-3, 1e-4, 1e-5])
    parser.add_argument("--iterations", type=int, nargs="+", default=[4000, 8000, 8000, 8000])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    tag = args.tag or f"nd{args.dim}_pen{args.penalty}_m{args.margin}_seed{args.seed}"
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", tag)
    os.makedirs(outdir, exist_ok=True)
    wandb.init(mode="disabled")

    d = args.dim
    if args.arch == "lseq":
        net = LSEQuadNet(d, K=args.lseq_K, m=args.lseq_m, tau=args.lseq_tau)
    elif args.arch == "ficnn":
        net = FICNNet(d, width=args.hidden, depth=args.depth, m=args.ficnn_m,
                      use_skip=args.ficnn_skip, positivity=args.ficnn_positivity)
    else:
        net = ConvexNeuralNet(
            [d] + [args.hidden] * args.depth + [1],
            activation=ActivationFunctionEnum.GELU,
            positivity=PositivityFunctionEnum.FC,
            init=InitFunctionEnum.TRADITIONAL,
        )
    device = net.device
    eps_final = args.eps

    def residual(x, grad, eps_now):
        fx = (grad * (eps_now * x**3)).sum(dim=1, keepdim=True)
        gx = -1.0 / (4 * BETA) * (grad**2).sum(dim=1, keepdim=True)
        lx = 0.5 * (x**2).sum(dim=1, keepdim=True)
        return fx + gx + lx

    def directional_curvature(x, grad):
        curvs = []
        for _ in range(args.dirs_per_point):
            vv = torch.randn(x.shape[0], d, device=device)
            vv = vv / vv.norm(dim=1, keepdim=True)
            gv = (grad * vv).sum()
            hv = torch.autograd.grad(gv, x, create_graph=True)[0]
            curvs.append((hv * vv).sum(dim=1))
        return torch.stack(curvs, dim=1)  # (n, k) of v^T H v

    # ---- training loop (self-contained; mirrors the 2D stack) -------------
    step_total = 0
    optimizer = None
    for lr, iters in zip(args.lrs, args.iterations):
        optimizer = torch.optim.Adam(net.model.parameters(), lr=lr)
        for it in range(iters):
            frac = min(1.0, step_total / args.eps_continuation) if args.eps_continuation else 1.0
            eps_now = frac * eps_final

            x = (torch.rand(args.interior_points, d, device=device) * 2 - 1).requires_grad_(True)
            grad = net.computeValueFunctionDerivative(x)

            loss_res = (residual(x, grad, eps_now) ** 2).mean()
            if args.penalty > 0:
                curv = directional_curvature(x, grad)
                loss_cvx = torch.relu(args.margin - curv).mean()
            else:
                loss_cvx = torch.zeros((), device=device)
            x0 = torch.zeros(1, d, device=device, requires_grad=True)
            v0 = net.computeValueFunction(x0)
            g0 = net.computeValueFunctionDerivative(x0)
            loss_anchor = (v0**2).sum() + (g0**2).sum()

            loss = loss_res + args.penalty * loss_cvx + args.anchor * loss_anchor
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step_total % 1000 == 0:
                print(f"{step_total}  lr {lr:.0e}  eps {eps_now:.2f}  "
                      f"res {loss_res.item():.2e}  cvx {loss_cvx.item():.2e}  "
                      f"anchor {loss_anchor.item():.2e}", flush=True)
            step_total += 1

    # ---- evaluation --------------------------------------------------------
    x_eval = np.random.uniform(-1, 1, size=(5000, d))
    y_true = ground_truth(x_eval, eps_final)
    with torch.no_grad():
        y_pred = net.computeValueFunction(
            torch.tensor(x_eval, dtype=torch.float32, device=device)).cpu().numpy()
    y_pred = y_pred - y_pred.min() + y_true.min()  # shift correction
    eval_mse = float(((y_pred - y_true) ** 2).mean())

    # full Hessian eigenvalue check at random points
    def scalar_v(xs):
        return net.computeValueFunction(xs.unsqueeze(0)).squeeze()

    n_check = 300
    min_eigs = np.empty(n_check)
    pts = torch.rand(n_check, d, device=device) * 2 - 1
    for i in range(n_check):
        Hm = torch.autograd.functional.hessian(scalar_v, pts[i])
        Hm = 0.5 * (Hm + Hm.T)
        min_eigs[i] = torch.linalg.eigvalsh(Hm).min().item()

    metrics = {
        "tag": tag, "config": vars(args),
        "final_residual": float(loss_res.item()),
        "eval_mse": eval_mse,
        "hessian_min_eig": float(min_eigs.min()),
        "hessian_violation_fraction": float((min_eigs < -1e-5).mean()),
        "n_hessian_points": n_check,
    }
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    torch.save(net.state_dict(), os.path.join(outdir, "model.pt"))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
