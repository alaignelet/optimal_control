"""One-pass training of a convex value function for HJB problems.

The published paper (Borovykh, Kalise, Laignelet, Parpas 2022) used two passes:
supervised pre-training on SDRE data, then PDE-residual refinement. This
experiment trains in a SINGLE pass on the PDE residual alone, relying on the
architecture (ICNN: convex non-decreasing activations + non-negative weights
after the first layer) to enforce convexity of V instead of supervised data.

Usage (from the one_pass/ directory):
    python run_experiment.py --arch convex --activation relu --positivity icnn
    python run_experiment.py --arch linear                      # baseline
    python run_experiment.py --arch convex --activation gelu    # broken guarantee (ablation)

Results (metrics JSON + model weights) land in one_pass/results/<tag>/.
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
os.chdir(REPO_ROOT)  # pdes.py loads ground-truth CSVs via repo-root-relative paths

from enums import ActivationFunctionEnum, InitFunctionEnum, PositivityFunctionEnum
from neuralNet import ConvexNeuralNet, LinearNeuralNet
from pdes import LinearQuadraticRegulator2D, NonLinear2D

from convexity_check import convexity_report
from lseq import LSEQuadNet
from ficnn import FICNNet

# Only these activations preserve the ICNN convexity guarantee
# (convex AND non-decreasing). GELU/SiLU/Mish/Sigmoid/Tanh do NOT.
CONVEX_SAFE_ACTIVATIONS = {"RELU", "SOFTPLUS", "ELU", "CELU", "LEAKYRELU", "RELU6"}

PROBLEMS = {
    "lqr2d": LinearQuadraticRegulator2D,
    "nonlinear2d": NonLinear2D,
}


def min_hessian_eig_2d(x, grad):
    """Smallest eigenvalue of the 2x2 Hessian of V at each x (differentiable).

    grad = dV/dx at x, built with create_graph=True by the caller.
    """
    h0 = torch.autograd.grad(grad[:, 0].sum(), x, create_graph=True)[0]
    h1 = torch.autograd.grad(grad[:, 1].sum(), x, create_graph=True)[0]
    a, c = h0[:, 0], h1[:, 1]
    b = 0.5 * (h0[:, 1] + h1[:, 0])
    return 0.5 * (a + c) - torch.sqrt(0.25 * (a - c) ** 2 + b**2 + 1e-12)


def build_network(args, input_dim):
    layers = [input_dim] + [args.hidden] * args.depth + [1]
    if args.arch == "linear":
        return LinearNeuralNet(layers)
    if args.arch == "lseq":
        return LSEQuadNet(input_dim, K=args.lseq_K, m=args.lseq_m, tau=args.lseq_tau)
    if args.arch == "ficnn":
        return FICNNet(input_dim, width=args.hidden, depth=args.depth,
                       m=args.ficnn_m, use_skip=args.ficnn_skip,
                       positivity=args.ficnn_positivity)
    return ConvexNeuralNet(
        layers,
        activation=ActivationFunctionEnum[args.activation.upper()],
        positivity=PositivityFunctionEnum[args.positivity.upper()],
        init=InitFunctionEnum[args.init.upper()],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", default="lqr2d", choices=PROBLEMS.keys())
    parser.add_argument("--arch", default="convex",
                        choices=["convex", "linear", "lseq", "ficnn"])
    parser.add_argument("--ficnn-m", type=float, default=0.0,
                        help="baked-in strong-convexity floor for --arch ficnn")
    parser.add_argument("--ficnn-skip", action="store_true", default=True,
                        help="input skip connections (FICNN); default on")
    parser.add_argument("--ficnn-no-skip", dest="ficnn_skip", action="store_false",
                        help="disable skip connections (reproduces skip-free ICNN)")
    parser.add_argument("--ficnn-positivity", default="softplus",
                        choices=["softplus", "square", "exp", "clip"])
    parser.add_argument("--lseq-K", type=int, default=8,
                        help="number of quadratic components for --arch lseq")
    parser.add_argument("--lseq-m", type=float, default=0.1,
                        help="built-in strong-convexity floor for --arch lseq")
    parser.add_argument("--lseq-tau", type=float, default=1.0,
                        help="log-sum-exp temperature for --arch lseq")
    parser.add_argument("--activation", default="relu")
    parser.add_argument("--positivity", default="icnn", choices=["convex", "icnn", "fc"])
    parser.add_argument("--init", default="convex", choices=["traditional", "convex"])
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--gamma-data", type=float, default=0.0)
    parser.add_argument("--gamma-gradient", type=float, default=0.0)
    parser.add_argument("--gamma-residual", type=float, default=1.0)
    parser.add_argument("--interior-points", type=int, default=500)
    parser.add_argument("--data-points", type=int, default=50)
    parser.add_argument("--lrs", type=float, nargs="+", default=[1e-2, 1e-3, 1e-4])
    parser.add_argument("--iterations", type=int, nargs="+", default=[2000, 4000, 4000])
    parser.add_argument("--sampling", default="random", choices=["random", "grid"])
    parser.add_argument("--convexity-penalty", type=float, default=0.0,
                        help="weight of the soft hinge penalty on the smallest "
                             "Hessian eigenvalue at interior points (0 = off)")
    parser.add_argument("--margin", type=float, default=0.0,
                        help="require lambda_min >= margin in the penalty")
    parser.add_argument("--equilibrium-anchor", type=float, default=0.0,
                        help="weight of |V(0)|^2 + |gradV(0)|^2 (free anchor: "
                             "the origin is the equilibrium minimum; no data needed)")
    parser.add_argument("--resample", action="store_true",
                        help="draw fresh interior points every step for the "
                             "residual (and penalty) instead of a fixed set")
    parser.add_argument("--aug-lagrangian", type=float, default=0.0,
                        help="rho for an augmented-Lagrangian treatment of "
                             "lambda_min(x) >= margin on a fixed grid of "
                             "constraint points (0 = off)")
    parser.add_argument("--al-update-every", type=int, default=100,
                        help="multiplier update interval (steps)")
    parser.add_argument("--warm-shape", type=int, default=0,
                        help="pre-fit the network to the generic convex bowl "
                             "0.5||x||^2 for N steps before training (data-free)")
    parser.add_argument("--eps-continuation", type=int, default=0,
                        help="anneal the nonlinearity eps from 0 (=LQR) to 1 "
                             "over N steps — homotopy continuation from the "
                             "solvable linear problem (nonlinear2d only)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default=None, help="results subfolder name")
    parser.add_argument("--wandb", action="store_true", help="log to W&B (off by default)")
    parser.add_argument("--hessian-grid", type=int, default=20)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tag = args.tag or (
        f"{args.problem}_{args.arch}"
        + ("" if args.arch == "linear" else f"_{args.activation}_{args.positivity}_{args.init}")
        + f"_seed{args.seed}"
    )
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", tag)
    os.makedirs(outdir, exist_ok=True)

    wandb.init(
        mode="online" if args.wandb else "disabled",
        project="one_pass_convex_hjb",
        name=tag,
        config=vars(args),
    )

    if args.arch == "convex" and args.activation.upper() not in CONVEX_SAFE_ACTIVATIONS:
        print(
            f"WARNING: {args.activation} is not convex non-decreasing — "
            "the network output is NOT guaranteed convex (ablation mode)."
        )

    gamma = {
        "data": args.gamma_data,
        "gradient": args.gamma_gradient,
        "residual": args.gamma_residual,
    }
    one_pass = args.gamma_data == 0.0 and args.gamma_gradient == 0.0

    # The residual-only loss determines V up to a constant; correct the shift
    # before comparing against the ground truth.
    problem_cls = PROBLEMS[args.problem]
    input_dim = 2  # both lqr2d and nonlinear2d have 2D state
    network = build_network(args, input_dim)
    pde = problem_cls(network=network, gamma=gamma, correctShift=one_pass)

    lows = torch.tensor([lo for lo, _ in pde.domain], dtype=torch.float32,
                        device=network.device)
    spans = torch.tensor([hi - lo for lo, hi in pde.domain], dtype=torch.float32,
                         device=network.device)

    if args.warm_shape > 0:
        # Data-free convex warm shaping: pre-fit V to the generic strongly
        # convex bowl 0.5||x||^2 (values + gradients). Initialization, not
        # supervision — no problem-specific data involved.
        opt = torch.optim.Adam(network.model.parameters(), lr=1e-3)
        for step in range(args.warm_shape):
            x = (torch.rand(args.interior_points, input_dim, device=network.device)
                 * spans + lows).requires_grad_(True)
            v = network.computeValueFunction(x)
            g = network.computeValueFunctionDerivative(x)
            target = 0.5 * (x**2).sum(dim=1, keepdim=True)
            shape_loss = ((v - target) ** 2).mean() + ((g - x) ** 2).mean()
            opt.zero_grad()
            shape_loss.backward()
            opt.step()
        print(f"warm shaping done ({args.warm_shape} steps, final loss {shape_loss.item():.2e})")

    if args.equilibrium_anchor > 0.0:
        anchor_base_loss = pde.lossFunction

        def loss_with_anchor(xInt, gradInt, yData, gradData):
            lossData, lossGrad, residual = anchor_base_loss(xInt, gradInt, yData, gradData)
            x0 = torch.zeros(1, input_dim, device=network.device, requires_grad=True)
            v0 = network.computeValueFunction(x0)
            g0 = network.computeValueFunctionDerivative(x0)
            anchor = (v0**2).sum() + (g0**2).sum()
            return lossData, lossGrad, residual + args.equilibrium_anchor * anchor

        pde.lossFunction = loss_with_anchor

    if args.convexity_penalty > 0.0:
        # Soft convexity: hinge on the smallest Hessian eigenvalue at the
        # interior points, added to the residual term so the existing
        # 3-term training loop needs no changes. 2D-only (analytic 2x2 eig).
        base_loss = pde.lossFunction

        def loss_with_penalty(xInt, gradInt, yData, gradData):
            lossData, lossGrad, residual = base_loss(xInt, gradInt, yData, gradData)
            lam_min = min_hessian_eig_2d(xInt, gradInt)
            hinge = torch.relu(args.margin - lam_min).mean()
            return lossData, lossGrad, residual + args.convexity_penalty * hinge

        pde.lossFunction = loss_with_penalty

    if args.aug_lagrangian > 0.0:
        # Augmented Lagrangian for lambda_min(x) >= margin on a FIXED grid of
        # constraint points with persistent per-point multipliers mu_i.
        # Standard inequality AL: (rho/2)[max(0, mu/rho + c)^2 - (mu/rho)^2],
        # c_i = margin - lambda_min(x_i) <= 0; mu_i <- max(0, mu_i + rho c_i)
        # every --al-update-every steps.
        rho = args.aug_lagrangian
        axes = [torch.linspace(lo, hi, 20, device=network.device)
                for lo, hi in pde.domain]
        cpoints = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1) \
                       .reshape(-1, input_dim)
        mu = torch.zeros(cpoints.shape[0], device=network.device)
        al_state = {"step": 0}
        al_base_loss = pde.lossFunction

        def loss_auglag(xInt, gradInt, yData, gradData):
            lossData, lossGrad, residual = al_base_loss(xInt, gradInt, yData, gradData)
            x = cpoints.clone().requires_grad_(True)
            g = network.computeValueFunctionDerivative(x)
            c = args.margin - min_hessian_eig_2d(x, g)
            al = (rho / 2) * (torch.clamp(mu / rho + c, min=0) ** 2
                              - (mu / rho) ** 2).mean()
            al_state["step"] += 1
            if al_state["step"] % args.al_update_every == 0:
                with torch.no_grad():
                    mu.copy_(torch.clamp(mu + rho * c.detach(), min=0))
            return lossData, lossGrad, residual + al

        pde.lossFunction = loss_auglag

    if args.resample:
        # Outermost wrapper: draw fresh interior points every step so the
        # residual (and the penalty wrapped inside) target the EXPECTED loss
        # over the domain, not a fixed 500-point sample — standard PINN
        # practice, prevents collocation overfitting. The trainer's gradInt
        # (computed on its fixed xInt) is ignored.
        resample_base_loss = pde.lossFunction
        d = len(pde.domain)

        def loss_resampled(xInt, gradInt, yData, gradData):
            x = (torch.rand(args.interior_points, d, device=network.device)
                 * spans + lows).requires_grad_(True)
            g = network.computeValueFunctionDerivative(x)
            return resample_base_loss(x, g, yData, gradData)

        pde.lossFunction = loss_resampled

    if args.eps_continuation > 0:
        # Homotopy continuation: at eps=0 the problem is LQR, whose convex
        # solution the one-pass stack finds exactly; annealing eps -> 1 tracks
        # the continuous solution path and avoids the spurious basin.
        # Outermost wrapper: sets eps before anything inside computes.
        assert hasattr(pde, "eps"), "--eps-continuation needs a problem with eps"
        eps_final = pde.eps.detach().clone()
        cont_state = {"step": 0}
        cont_base_loss = pde.lossFunction

        def loss_continuation(xInt, gradInt, yData, gradData):
            frac = min(1.0, cont_state["step"] / args.eps_continuation)
            with torch.no_grad():
                pde.eps.copy_(frac * eps_final)
            cont_state["step"] += 1
            return cont_base_loss(xInt, gradInt, yData, gradData)

        pde.lossFunction = loss_continuation

    print(f"Training [{tag}] gamma={gamma} (one-pass={one_pass}, "
          f"convexity_penalty={args.convexity_penalty}, resample={args.resample})")
    history = pde.train(
        interiorPointCount=args.interior_points,
        dataPointCount=args.data_points,
        lrs=args.lrs,
        iterations=args.iterations,
        sampling=args.sampling,
    )
    history.to_csv(os.path.join(outdir, "training_history.csv"), index=False)

    # --- Out-of-sample accuracy against the ground truth -------------------
    x_eval = pde.getEvaluationPoints().to(network.device)
    y_eval = network.computeValueFunction(x_eval)
    eval_mse = pde.evaluationFunction(y_eval).item()

    # --- Convexity of the LEARNED network (exact autograd Hessian) ---------
    report = convexity_report(network, pde.domain, points_per_dim=args.hessian_grid)

    metrics = {
        "tag": tag,
        "config": vars(args),
        "final_loss": float(history["loss"].iloc[-1]),
        "eval_mse": eval_mse,
        "hessian_global_min_eigenvalue": report["global_min_eigenvalue"],
        "hessian_violation_fraction": report["violation_fraction"],
        "hessian_n_grid_points": report["n_points"],
    }
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    if len(report["violation_points"]):
        np.savetxt(
            os.path.join(outdir, "non_convex_points.csv"),
            np.column_stack([report["violation_points"], report["violation_min_eigs"]]),
            delimiter=",",
            header=",".join(f"x{i}" for i in range(report["violation_points"].shape[1]))
            + ",min_eig",
            comments="",
        )
    torch.save(network.state_dict(), os.path.join(outdir, "model.pt"))

    print(json.dumps(metrics, indent=2))
    convex = report["violation_fraction"] == 0.0
    print(f"\n=> Network is {'CONVEX' if convex else 'NOT convex'} on the test grid "
          f"(min eig {report['global_min_eigenvalue']:.3e}, "
          f"violations {report['violation_fraction']:.1%}). "
          f"Eval MSE vs ground truth: {eval_mse:.3e}")
    wandb.finish()


if __name__ == "__main__":
    main()
