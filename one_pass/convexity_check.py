"""Exact convexity verification for a trained value-function network.

Computes the autograd Hessian of V(x) on a grid over the domain and reports
the minimum eigenvalue at each point. This checks the NETWORK's convexity
(unlike the script.ipynb study, which checked the convexity of the
SDRE-simulated true value function).
"""

import numpy as np
import torch


def hessian_eigenvalues_on_grid(network, domain, points_per_dim=20):
    """Return (points, min_eigenvalues) for V's Hessian over a grid.

    Args:
        network: a BaseNeuralNet subclass with computeValueFunction.
        domain: list of (low, high) per input dimension.
        points_per_dim: grid resolution per dimension.
    """
    device = network.device
    axes = [np.linspace(low, high, points_per_dim) for (low, high) in domain]
    grid = np.stack(np.meshgrid(*axes), axis=-1).reshape(-1, len(domain))

    def scalar_value(x_single):
        return network.computeValueFunction(x_single.unsqueeze(0)).squeeze()

    min_eigs = np.empty(grid.shape[0])
    for i, point in enumerate(grid):
        x = torch.tensor(point, dtype=torch.float32, device=device)
        hess = torch.autograd.functional.hessian(scalar_value, x)
        # Symmetrise to kill numerical asymmetry before eigvalsh
        hess = 0.5 * (hess + hess.T)
        min_eigs[i] = torch.linalg.eigvalsh(hess).min().item()

    return grid, min_eigs


def convexity_report(network, domain, points_per_dim=20, tol=-1e-5):
    """Summarise convexity of the network over the domain.

    Returns a dict with the global minimum eigenvalue, the fraction of grid
    points violating positive semi-definiteness (below tol), and the points
    where violations occur.
    """
    grid, min_eigs = hessian_eigenvalues_on_grid(network, domain, points_per_dim)
    violations = min_eigs < tol
    return {
        "n_points": len(grid),
        "global_min_eigenvalue": float(min_eigs.min()),
        "violation_fraction": float(violations.mean()),
        "violation_points": grid[violations],
        "violation_min_eigs": min_eigs[violations],
    }
