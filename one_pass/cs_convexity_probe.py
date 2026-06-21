"""Is the Cucker-Smale value function convex in the domain INTERIOR?

The Oct 2024 study found non-convex points of the SDRE-rollout value function
only near the domain boundary (eps=1e-4 one-sided stencil) — possibly
artefacts. This probe redoes the analysis properly: random points in the
interior [-2,2]^4 (domain is [-3,3]^4 for dim=2 particles), central
finite differences with h=0.05.

Caveat: the SDRE rollout V is itself an approximation of the true value
function; this is evidence, not proof.
"""

import numpy as np
import scipy.linalg

DIM = 2          # particles -> state is (y1, y2, v1, v2) in R^4
BETA = 0.5
T, DT = 10.0, 0.1
H = 0.05         # FD stencil
N_POINTS = 150
RNG = np.random.default_rng(0)

Q = (1.0 / DIM) * np.eye(2 * DIM)
R = (1.0 / DIM) * np.eye(DIM)
B = np.vstack([np.zeros((DIM, DIM)), np.eye(DIM)])


def build_A(y):
    """State-dependent A for frozen positions y (Cucker-Smale, pdes.py)."""
    a = lambda yi, yj: 1.0 / (1.0 + (yi - yj) ** 2)
    A_small = np.zeros((DIM, DIM))
    for i in range(DIM):
        for j in range(DIM):
            if i == j:
                A_small[i, j] = -(1.0 / DIM) * sum(a(y[i], y[k]) for k in range(DIM))
            else:
                A_small[i, j] = (1.0 / DIM) * a(y[i], y[j])
    A = np.zeros((2 * DIM, 2 * DIM))
    A[:DIM, DIM:] = np.eye(DIM)
    A[DIM:, DIM:] = A_small
    return A


def value(x0):
    """SDRE-feedback rollout cost from x0 (same scheme as script.ipynb)."""
    x = x0.copy()
    xs = np.zeros((int(T / DT), 2 * DIM))
    us = np.zeros((int(T / DT), DIM))
    for i in range(int(T / DT)):
        A = build_A(x[:DIM])
        P = scipy.linalg.solve_continuous_are(A, B, Q, R)
        u = -1.0 / (2 * BETA) * np.linalg.inv(R) @ B.T @ P @ x
        xs[i], us[i] = x, u
        x = x + (A @ x + B @ u) * DT
    cost = np.trapezoid((xs @ Q * xs).sum(axis=1), dx=DT) \
         + np.trapezoid((us @ R * us).sum(axis=1), dx=DT)
    return cost


def hessian_fd(x0):
    n = len(x0)
    Hm = np.zeros((n, n))
    f0 = value(x0)
    for i in range(n):
        ei = np.zeros(n); ei[i] = H
        Hm[i, i] = (value(x0 + ei) - 2 * f0 + value(x0 - ei)) / H**2
        for j in range(i + 1, n):
            ej = np.zeros(n); ej[j] = H
            Hm[i, j] = Hm[j, i] = (
                value(x0 + ei + ej) - value(x0 + ei - ej)
                - value(x0 - ei + ej) + value(x0 - ei - ej)
            ) / (4 * H**2)
    return Hm


def main():
    pts = RNG.uniform(-2, 2, size=(N_POINTS, 2 * DIM))
    min_eigs = np.empty(N_POINTS)
    for k, p in enumerate(pts):
        min_eigs[k] = np.linalg.eigvalsh(hessian_fd(p)).min()
        if (k + 1) % 25 == 0:
            print(f"{k+1}/{N_POINTS}  running min {min_eigs[:k+1].min():.4f}  "
                  f"frac<0 {(min_eigs[:k+1] < -1e-3).mean():.1%}", flush=True)
    print("\n=== Cucker-Smale (dim=2, 4D state) SDRE value function, interior [-2,2]^4 ===")
    print(f"min eig: min {min_eigs.min():.4f}, p5 {np.percentile(min_eigs,5):.4f}, "
          f"median {np.median(min_eigs):.4f}, max {min_eigs.max():.4f}")
    print(f"fraction non-convex (eig < -1e-3): {(min_eigs < -1e-3).mean():.1%}")
    bad = pts[min_eigs < -1e-3]
    if len(bad):
        print(f"non-convex points: |pos| up to {np.abs(bad[:, :DIM]).max():.2f}, "
              f"|vel| up to {np.abs(bad[:, DIM:]).max():.2f}")
        print(f"worst point: {pts[min_eigs.argmin()]}  eig {min_eigs.min():.4f}")


if __name__ == "__main__":
    main()
