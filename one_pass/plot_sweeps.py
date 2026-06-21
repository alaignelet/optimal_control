"""Capacity-sweep figure: eval MSE vs model size for LSE-quad (vs K) and
FICNN (vs width), on NonLinear2D. Shows the universal-approximation trend of
LSE-quad against the trainability ceiling of the FICNN."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def mse(tag):
    return json.load(open(os.path.join(R, tag, "metrics.json")))["eval_mse"]


lseq_K = [1, 4, 16, 64]
lseq_mse = [mse(f"lseqK{k}_nl_t2") for k in lseq_K]

# FICNN: parameter count grows with width^2*depth; plot vs width for fixed d=3
ficnn = [(64, mse("ficnn_nl")), (128, mse("ficnn_nl_w128d3")),
         (256, mse("ficnn_nl_w256d3"))]

fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))

ax = axs[0]
ax.loglog(lseq_K, lseq_mse, "o-", color="C0")
ax.axhline(5.9e-6, ls="--", color="grey", lw=1, label="penalty+MLP (5.9e-6)")
ax.set_xlabel("number of quadratic components $K$")
ax.set_ylabel("eval MSE vs ground truth")
ax.set_title("LSE-quad: accuracy improves with $K$\n(universal convex approximation)")
ax.grid(True, which="both", alpha=0.2)
ax.legend(fontsize=8)
for k, v in zip(lseq_K, lseq_mse):
    ax.annotate(f"{v:.1e}", (k, v), textcoords="offset points", xytext=(4, 6), fontsize=7)

ax = axs[1]
ws = [w for w, _ in ficnn]
vs = [v for _, v in ficnn]
ax.semilogy(ws, vs, "s-", color="C3")
ax.set_xlabel("hidden width (depth 3)")
ax.set_ylabel("eval MSE vs ground truth")
ax.set_title("FICNN: accuracy does NOT improve with width\n(trainability ceiling)")
ax.grid(True, which="both", alpha=0.2)
ax.set_xticks(ws)
for w, v in zip(ws, vs):
    ax.annotate(f"{v:.1e}", (w, v), textcoords="offset points", xytext=(4, 6), fontsize=7)

fig.tight_layout()
out = os.path.join(R, "capacity_sweeps.png")
fig.savefig(out, dpi=150)
print(out)
