# optimal_control — one-pass convex HJB solver

Research codebase for learning value functions of optimal-control problems by
solving the Hamilton–Jacobi–Bellman (HJB) PDE with neural networks.

## Project history (read this first)

1. **Published paper (two-pass):** *Data-driven initialization of deep learning
   solvers for Hamilton–Jacobi–Bellman PDEs* — Borovykh, Kalise, Laignelet,
   Parpas (2022). Method: **pass 1** supervised on SDRE
   (State-Dependent Riccati Equation) values + gradients, **pass 2** reload
   weights and train on PDE residual only. Paper-era code lives in
   `../optimal_control_good/` (see `experiments.py:experimentTwoStepsLearning`
   and `sdre_vs_residual.ipynb`); the earliest version is in
   `../controlProblemSolver/` (2021).
2. **This repo (2024–): one-pass with architectural convexity.** Replace the
   two passes with a single residual-only pass, enforcing convexity of V via
   Input-Convex Neural Networks (ICNN: non-negative weights after layer 1 +
   convex non-decreasing activations). Work stopped Oct 2024; resumed Jun 2026.
3. **Current experiment:** `one_pass/` — see its README for the protocol.

`../optimal_control_old/` is this same git history ~10 commits behind, plus
779 TensorBoard runs from the Sep 2024 activation-function sweeps. Don't
develop there.

## State when work stopped (Oct 2024) — and the two gotchas

- **GELU broke the convexity guarantee.** Past Cucker–Smale runs used
  `ActivationFunctionEnum.GELU` inside `ConvexNeuralNet`. GELU is not convex,
  so those networks were never guaranteed convex regardless of weight
  positivity. Guaranteed-safe activations: RELU, SOFTPLUS, ELU, CELU,
  LEAKYRELU, RELU6.
- **The Hessian study in `script.ipynb` (cells ~88–109) analysed the SDRE
  ground truth, not the network.** It found ~104 non-convex points of the
  *true* Cucker–Smale value function (dim=2, mostly at the domain boundary —
  possibly finite-difference artefacts, epsilon=1e-4 stencil at the edge).
  Whether the true Cucker–Smale V is convex on the training domain is an OPEN
  QUESTION; if it isn't, a convex ansatz is misspecified for that problem.
- Last commit (`55224f4`, 2024-10-05) analysed the trained ICNN weight
  distributions.

## Code map

| File | Contents |
|---|---|
| `pdes.py` | `HamiltonJacobiBellman` base (sampling, loss, HJB residual) + problems: `LinearQuadraticRegulator{,ND,2D}`, `NonLinear{,2D}`, `CuckerSmale` (default dim=20) |
| `neuralNet.py` | `BaseNeuralNet` (training loop, autograd derivatives), `LinearNeuralNet` (Sigmoid MLP baseline), `ConvexNeuralNet` (ICNN: first layer free, rest `ConvexLinear`), `Matrix{Linear,Convex}NeuralNet` (output = upper-triangular P, V = ½xᵀPx) |
| `icnn.py` | `ConvexLinear` + positivity strategies (`ExponentialPositivity`=CONVEX, `LazyClippedPositivity`=ICNN, `NoPositivity`=FC) and `ConvexInitialiser` (Hoedt & Klambauer, NeurIPS 2023; reference clone at `~/Documents/08_PhD/convex-init/`) |
| `enums.py` | `ActivationFunctionEnum`, `PositivityFunctionEnum`, `InitFunctionEnum` |
| `generateData.py` | uniform / grid samplers over the domain |
| `one_pass/` | **current experiment** — one-pass runner + exact network-Hessian convexity check |
| `script.ipynb` | 2024 working notebook (Cucker–Smale, activation sweeps, SDRE Hessian study) |
| `inputs/` | precomputed ground truths: `non_linear_true_solution/` (semi-Lagrangian), `cucker_smale_data_solution/` |
| `exports/` | SPD-parameterisation figures (Cholesky vs Eigen, eigenvalue activations) |

## Conventions & gotchas for agents

- **Loss:** `gamma = {"data": γ_d, "gradient": γ_g, "residual": γ_r}` weights a
  3-term loss (supervised values, supervised gradients, HJB residual). One-pass
  = `{0, 0, 1}`. Two-pass = run twice with `{1,1,0}` then `{0,0,1}`.
- **Shift indeterminacy:** the stationary HJB residual only involves ∇V, so
  residual-only training determines V up to an additive constant. Pass
  `correctShift=True` so evaluation subtracts the min before comparing to the
  ground truth.
- **wandb:** `BaseNeuralNet.train` calls `wandb.log` unconditionally — always
  `wandb.init(...)` first (use `mode="disabled"` offline).
- **Imports are flat** (`from icnn import ...`), so scripts must run from the
  repo root or prepend it to `sys.path` (as `one_pass/run_experiment.py` does).
- **Environment:** `venv/` in this folder (Python 3.12, torch 2.4.1, numpy 2.x,
  scipy, pandas, wandb). Run as `./venv/bin/python ...`. NOTE: `requirements.txt`
  is stale (it's the convex-init repo's list, pins torch==1.12) — trust the venv,
  not the file.
- **iCloud eviction trap:** this repo lives under `~/Documents`, which iCloud
  "Optimize Mac Storage" evicts. If `import torch` hangs for minutes, the venv
  files are dataless — check with `stat` (`st_blocks==0` while size>0) and fix
  in bulk with `brctl download venv` BEFORE running anything. Consider moving
  the venv outside `~/Documents` permanently.
- Training is float32; `lossFunction` upcasts to double for the MSEs.
- `GenerateData.sampleGrid` ignores the domain bounds and hardcodes [-1, 1] —
  fine for LQR2D/NonLinear2D, wrong for Cucker–Smale ([-3, 3]); use random
  sampling there or fix it.
- **Transpose bug (fixed 2026-06-12):** `NonLinear._loadTrueSolution` used to
  flatten the CSV without transposing, mismatching the meshgrid order of
  `getEvaluationPoints` — every NonLinear2D eval MSE computed before the fix
  (including any 2024 numbers) was against a transposed ground truth. See
  `one_pass/README.md` and `one_pass/reevaluate.py`.
- **Interior points are sampled ONCE** in `HamiltonJacobiBellman.train` and
  fixed for the whole run — residual values can collocation-overfit (observed:
  100x degradation off-sample). `one_pass/run_experiment.py --resample`
  resamples every step.

## Roadmap (Jun 2026)

1. Run the `one_pass/` ladder on LQR2D: linear baseline vs ReLU-ICNN vs
   Softplus-ICNN vs GELU ablation. Expect: ICNN convex everywhere
   (`hessian_violation_fraction == 0`), eval MSE competitive with two-pass.
2. Same on NonLinear2D.
3. Settle the Cucker–Smale convexity question (redo the SDRE Hessian study in
   the domain interior, away from boundary artefacts).
4. If true V is non-convex somewhere: consider convex-plus-small-correction
   architectures or restrict claims to convex-V problem classes.
5. Write up: the one-pass result is the delta over the 2022 paper.
