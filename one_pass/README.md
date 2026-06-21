# One-pass convex value-function experiment

**Goal:** replace the published two-pass training (SDRE-supervised pre-training →
PDE-residual refinement; Borovykh, Kalise, Laignelet, Parpas 2022) with a
**single pass on the PDE residual only**, where convexity of the value function
is enforced **architecturally** (ICNN) instead of being instilled by supervised
data.

## Why the previous attempt was inconclusive

Two issues found when this work stopped in Oct 2024:

1. **Non-convex activations were used.** The Cucker–Smale runs used GELU (and
   SELU/SiLU/Mish were shortlisted in `script.ipynb`). An ICNN is only
   guaranteed convex if activations are *convex and non-decreasing*
   (ReLU, Softplus, ELU, CELU, LeakyReLU). With GELU, positive weights do
   **not** imply a convex network — so the guarantee was silently broken.
2. **The Hessian study targeted the wrong object.** The final cells of
   `../script.ipynb` checked convexity of the *SDRE-simulated true value
   function* of Cucker–Smale (finding ~104 non-convex points near the domain
   boundary for dim=2), not of the trained network. Open modelling question:
   the true Cucker–Smale value function may not be globally convex, in which
   case a convex ansatz is misspecified there (but fine for LQR, and to be
   checked for NonLinear2D).

## Protocol

Sanity ladder, in order:

1. **LQR 2D** — true V is a convex quadratic (Riccati). A one-pass residual-only
   ICNN must recover it. If this fails, the problem is optimisation, not
   modelling.
2. **NonLinear2D** — semi-Lagrangian ground truth available in `../inputs/`.
3. **Cucker–Smale** — only after settling whether its true V is convex on the
   training domain (re-run the SDRE Hessian study away from the boundary with a
   larger finite-difference stencil; boundary points may be artefacts).

For each rung compare:
- `--arch linear` (unconstrained baseline, Sigmoid MLP)
- `--arch convex --activation relu --positivity icnn --init convex` (guaranteed ICNN)
- `--arch convex --activation softplus --positivity convex --init convex` (smooth ICNN — Softplus has nonvanishing 2nd derivative, may help residual training)
- `--arch convex --activation gelu ...` (ablation: reproduces the broken-guarantee setting)

## Running

```sh
cd one_pass
../venv/bin/python run_experiment.py --arch convex --activation relu --positivity icnn
../venv/bin/python run_experiment.py --arch linear
```

Outputs per run in `results/<tag>/`: `metrics.json` (final loss, eval MSE vs
ground truth, Hessian min-eigenvalue summary), `training_history.csv`,
`non_convex_points.csv` (if any), `model.pt`.

Key fields in `metrics.json`:
- `eval_mse` — out-of-sample MSE against the analytical/numerical ground truth,
  shift-corrected (residual-only training determines V up to a constant — the
  HJB residual only involves ∇V — hence `correctShift=True` for one-pass runs).
- `hessian_global_min_eigenvalue` / `hessian_violation_fraction` — exact
  autograd Hessian of the *learned network* on a grid (`convexity_check.py`).
  For a guaranteed ICNN these should be ≥ 0 up to numerical tolerance; if not,
  there is a bug.

## First results — LQR2D ladder (2026-06-12)

One-pass residual-only, default budgets (lrs 1e-2/1e-3/1e-4 × 2k/4k/4k epochs),
seed 0, results in `results/full_*/`:

| Run | final residual loss | eval MSE | Hessian min eig | convex? |
|---|---|---|---|---|
| linear baseline (Sigmoid MLP) | 1.5e-5 | 4.35e-2 | -0.46 | no (100% viol) |
| ReLU + LazyClipped ICNN | 1.6e-1 | 7.71e-2 | 0.0 | yes (trivially) |
| Softplus + ExponentialPositivity | 5.7e+2 | 9.69e-1 | +1.05 | yes (strictly) |
| GELU ablation (broken guarantee) | 4.7e-8 | 4.63e-2 | -0.29 | no (100% viol) |

Interpretation:

1. **Trainability vs guarantee is the core tension.** Convex-safe activations
   train poorly here (ReLU stalls — its eval MSE is bit-identical to a
   150-iteration smoke run, i.e. it stops improving almost immediately;
   Softplus+exp-positivity barely trains, loss 571), while GELU trains
   beautifully but voids the guarantee. This empirically reproduces why the
   2024 attempt felt inconclusive.
2. **ReLU's "convex" verdict is vacuous.** A ReLU network is piecewise linear:
   Hessian ≡ 0 a.e., so min-eig 0.0 passes the check without measuring
   anything. Use Softplus/ELU/CELU for a meaningful curvature check.
3. **Residual ≈ 0 ≠ correct solution.** GELU drove the residual to 4.7e-8 yet
   its eval MSE (4.6e-2) is no better than the baseline: residual-only
   training can settle on spurious near-solutions of the HJB equation (the
   PDE does not have a unique classical solution — this is exactly why the
   2022 paper needed the SDRE data pass). **Working hypothesis: convexity is
   the structural prior that selects the right solution without data** — that
   is the one-pass thesis to test properly.

## Round 2 — convex-safe sweep (2026-06-12, `results/r2_*`)

- **ReLU stall explained** (`diagnose_stall.py` on `full_relu_icnn`): dying
  ReLU + positivity. Layer 2: 25/32 dead units; layer 4: 31/32 dead; gradient
  exactly zero everywhere except the output weights. Non-negative weights +
  the convex-init's negative biases kill units; under ReLU they never recover.
  Note `ConvexNeuralNet` applies the convex initialiser only to ConvexLinear
  layers — the first Linear layer keeps PyTorch default init (inconsistent
  with the Hoedt–Klambauer scheme).
- **Smooth convex activations fix the death**: Softplus/CELU/ELU + LazyClipped
  + convex-init train (residual ~6.7e-3) and are STRICTLY convex (min eig > 0).
- **Stall basin is universal**: LeakyReLU, Softplus+traditional-init, and the
  wide Softplus all collapse to bit-identical loss (1.57e-1) — same degenerate
  near-affine function. Convex-init is necessary for Softplus.
- BUT all convex runs sit at eval MSE 7e-2–9e-2 vs baseline 4.4e-2; residual
  plateaus ~6e-3 vs GELU's 4.7e-8.

## Round 3 — longer budgets (`results/r3_*`)

28k epochs / 4-stage lr changed NOTHING (CELU 6.6e-3, Softplus 6.0e-3,
eval identical). **The hard-ICNN underfit is a wall, not slow convergence.**

Additional headline across all rounds: even residual = 4.7e-8 (GELU) gives
eval MSE 4.6e-2 → one-pass residual-only training converges to **spurious
solutions of the stationary HJB** (non-unique without side conditions; for LQR
the anti-stabilizing Riccati branch also solves the PDE). Convexity is the
candidate selection principle — IF the convex model class can fit the
residual. Hard ICNNs can't. Hence:

## Round 4 — soft convexity penalty (`results/r4_*`)

Unconstrained net (GELU, `--positivity fc`) + differentiable hinge penalty on
the smallest Hessian eigenvalue at the interior points
(`--convexity-penalty W --margin M`, analytic 2x2 eigenvalue, reuses the
training loop's gradInt). One pass, convexity imposed on the OUTPUT over the
domain rather than on the weights. Sweep W ∈ {0.1, 1, 10}, margin ∈ {0, 0.1}.

**RESULT (2026-06-12): CONFIRMED — and decisively.** On LQR2D, seed 0:

| config | residual | eval MSE | min eig | convex |
|---|---|---|---|---|
| unconstrained GELU | 4.7e-8 | 4.63e-2 | -0.29 | no |
| penalty 0.1 | 2.8e-2 | 4.58e-2 | -0.28 | no |
| penalty 1.0, margin 0 | 1.6e-4 | 4.9e-3 | -1.67 | 2% viol |
| penalty 10 | 3.2e-4 | 7.6e-3 | -0.28 | 3% viol |
| **penalty 1.0, margin 0.1** | **1.9e-7** | **1.7e-7** | **+0.60** | **strict** |

The strong-convexity margin (require λ_min ≥ 0.1 at interior points) recovers
the TRUE Riccati value function to 1.7e-7 — five orders better than anything
else — in one pass with zero supervised data. Interpretation: the true LQR
Hessian is P ≈ 0.69·I, so the margin keeps the truth feasible while excluding
the spurious HJB branches. Plain hinge at 0 already gives 10× over the
unconstrained baseline; the margin closes it completely. **Strong convexity is
the solution-selection principle that replaces the SDRE data pass.**

## Round 5 — robustness + NonLinear2D (`results/r5_*`)

- **LQR robustness: confirmed.** Seeds 1, 2 with penalty 1.0 / margin 0.1:
  eval MSE 2.7e-6 and 7.7e-8, strictly convex, 0% violations.
- **NonLinear2D first attempt: all failed** (eval ~1.2 for unconstrained,
  pen1, pen1+margin0.05 alike). Analysis of the ground truth
  (`inputs/non_linear_true_solution/neural_net/true_solution.csv`, layout =
  index column + 100x100 value matrix on [-1,1]²) shows the true V **is**
  convex — min Hessian eig ≥ 0.02 everywhere, median 0.73, range [0, 5.7].
  So the ansatz is well-specified; the failures are tuning:
  (a) margin 0.05 exceeded the true curvature floor 0.02 in places
  (infeasible constraint); (b) V scale ~8x LQR, so penalty weight 1.0 is
  relatively much weaker against the residual term; (c) the pen1 run's
  min-eig of -53 indicates the two loss terms fighting at too-high lr.

## Round 6 — NonLinear2D, corrected margins (`results/r6_*`)

Feasible margin 0.01, penalty ∈ {10, 100}, 28k epochs: penalty achieves
convexity (pen100: 0% violations) but residual walls at ~3e-2 and eval stays
~1.2. Unconstrained control: residual 5.3e-6, eval ~1.2, wildly non-convex.

## Rounds 7–8 — diagnostics (`results/r7_*`, `r8_*`)

- Supervised fit (γ_data=γ_grad=1): trains to 1.1e-3. **Exposed an eval bug**
  (below). Equilibrium anchor |V(0)|²+|∇V(0)|² (free problem knowledge, no
  data): on LQR improves the margin run to eval 2.1e-8 (best overall); on
  NonLinear2D anchor-only solves residual (4.9e-5) but still spurious.

## The transpose bug (FIXED in ../pdes.py, 2026-06-12)

`NonLinear._loadTrueSolution` flattened the 100x100 CSV without transposing,
while `getEvaluationPoints`' meshgrid flattening has x1 varying slowly →
ALL historical NonLinear2D eval MSEs were computed against a transposed
ground truth. Verified via the analytic SDRE formula: MSE(formula, csv)
= 0.108 as-is vs 0.0076 transposed. After the fix (`reevaluate.py` reloads
saved models): supervised run = **8.6e-3** ≈ exactly the SDRE-vs-truth gap,
i.e. the supervised model was near-perfect and the pipeline is sound.
Residual-only runs remain ~1.2 → genuinely spurious solutions.

## Feasibility + overfitting facts (2026-06-12)

- The CSV truth satisfies the coded residual to **MSE 3.4e-7** (finite-diff
  check) — formulation matches; "convex + residual≈0 + anchored" is feasible.
- The analytic SDRE formula has residual MSE 0.53 — it is NOT a solution
  (it's the pass-1 warm start of the 2022 paper, nothing more).
- Off-sample residual of the "residual 5e-6" unconstrained model: **5.1e-4**
  (100x degradation, max 0.62) — partial collocation overfitting to the FIXED
  500 interior points (sampled once in `pdes.train`). The convex pen100 model
  does not overfit (3.7e-2 on/off-sample) but is stuck in a local minimum
  3 orders above the truth's residual.

## Round 9 — resampled collocation (`results/r9_*`)

`--resample`: fresh interior points every step (expected-residual training,
standard PINN practice, absent from the original codebase). Full one-pass
stack on NonLinear2D = resample + penalty 10 + margin 0.01 + anchor 10,
28k epochs; unconstrained-resampled control; LQR sanity.

**Results:** LQR sanity fine (1.2e-6). NonLinear2D unconstrained+resampled:
expected residual 2.3e-5, eval still 1.20 → **the spurious branch is a
genuine alternative (near-)solution of the PDE, not a collocation artefact.**
Full stack: convex (min eig +0.0099, pinned at the margin), residual stuck
3.4e-2, eval 1.25. Conclusion of the day: on NonLinear2D the obstacle is now
purely the OPTIMIZATION LANDSCAPE inside the convex-constrained class — the
truth (convex, min-eig ≥ 0.02, residual 3.4e-7, anchored) is feasible but
Adam + fixed-weight hinge penalty doesn't reach it; the iterate sits pressed
against the margin surface while the truth lies strictly inside it.

## Round 10 — augmented Lagrangian + warm shaping (`results/r10_*`)

Both implemented (`--aug-lagrangian RHO --al-update-every K`: per-point
multipliers on a fixed 20x20 constraint grid; `--warm-shape N`: data-free
pre-fit to ½‖x‖²). **Neither breaks the wall**: all three combos land at
residual 2.6–3.5e-2, eval ~1.2, convex with min-eig pinned at the margin.
The optimizer converges to the SAME convex non-solution every time.

## Capacity probe (2026-06-12)

Direct supervised fit of the 32x3 GELU net to the CSV truth: **MSE 3.7e-6**
— capacity is not the issue. The truth-fitted network's own residual is
1.2e-3 (fit error amplified through derivatives). So the convex-constrained
stall at 3e-2 is purely an optimization-landscape failure, ~25x above where
the truth sits within this very function class.

## Round 11 — ε-continuation: SOLVED (`results/r11_*`)

`--eps-continuation N`: anneal the nonlinearity ε from 0 to 1 over the first
N steps. At ε=0 the problem IS LQR — which the one-pass stack solves exactly —
and the solution path V(ε) is continuous, so tracking it avoids the spurious
basin entirely. Homotopy continuation; data-free; conceptually linked to the
continuation theorems in the gradient-smoothing paper.

**RESULT (2026-06-12): NonLinear2D solved one-pass.**

| config | residual | eval MSE | min eig |
|---|---|---|---|
| continuation + penalty 10 / margin 0.01 + anchor + resample | 1.1e-5 | **5.9e-6** | +0.0202 |
| continuation + AL ρ=10 / margin 0.01 + anchor + resample | 5.6e-6 | **8.9e-6** | +0.0098 |

Six orders of magnitude better than every non-continuation attempt (~1.2),
three orders better than the SDRE warm start the 2022 two-pass paper used as
its supervised data (7.6e-3 from truth), and the learned curvature floor
(+0.0202) matches the true solution's measured floor (0.02) exactly. Both
constraint mechanisms (hinge penalty, augmented Lagrangian) work.

**THE ONE-PASS RECIPE (both problems solved, zero precomputed data):**
PDE residual (resampled collocation)
+ convexity hinge with strong-convexity margin
+ equilibrium anchor |V(0)|² + |∇V(0)|²
+ homotopy continuation in the nonlinearity (LQR → target problem).

Each ingredient is necessary per the ablation trail: no penalty → spurious
branch (r4); no margin → partial violations (r4); no continuation → landscape
wall (r6, r9, r10); resampling prevents collocation overfit (r9 facts).

## Cucker–Smale convexity: SETTLED — NOT convex (2026-06-12)

`cs_convexity_probe.py` (SDRE-rollout V, dim=2 → 4D state, 150 interior
points in [-2,2]^4, central differences h=0.05): **4.7% non-convex points,
min eig down to −0.25** (median curvature +0.34), violations well inside the
domain (|pos| ≤ 1.27, |vel| ≤ 1.84). The Oct 2024 boundary finding was real.
A convex ansatz is misspecified for Cucker–Smale on [-3,3] domains.
Options: convexity-region restriction, convex-plus-correction, or
anchor+continuation without the convexity selector (eval limited: no dense
ground truth for CS).

## 10D provably-convex nonlinear benchmark (`run_nd.py`)

Decoupled cubic: ẏᵢ = εyᵢ³ + uᵢ, l = ½‖y‖², β = ½, domain [-1,1]^10.
Scalar HJB has closed form v'(s) = 2βεs³ + s√(4β²ε²s⁴+2β) ⇒ V = Σᵢ v(xᵢ)
is STRICTLY convex with curvature floor √(2β) = 1 (margin 0.1 provably
feasible); ground truth by quadrature. The network sees a generic 10D
problem. High-d recipe changes: 2×2 analytic λ_min → **directional curvature
hinge** E_v[relu(m − vᵀHv)] on 4 random unit directions per point per step;
final audit = full autograd Hessian eigvalsh at 300 random points.
ε-continuation from the decoupled 10D LQR. Results in `results/nd10_full/`.

## LSE-of-quadratics architecture (`lseq.py`, runs `results/lseq_*`)

The "elegant" alternative to the penalty: V(x) = (m/2)‖x‖² + (1/τ)LSE(τ·qᵢ(x))
with qᵢ = ½xᵀLᵢLᵢᵀx + bᵢᵀx + cᵢ. Globally m-STRONGLY convex BY CONSTRUCTION
(Hess ⪰ mI + Σpᵢ LᵢLᵢᵀ ⪰ mI), with UNCONSTRAINED Euclidean parameters (any
Lᵢ gives PSD LᵢLᵢᵀ — no clipping, no dead faces; this is what the ICNN got
wrong: it enforces a sufficient weight-space condition with hostile
optimization geometry, whereas LSEQ builds the Hessian property into the
function form). Smooth max-of-quadratics = the max-plus (McEneaney) function
class for HJB; ∇V = mx + Σpᵢ(Aᵢx+bᵢ) is a softmax-gated mixture of linear
feedback laws; LQR exact at K=1. Flags: `--arch lseq --lseq-K --lseq-m
--lseq-tau` in both runners; trained with residual + anchor (+continuation)
and NO convexity penalty. Diagnosis recap of why ICNN failed (it is NOT a
locality issue — ICNN convexity is global): (1) clipped positivity = dead
faces/dead units, (2) nonneg weights bias signal propagation into one
degenerate attractor, (3) skip-free variant may lack expressivity.

## Later

1. LBFGS polishing after Adam (the repo's older NeuralNetNewtonMethod did this).
2. ICNN distillation of solved networks (certified convexity) — possibly
   superseded by LSEQ, which is already certified by construction.
3. ICNN-with-skip-connections control experiment.

## Success criteria

One-pass ICNN matches (or beats) the two-pass baseline's eval MSE on LQR2D and
NonLinear2D **without any supervised data** (`gamma data = gradient = 0`), with
`hessian_violation_fraction == 0`. The two-pass baseline numbers can be
regenerated from `../../optimal_control_good/experiments.py`
(`experimentTwoStepsLearning`).
