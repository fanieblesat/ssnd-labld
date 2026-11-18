# ssnd_labld — reference implementation for the SSND / bundled-Lagrangian draft

Companion code for the manuscript draft *Learning-Augmented Lagrangian Decomposition for
Sustainable and Resilient Multimodal Freight Service Network Design under Uncertainty*.
It implements the model of Section 3 (with the audit fixes), the deterministic equivalent,
and the bundled Lagrangian method of Section 4 with a **consistent** projection/subgradient
convention. There is **no GNN here yet** — the code exposes the hook where a learned predictor
plugs in (`BundledLagrangian.run(candidates=[lam_ml])`) and the guard of eq. (guard) takes over.

```
deliverables/
├── AUDIT_REPORT.md          line-level audit of main.tex / references.bib / shipped PDF
├── references.bib           cleaned bibliography (drop-in replacement) + suggested additions
├── requirements.txt
├── run_example.py           end-to-end run + numerical checks of Propositions 1 and 2
├── tests/test_validity.py   pytest version of the same checks (small instance)
└── ssnd_labld/
    ├── instance.py          Arc / Commodity / Scenario / SSNDInstance   (A = A_S ∪ A_0)
    ├── generator.py         synthetic multimodal corridor + joint demand/disruption scenarios
    ├── models.py            recourse LP, deterministic equivalent, bundle subproblem (PuLP)
    ├── bundling.py          singleton / random / k-means (demand–disruption) / solution-similarity
    ├── warmstart.py         Pi projection, zero start, LP-dual start (+ hook for learned starts)
    └── lagrangian.py        BundledLagrangian: guard, projected subgradient, Polyak/diminishing,
                             primal recovery (voting + repair + add/drop), OHC diagnostics
```

## Install and run

```bash
pip install -r requirements.txt          # numpy, scipy, pulp (bundles the CBC solver), scikit-learn (optional)
python run_example.py                    # default: 4 terminals, 4 periods, 6 commodities, 12 scenarios
python run_example.py --scenarios 8 --bundles 3 --iters 12 --quiet
python run_example.py --spot-requires-activation      # model variant B (audit 4.1)
python run_example.py --solver GUROBI                 # any PuLP solver name (GUROBI, CPLEX_CMD, HiGHS, ...)
pytest -q tests                                       # ~1 min with CBC
```

The example (i) solves the deterministic equivalent for z*, (ii) checks L(λ) ≤ z* on random
balanced λ (Prop. 1), (iii) checks that coarsening the partition does not weaken the bound
(Prop. 2, using μ_B = Σ_{b⊂B} λ_b), (iv) runs the Lagrangian method with four partitions and
asserts LB* ≤ z* ≤ UB* for each.

## Model implemented (Section 3 with the audit fixes)

* First stage: y_a ∈ {0,1} for **service arcs A_S only**; auxiliary arcs A_0 (holding/transfer)
  have no activation variable and are optionally capacitated (audit 4.3).
* Recourse per scenario ω: flows x_{ak}^ω ≥ 0, spot capacity 0 ≤ r_a^ω ≤ r̄_a, shortage
  0 ≤ u_k^ω ≤ d_k^ω; flow conservation with net supply b_{ik}(d_k − u_k); capacity
  Σ_k x_{ak} ≤ ξ_a^ω U_a y_a + r_a^ω on A_S, Σ_k x_{ak} ≤ cap_a on capacitated A_0.
* `spot_requires_activation=True` adds r_a^ω ≤ r̄_a y_a (variant B, audit 4.1).
* Objective: Σ F_a y_a + Σ_ω p_ω [ Σ (c_{ak} + κ e_{ak}) x + Σ (c_a^r + κ e_a^r) r + Σ π_k u ].
  The optional `e_r` field is the emissions surcharge on spot capacity that makes the
  "normal vs. emergency emissions" sentence of the paper true (audit 4.2).
* Reported: EM(y), REL(y), expected spot use, per-scenario recourse.

## Lagrangian method (Section 4, with the audit fixes)

* Bundle subproblem φ_b(λ_b) = min P_b Fᵀy^b + λ_bᵀy^b + Σ_{ω∈B_b} p_ω c_ω(·) — a MILP with one
  binary design vector and the recourse LPs of the bundle's scenarios (original p_ω, not
  renormalised, so Σ_b φ_b is directly comparable with z*).
* Balanced space Λ = {Σ_b λ_b = 0}; projection Π(λ)_b = λ_b − (1/m)Σ_c λ_c;
  **projected subgradient g_b = y^b − (1/m)Σ_c y^c** (audit 4.7 — not the PH-weighted
  centring). Update λ ← Π(λ + α g).
* Steps: `polyak` (θ_t (U − L)/‖g‖², needs U from primal recovery) or `diminishing`
  (α₀/√(t+1)).
* Guarded start over any list of candidates: {0} ∪ {Π(candidate)}; the run reports which one won
  (`start_source`). With CBC the LP-dual start needs the **negative** sign (λ = −π); pass both
  `[lp, -lp]` and let the guard choose, as `run_example.py` does.
* Primal recovery: vote s_a = Σ_b P_b y_a^b, ŷ = 1[s_a ≥ τ]; recourse repair (add strongest-vote
  services while shortage/spot remains and it pays); first-improvement add/drop local search;
  every candidate is evaluated exactly with all |Ω| recourse LPs (valid U). Evaluation budget
  `ls_max_evals` per iteration.
* Bundle solutions at λ⁰ are reused in iteration 0 (no redundant re-solve; audit 5.1).
* Diagnostics: `time_to_target(history, target)` and
  `oracle_headroom_capture(T0, Tml, Toracle, min_headroom=0.1)` (OHC only reported when
  headroom ≥ 10 %, audit 5.3).

## Where the learned components plug in

* **Learn to decompose**: `bundling.scenario_features()` builds φ(ω) of eq. (scenfeatures);
  replace `kmeans_bundles` by any learned partition (return a list of scenario-id lists).
* **Learn to dualize**: produce an (m × n) array of raw multipliers from your GNN and pass
  `candidates=[lam_ml]` to `run()`; the projection and the guard are applied automatically.
  Training targets λ* can be generated by long runs (`max_iter` large, `step_rule='polyak'`)
  and reading `result.lam_final`.

## Notes / limitations

* PuLP + CBC keeps everything licence-free; for the real study switch to Gurobi/CPLEX via
  `get_solver("GUROBI")` etc. Bundle subproblems are solved in parallel threads (`parallel=k`).
* The generator is a controlled synthetic corridor (road/rail/barge, joint demand + disruption
  regimes). It is meant for method development and the ablation protocol, not as the benchmark
  set of the paper.
* The multimodal structure currently enters through mode-specific cost/emission/capacity/travel
  time of service arcs and holding arcs between periods; explicit inter-modal transfer arcs and
  fleet constraints in Y are straightforward additions (see audit 4.5).

## Update: running larger experiments (time limits, cache, HiGHS, cost-signature bundling)

New in this version
* `--de-time-limit` (default 600 s) and `--bundle-time-limit` (default 120 s). Validity is preserved: if a
  bundle MILP is stopped early, the **LP-relaxation value** is used for L(lambda) (still a valid lower bound);
  if the DE is stopped early, it is reported as "incumbent + LP bound" and the checks that need a proved z* are
  skipped. Unproved bundle solves are counted in the run summary.
* `evaluate_design` caches results by the 0/1 design; recourse LPs are solved in parallel (`--parallel`).
* `--recovery-every k`, `--ls-evals n` reduce the cost of primal recovery (the dominant cost on large instances).
* `--partitions singleton,random,kmeans,solsim,costsig` selects which partitions to run; `costsig` is the new
  cost-signature bundling (k-means on recourse costs at a few candidate designs).
* `--no-checks`, `--skip-de` for big instances; `--solver HiGHS` (pip install highspy) is accepted.

Measured here (2 cores, CBC), variant B (`--spot-requires-activation`)
| size (T/P/K/S/m) | DE time | one partition run |
|---|---|---|
| 4/4/6/12/4  | 1.5 s  | ~35 s (15 iters) |
| 5/5/8/20/5  | 40 s   | ~90-110 s (10 iters, recovery every 2) |
| 6/6/10/30/6 | 928 s (your run) | 1200-4750 s (30 iters) |

Suggested ladder
    python run_example.py --spot-requires-activation --terminals 4 --periods 4 --commodities 6 --scenarios 12 --bundles 4 --iters 25 --parallel 4
    python run_example.py --spot-requires-activation --terminals 5 --periods 5 --commodities 8 --scenarios 20 --bundles 5 --iters 25 --parallel 4 --recovery-every 2 --no-checks
    python run_example.py --spot-requires-activation --terminals 6 --periods 6 --commodities 10 --scenarios 30 --bundles 6 --iters 30 --parallel 4 --recovery-every 3 --no-checks --partitions kmeans,costsig,solsim --de-time-limit 1800

## Update v3: step-size control and balanced bundles
* Polyak theta is halved after `--theta-patience` (default 4) iterations without LB improvement, and the iterate
  restarts from the best multiplier. v2 kept theta = 1 throughout, so with a poor early UB the step overshot and
  L(lambda) oscillated (e.g. 2861 -> 741 in one step in the 6/6/10/30 run).
* `--max-bundle-size` (default ceil(1.5*S/m); 0 = no cap) caps bundle sizes for kmeans/solsim/costsig via a
  capacitated reassignment to the cluster centroids. Uncapped clustering produced bundles of 13-20 scenarios,
  which are nearly as hard as the DE and hit the bundle time limit (their LP fallback weakens the bound).

## Update v4: stratified partition and the seed-study script
* `stratified_bundles` (partition name `stratified`): clusters scenarios into ceil(S/m) strata of similar
  scenarios and deals each stratum across the bundles, so every bundle is diverse by construction.
* `run_study.py`: seeds x partitions, writes `summary.csv`, `history.csv`, `de.csv` to `--out`.
  It appends and skips finished (seed, partition) pairs, so an interrupted study is resumed by rerunning
  the same command. Defaults: variant B, 30 iterations, recovery every 2, bundle limit 600 s, DE limit 3600 s.

## Update v5: learning components (`ssnd_labld/learning.py`, `run_learning.py`)
Both components are *safe by construction*: they only choose where the decomposition starts, never
what it proves. Every L(lambda) remains a valid lower bound (Proposition 1).

* **Learned dual warm start** (`DualStartPolicy`). lambda(w) = F (.) Pi(sum_k w_k d_k), where the
  directions d_k come from size-independent features: each bundle's availability of each service arc
  relative to the whole scenario set, its demand level, its LP-relaxation design at lambda = 0, and its
  probability mass. The weights w are fitted on training instances by derivative-free coordinate search
  that maximises the mean *closure* (L(lambda(w)) - L(0)) / (LB* - L(0)), the share of the bound gap the
  start alone closes; a share of the training runs is held out for validation. The start is projected
  onto Lambda and passed to the guard, so it is never worse than lambda = 0.
  `--ws-model regression` instead fits `MultiplierPredictor` (gradient boosting onto lambda* of long
  runs). It is kept for comparison: lambda* is not unique, and the regression did not generalise in our
  tests (grouped-CV R^2 below 0).
* **Learned partition selection** (`PartitionScorer`). Scores a pool of valid partitions (k-means,
  cost-signature, stratified on two bases, random, swap perturbations) with ridge regression on
  structural features standardised within the pool: diversity inside bundles, similarity between
  bundles, regime coverage, probability and size balance; the default hybrid scorer adds the LP proxy
  L_LP(0) (sum of bundle LP relaxations, m LPs per candidate) as a feature (`--scorer struct` drops it,
  then selection needs no solves). Trained on L(0) standardised within each instance, i.e. it learns a
  ranking. The raw LP proxy is evaluated as the non-learning reference.

Smoke test in the build container (2 cores; 40 training instances 3/4/5/10/3; not for the paper):
partition scorer grouped-CV Spearman 0.61 vs 0.59 for the raw LP proxy (top-1 regret 0.58% vs 0.64%);
ridge coefficients: + diversity inside bundles, - difference between bundles (the Study A pattern);
policy closure 0.28 on training runs, 0.14 on held-out runs.

Workflow (training seeds are disjoint from the test seeds):
```bash
python run_learning.py collect --seeds 101-160 --terminals 4 --periods 4 --commodities 6 \
    --scenarios 12 --bundles 4 --solver HiGHS --out data/train_small
python run_learning.py train --data data/train_small --out models/small.pkl
python run_learning.py eval --models models/small.pkl --seeds 1-5 --terminals 5 --periods 5 \
    --commodities 8 --scenarios 20 --bundles 5 --solver HiGHS --de-csv results/studyA/de.csv \
    --pool-l0 --out results/learnA
```
`eval` writes `warmstart.csv` (zero / ml / oracle starts on the same partition; add `lpdual` via
`--starts`), `bundling.csv` (random / stratified / learned / LP-proxy choice; `pool_best_L0` is the best
L(0) attainable from the pool) and `history.csv`. All three commands are resumable.

Other v5 changes
* `requirements.txt`: PuLP pinned below 4 (PuLP 4 changed the modelling API), `highspy` added,
  scikit-learn now required.
* `get_solver`: falls back to HiGHS with a warning when the CBC binary is not available.
* `pytest.ini`: `pythonpath = .`, so plain `pytest -q` finds the package.

## Update v6: warm-transfer baseline and scaling study
* **Warm transfer** (`WarmTransfer` in `learning.py`, start `transfer` in `run_learning.py eval`): the
  best multiplier of the nearest solved training instance of the *same network and number of
  bundles*, bundles matched by an assignment on bundle features (mass, demand level, regime mix,
  availability), transferred as lambda*/F and rescaled by the test instance's fixed costs, projected,
  and guarded. Training runs at the test size come from `collect --skip-pool` on disjoint seeds:
  ```bash
  python run_learning.py collect --seeds 201-220 --terminals 5 --periods 5 --commodities 8 \
      --scenarios 20 --bundles 5 --mult-parts stratified --skip-pool --solver HiGHS --parallel 6 \
      --out data/transfer_A
  python run_learning.py eval --models models/small.pkl --seeds 1-5 --terminals 5 --periods 5 \
      --commodities 8 --scenarios 20 --bundles 5 --starts zero,ml,transfer,lpdual,oracle \
      --transfer-data data/transfer_A --skip-bundling --solver HiGHS --parallel 6 \
      --de-csv results/studyA_ryzen/de.csv --out results/learnA_transfer
  ```
* **Scaling study** (`run_scaling.py`): DE and LD under the same wall-clock budget for increasing
  |Omega|, m = |Omega| / bundle size. Reports the DE incumbent, its proven bound and gap at the
  time limit, and the LD bounds over time (`ld_lb_<T>s` columns for equal budgets).
* Time-limited MILPs (DE and bundle subproblems) now report the HiGHS MIP dual bound when it is
  stronger than the LP relaxation; both are valid lower bounds.

* `eval --ws-partition learned` (or `lpproxy`) runs the warm-start comparison on the partition chosen
  from the candidate pool; with `--starts zero,ml` this gives "learned partition + zero start" and the
  full LABLD (learned partition + learned start) on the same partition.
* `eval --starts ""` with `--skip-bundling` omitted runs only the partition comparison.

## Update v7: primal polishing in the scaling study
`run_scaling.py --polish-time T` (default 300 s) reserves the last T seconds of the budget for a
restricted deterministic equivalent: every service that no bundle solution opened during the run
is fixed closed, and HiGHS solves the remaining problem for the rest of the budget. Any solution is
feasible for the full problem, so its exact expected cost (re-evaluated on all scenarios) is a
valid upper bound; the final LD upper bound is the better of the voting/local-search incumbent and
the polished one. The lower bound is unaffected. `scaling.csv` reports `ld_ub_before_polish`,
`polish_ub`, `polish_services` and `polish_time`; DE and LD still share the same total budget.

## Update v8: sustainability and resilience study (`run_sustainability.py`)
For each seed, disruption setting (`base`: default regime probabilities; `stressed`: 20/25/30/25 %
nominal/localized/corridor/severe) and carbon price kappa, the deterministic equivalent is solved on
|Omega_train| scenarios and the design is evaluated out of sample on |Omega_test| independent
scenarios of the same network (the generator draws the network before the scenarios, so the
training scenarios equal those of Study B for the same seed). Reported per row: operating cost
without the carbon charge, CO2e, CO2e from emergency capacity, reliability (overall, nominal and
severe scenarios), emergency capacity, unserved share, tonne-km shares by mode, services opened by mode.
`--spot-emission-factor` (default 1.0 in this study only) makes emergency capacity emit that multiple
of road CO2e per unit and distance; all other studies keep 0, so earlier results are unchanged.
```bash
python run_sustainability.py --seeds 1 2 3 4 5 --kappas 0 0.5 1 2 4 --settings base stressed \
    --solver HiGHS --parallel 6 --out results/sustainability
```


## v9: scale-invariant warm start and training at 60 scenarios

The learned start is now lambda_b = Pi(P_b F (.) sum_k w_k d_k): multipliers are proportional to the
fixed cost each bundle carries (P_b F), so the same weights apply for any number of bundles.
Policies up to v8 used lambda_b = Pi(F (.) ...), which overshoots by m/m_train at larger m; this is
why the guard rejected the start at 240 scenarios (m = 48). `run_learning.py rescale` converts an
old policy exactly at its training m; `train --init-from --scorer-from --scale` start a new policy
search from it. `check_start.py` compares L(0) and L(lambda_ml) in minutes before a long run.

    bash run_all.sh        # phase1.sh then phase2.sh, logs in logs/


## v10: smaller first step after an accepted learned start

`BundledLagrangian(theta_warm=...)` starts the Polyak factor at `theta_warm` instead of 1 when the guard
accepted a non-zero start. `tune_theta.py` chooses the value on training instances only; `run_scaling.py`
and `run_learning.py eval` take `--theta-warm`. `phase3.sh` runs the choice and the test runs.

## Repository layout

    ssnd_labld/        the method: generator, models, bundling, Lagrangian decomposition, learning
    run_*.py           experiment scripts (studies, learning, scaling, sustainability)
    check_start.py     quick acceptance check of a learned start
    tune_theta.py      choice of theta_warm on training instances
    phase*.sh          the pipelines used for the paper's later experiments
    models/            trained models (scorer and warm-start policies)
    paper_results/     the result files behind every table and figure, and the plotting scripts
    tests/             unit tests (python -m pytest -q)

Raw run outputs (`data/`, `results/`, `logs/`) are not tracked.

## v11: Gurobi

`--solver GUROBI` uses Gurobi through gurobipy (`pip install gurobipy`, plus a licence). Every solve gets
its own Gurobi environment, so the decomposition can solve bundles concurrently; bundle MILPs use one
thread each, and the deterministic equivalent uses `--parallel` threads. Time-limited solves report
Gurobi's MIP dual bound (ObjBound), as with HiGHS.

## v12: SND instances (IDSR benchmark) and a persistent Gurobi backend

`ssnd_labld/snd_io.py` reads the SND_V1 format (multicommodity network design with stochastic demand);
the model is the special case of ours with design arcs only, no spot capacity and no disruptions.
`ssnd_labld/grb_backend.py` builds every bundle MILP and recourse LP once with gurobipy and only updates
objective coefficients (bundles) or design bounds (recourse) afterwards; `BundledLagrangian(backend=...)`
uses it. `solve_de_gurobi` solves the extensive form with the same strengthening as the benchmark
(q_aks <= min(U_a, d_ks) y_a). `run_snd.py` compares both under the same time budget.
