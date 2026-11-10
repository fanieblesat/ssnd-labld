"""Choose the initial Polyak factor after an accepted learned start (theta_warm) on TRAINING instances.

    python tune_theta.py --models models/s60.pkl --seeds 301 302 303 --scenarios 60 --bundles 12 \
        --thetas 1 0.5 0.25 --iters 20 --out results/tune_theta

For every training seed and theta: stratified partition, learned start through the guard, `iters`
iterations of the decomposition. The chosen theta maximizes the mean lower bound after `iters`
iterations, relative to theta = 1 on the same instance; ties go to the larger theta (the default).
The test instances (Study A and the 240-scenario instances) are never used for this choice.
Writes tune.csv (one row per seed and theta), history.csv and chosen_theta.txt. Resumable.
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from ssnd_labld import BundledLagrangian, generate_instance, get_solver, stratified_bundles
from ssnd_labld.learning import DualStartPolicy, bundle_lp_designs, load_models


def append(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[301, 302, 303])
    ap.add_argument("--scenarios", type=int, default=60)
    ap.add_argument("--bundles", type=int, default=12)
    ap.add_argument("--terminals", type=int, default=6)
    ap.add_argument("--periods", type=int, default=6)
    ap.add_argument("--commodities", type=int, default=10)
    ap.add_argument("--thetas", type=float, nargs="+", default=[1.0, 0.5, 0.25])
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--bundle-time-limit", type=float, default=300.0)
    ap.add_argument("--recovery-every", type=int, default=3)
    ap.add_argument("--solver", type=str, default="HiGHS")
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    T_csv, H_csv = os.path.join(args.out, "tune.csv"), os.path.join(args.out, "history.csv")
    done = set()
    if os.path.exists(T_csv):
        with open(T_csv) as f:
            done = {(r["seed"], r["theta"]) for r in csv.DictReader(f)}
    policy, _, _ = load_models(args.models)
    if not isinstance(policy, DualStartPolicy):
        raise SystemExit("tune_theta needs a DualStartPolicy model")
    lp = get_solver(args.solver)
    for seed in args.seeds:
        inst = generate_instance(n_terminals=args.terminals, n_periods=args.periods,
                                 n_commodities=args.commodities, n_scenarios=args.scenarios,
                                 kappa=0.0, seed=seed, spot_requires_activation=True)
        B = stratified_bundles(inst, args.bundles, seed=seed, solver=lp, parallel=args.parallel)
        _, y_lp = bundle_lp_designs(inst, B, lp)
        lam = policy.multipliers(inst, B, y_lp=y_lp)
        for th in args.thetas:
            if (str(seed), str(th)) in done:
                continue
            ld = BundledLagrangian(inst, B, get_solver(args.solver, time_limit=args.bundle_time_limit),
                                   step_rule="polyak", theta=1.0, tau=0.5, max_iter=args.iters,
                                   gap_tol=1e-3, parallel=args.parallel, verbose=False,
                                   recovery_every=args.recovery_every, recourse_solver=lp,
                                   eval_cache={}, theta_warm=th)
            res = ld.run([lam], ["ml"])
            for h in res.history:
                append(H_csv, dict(seed=seed, theta=th, t=h["t"], L=round(h["L"], 4),
                                   LB=round(h["best_lb"], 4), time=round(h["time"], 2)))
            append(T_csv, dict(seed=seed, theta=th, start=res.start_source, lb=round(res.best_lb, 4),
                               ub=round(res.best_ub, 4) if np.isfinite(res.best_ub) else "",
                               iters=res.iterations, seconds=round(res.seconds, 1)))
            print(f"seed {seed}  theta_warm={th:<5}  start={res.start_source}  LB={res.best_lb:.2f}  "
                  f"({res.seconds:.0f}s)", flush=True)
    # choice: mean LB relative to theta = 1 on the same seed
    with open(T_csv) as f:
        R = list(csv.DictReader(f))
    lb = {(int(r["seed"]), float(r["theta"])): float(r["lb"]) for r in R}
    base = 1.0 if 1.0 in args.thetas else max(args.thetas)
    score = {th: float(np.mean([100 * (lb[(s, th)] / lb[(s, base)] - 1) for s in args.seeds]))
             for th in args.thetas}
    for th in sorted(score, reverse=True):
        print(f"theta_warm={th:<5}  mean LB change vs theta={base}: {score[th]:+.3f}%")
    best = max(sorted(args.thetas, reverse=True), key=lambda th: score[th])
    open(os.path.join(args.out, "chosen_theta.txt"), "w").write(f"{best}\n")
    print(f"chosen theta_warm = {best}")


if __name__ == "__main__":
    main()
