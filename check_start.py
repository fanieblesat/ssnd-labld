"""Quick check of a learned warm start before a long run: L(0) against L(lambda_ml), and against
scaled starts L(alpha * lambda_ml), on the stratified partition that run_scaling.py uses.

    python check_start.py --models models/small_bundle.pkl --seeds 1 2 3 4 5 --scenarios 240 \
        --out results/check240_small.csv

The guard of the decomposition accepts the learned start only if L(lambda_ml) > L(0), so a
negative `gain` at alpha = 1 means the start will be rejected. One row per seed and alpha.
Takes a few minutes per seed (one bundle-MILP round per alpha). Resumable.
"""
from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np

from ssnd_labld import BundledLagrangian, generate_instance, get_solver, stratified_bundles
from ssnd_labld.learning import DualStartPolicy, bundle_lp_designs, load_models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--scenarios", type=int, default=240)
    ap.add_argument("--bundle-size", type=int, default=5)
    ap.add_argument("--terminals", type=int, default=6)
    ap.add_argument("--periods", type=int, default=6)
    ap.add_argument("--commodities", type=int, default=10)
    ap.add_argument("--alphas", type=float, nargs="+", default=[1.0, 0.5, 2.0])
    ap.add_argument("--bundle-time-limit", type=float, default=300.0)
    ap.add_argument("--solver", type=str, default="HiGHS")
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    mult, _, _ = load_models(args.models)
    if not isinstance(mult, DualStartPolicy):
        raise SystemExit("check_start needs a DualStartPolicy model")
    print(f"policy: w={np.round(np.asarray(mult.w, dtype=float), 4)}  scale={getattr(mult, 'scale', 'none')}")
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = {r["seed"] for r in csv.DictReader(f)}
    lp = get_solver(args.solver)
    for seed in args.seeds:
        if str(seed) in done:
            continue
        t0 = time.time()
        inst = generate_instance(n_terminals=args.terminals, n_periods=args.periods,
                                 n_commodities=args.commodities, n_scenarios=args.scenarios,
                                 kappa=0.0, seed=seed, spot_requires_activation=True)
        m = max(2, args.scenarios // args.bundle_size)
        B = stratified_bundles(inst, m, seed=seed, solver=lp, parallel=args.parallel)
        _, y_lp = bundle_lp_designs(inst, B, lp)
        lam = mult.multipliers(inst, B, y_lp=y_lp)
        ld = BundledLagrangian(inst, B, get_solver(args.solver, time_limit=args.bundle_time_limit),
                               verbose=False, parallel=args.parallel)
        L0 = ld.evaluate(np.zeros_like(lam))[0]
        print(f"seed {seed}  |Omega|={args.scenarios}  m={m}  L(0)={L0:.2f}", flush=True)
        for a in args.alphas:
            La = ld.evaluate(a * lam)[0]
            row = dict(seed=seed, S=args.scenarios, m=m, alpha=a, L0=round(L0, 4), L=round(La, 4),
                       gain=round(La - L0, 4), gain_pct=round(100 * (La - L0) / abs(L0), 4),
                       accepted=int(La > L0), seconds=round(time.time() - t0, 1))
            new = not os.path.exists(args.out)
            with open(args.out, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                if new:
                    w.writeheader()
                w.writerow(row)
            print(f"  alpha={a:<4}  L={La:.2f}  gain={La - L0:+.2f} ({row['gain_pct']:+.2f}%)  "
                  f"{'accepted' if La > L0 else 'REJECTED'}", flush=True)
    print(f"\nresults in {args.out}")


if __name__ == "__main__":
    main()
