"""Seed study: run several partitions on several seeds and write the results to CSV.

    python run_study.py --seeds 1 2 3 4 5 --scenarios 20 --terminals 5 --periods 5 --commodities 8 \
        --bundles 5 --partitions random,costsig,kmeans,stratified --out results/S20

Writes (appending, so an interrupted study can simply be restarted; finished runs are skipped):
  <out>/summary.csv   one row per (seed, partition): DE reference, L(0), LB*, UB*, gaps, counts, times
  <out>/history.csv   one row per iteration: t, L, LB*, UB*, theta, elapsed time
  <out>/de.csv        one row per seed: DE incumbent, bound, proved flag, time
Gaps vs. z* are computed only when the DE was proved optimal.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import time

import numpy as np

from ssnd_labld import (BundledLagrangian, cost_signature_bundles, generate_instance, get_solver,
                        kmeans_bundles, lp_dual_multipliers, random_bundles, singleton_bundles,
                        solution_similarity_bundles, solve_deterministic_equivalent, stratified_bundles)


def append(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def done(path, seed, part):
    if not os.path.exists(path):
        return False
    with open(path) as f:
        return any(r["seed"] == str(seed) and r["partition"] == part for r in csv.DictReader(f))


def read_de(path, seed):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["seed"] == str(seed):
                return r
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--terminals", type=int, default=5)
    ap.add_argument("--periods", type=int, default=5)
    ap.add_argument("--commodities", type=int, default=8)
    ap.add_argument("--scenarios", type=int, default=20)
    ap.add_argument("--bundles", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--kappa", type=float, default=0.0)
    ap.add_argument("--variant-a", action="store_true", help="spot capacity on any lane (default: variant B)")
    ap.add_argument("--partitions", type=str, default="random,costsig,kmeans,stratified")
    ap.add_argument("--max-bundle-size", type=int, default=-1, help="-1 = ceil(1.5 S/m), 0 = no cap")
    ap.add_argument("--solver", type=str, default="CBC")
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--de-time-limit", type=float, default=3600.0, help="0 = no limit")
    ap.add_argument("--skip-de", action="store_true")
    ap.add_argument("--bundle-time-limit", type=float, default=600.0, help="0 = no limit")
    ap.add_argument("--recovery-every", type=int, default=2)
    ap.add_argument("--ls-evals", type=int, default=12)
    ap.add_argument("--theta-patience", type=int, default=4)
    ap.add_argument("--lpdual", action="store_true", help="also offer +/-LP-dual starts to the guard")
    ap.add_argument("--out", type=str, default="results/study")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    S_csv, H_csv, D_csv = (os.path.join(args.out, f) for f in ("summary.csv", "history.csv", "de.csv"))
    tl = lambda v: None if not v else v
    solver = get_solver(args.solver, time_limit=tl(args.bundle_time_limit))
    lp_solver = get_solver(args.solver)
    de_solver = get_solver(args.solver, time_limit=tl(args.de_time_limit))
    cap = (math.ceil(1.5 * args.scenarios / args.bundles) if args.max_bundle_size < 0
           else (args.max_bundle_size or None))
    parts = [p.strip() for p in args.partitions.split(",") if p.strip()]

    for seed in args.seeds:
        inst = generate_instance(n_terminals=args.terminals, n_periods=args.periods,
                                 n_commodities=args.commodities, n_scenarios=args.scenarios,
                                 kappa=args.kappa, seed=seed,
                                 spot_requires_activation=not args.variant_a)
        print(f"\n##### seed {seed}: {inst.summary()}", flush=True)
        de = read_de(D_csv, seed)
        if de is None and not args.skip_de:
            r = solve_deterministic_equivalent(inst, de_solver)
            de = dict(seed=seed, instance=inst.name, de_obj=r.objective, de_bound=r.bound,
                      de_proved=int(r.proved_optimal), de_time=round(r.seconds, 2),
                      de_active=int(sum(r.y.values())), n_services=len(inst.service_ids))
            append(D_csv, de)
            de = {k: str(v) for k, v in de.items()}
        if de is not None:
            print(f"  DE: obj={float(de['de_obj']):.3f} proved={de['de_proved']} time={de['de_time']}s", flush=True)
        zstar = float(de["de_obj"]) if de is not None and de["de_proved"] == "1" else None

        makers = {
            "singleton": lambda: singleton_bundles(inst),
            "random": lambda: random_bundles(inst, args.bundles, seed=seed),
            "kmeans": lambda: kmeans_bundles(inst, args.bundles, seed=seed, max_size=cap),
            "costsig": lambda: cost_signature_bundles(inst, args.bundles, seed=seed, solver=lp_solver,
                                                      parallel=args.parallel, max_size=cap),
            "stratified": lambda: stratified_bundles(inst, args.bundles, seed=seed, solver=lp_solver,
                                                     parallel=args.parallel),
            "solsim": lambda: solution_similarity_bundles(inst, args.bundles, solver=solver, max_size=cap),
        }
        for part in parts:
            if done(S_csv, seed, part):
                print(f"  {part}: already done, skipped", flush=True)
                continue
            t0 = time.time()
            bundles = makers[part]()
            t_part = time.time() - t0
            ld = BundledLagrangian(inst, bundles, solver, step_rule="polyak", theta=1.0, tau=0.5,
                                   max_iter=args.iters, gap_tol=1e-3, parallel=args.parallel, verbose=False,
                                   ls_max_evals=args.ls_evals, recovery_every=args.recovery_every,
                                   recourse_solver=lp_solver, theta_patience=args.theta_patience)
            cands, names = [], []
            if args.lpdual:
                lp = lp_dual_multipliers(inst, bundles, lp_solver)
                cands, names = [lp, -lp], ["+LPdual", "-LPdual"]
            res = ld.run(candidates=cands, candidate_names=names)
            L0 = res.history[0]["L"]
            sizes = sorted(len(b) for b in bundles)
            row = dict(seed=seed, partition=part, m=len(bundles), bundle_sizes="-".join(map(str, sizes)),
                       zstar=zstar if zstar is not None else "", L0=round(L0, 4),
                       LB=round(res.best_lb, 4), UB=round(res.best_ub, 4), gap_pct=round(100 * res.gap, 4),
                       L0_gap_pct=round(100 * (zstar - L0) / zstar, 4) if zstar else "",
                       LB_gap_pct=round(100 * (zstar - res.best_lb) / zstar, 4) if zstar else "",
                       UB_gap_pct=round(100 * (res.best_ub - zstar) / zstar, 4) if zstar else "",
                       iters=res.iterations, bundle_solves=res.n_bundle_solves,
                       unproved_bundles=res.n_unproved_bundle_solves, designs=res.n_design_evals,
                       recourse_lps=res.n_recourse_solves, start=res.start_source,
                       partition_time=round(t_part, 2), ld_time=round(res.seconds, 2),
                       sec_per_iter=round(res.seconds / max(1, res.iterations), 2))
            if zstar is not None:
                assert res.best_lb <= zstar + 1e-6 * max(1, abs(zstar)), "LB > z*: bug"
                assert res.best_ub >= zstar - 1e-6 * max(1, abs(zstar)), "UB < z*: bug"
            append(S_csv, row)
            for h in res.history:
                append(H_csv, dict(seed=seed, partition=part, t=h["t"], L=round(h["L"], 4),
                                   LB=round(h["best_lb"], 4), UB=round(h["best_ub"], 4) if np.isfinite(h["best_ub"]) else "",
                                   theta=h.get("theta", ""), time=round(h["time"], 2)))
            print(f"  {part:<11} sizes={row['bundle_sizes']:<14} LB={res.best_lb:.2f} UB={res.best_ub:.2f} "
                  f"gap={100 * res.gap:.2f}%  {res.seconds:.0f}s", flush=True)
    print(f"\nresults in {args.out}/")


if __name__ == "__main__":
    main()
