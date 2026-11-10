"""Scaling study: deterministic equivalent (DE) versus bundled Lagrangian decomposition (LD)
under the same wall-clock budget, for an increasing number of scenarios.

    python run_scaling.py --scenarios 60 120 240 --seeds 1 2 --terminals 6 --periods 6 \
        --commodities 10 --bundle-size 5 --time-limit 1800 --solver HiGHS --parallel 6 \
        --out results/scaling

For every (seed, |Omega|):
  DE  HiGHS on the extensive form with --time-limit; records the incumbent (UB), the proven lower
      bound (HiGHS MIP dual bound, else LP relaxation) and whether optimality was proved.
  LD  stratified partition with m = |Omega| / bundle-size bundles, zero start (or a learned start,
      --models), Polyak steps, until --time-limit minus --polish-time; then primal polishing: the DE
      restricted to the services opened in any bundle solution, for the rest of the budget.
      Records the best LB/UB over time and the UB before and after polishing.
Both use all --parallel threads. Output: scaling.csv (one row per seed and size),
history.csv (LD trajectory). Resumable: finished (seed, |Omega|) pairs are skipped.
The point where the DE stops proving optimality within the budget, and the LD gap there,
is the result this study is for.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np

from ssnd_labld import (BundledLagrangian, generate_instance, get_solver, random_bundles,
                        solve_deterministic_equivalent, stratified_bundles)
from ssnd_labld.models import solve_restricted_de
from ssnd_labld.learning import DualStartPolicy, load_models


def append(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def done(path):
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {(r["seed"], r["S"]) for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", type=int, nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--terminals", type=int, default=6)
    ap.add_argument("--periods", type=int, default=6)
    ap.add_argument("--commodities", type=int, default=10)
    ap.add_argument("--kappa", type=float, default=0.0)
    ap.add_argument("--variant-a", action="store_true")
    ap.add_argument("--bundle-size", type=int, default=5, help="m = |Omega| / bundle-size")
    ap.add_argument("--partition", choices=["stratified", "random"], default="stratified")
    ap.add_argument("--time-limit", type=float, default=1800.0, help="wall-clock budget for DE and for LD")
    ap.add_argument("--bundle-time-limit", type=float, default=300.0)
    ap.add_argument("--solver", type=str, default="HiGHS")
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--recovery-every", type=int, default=3)
    ap.add_argument("--ls-evals", type=int, default=8)
    ap.add_argument("--checkpoints", type=float, nargs="+", default=[60, 300, 600, 1200, 1800, 3600])
    ap.add_argument("--theta-warm", type=float, default=0.0,
                    help="initial Polyak factor after an accepted learned start (0 = 1, as for the zero start)")
    ap.add_argument("--models", type=str, default="", help="optional: learned warm start (models .pkl)")
    ap.add_argument("--skip-de", action="store_true")
    ap.add_argument("--polish-time", type=float, default=300.0,
                    help="seconds of the budget reserved for primal polishing (restricted DE); 0 = off")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.out, "scaling_args.json"), "w"), indent=1)
    S_csv, H_csv = os.path.join(args.out, "scaling.csv"), os.path.join(args.out, "history.csv")
    finished = done(S_csv)
    policy = None
    if args.models:
        mult, _, _ = load_models(args.models)
        policy = mult if isinstance(mult, DualStartPolicy) else None

    for S in args.scenarios:
        for seed in args.seeds:
            if (str(seed), str(S)) in finished:
                continue
            inst = generate_instance(n_terminals=args.terminals, n_periods=args.periods,
                                     n_commodities=args.commodities, n_scenarios=S, kappa=args.kappa,
                                     seed=seed, spot_requires_activation=not args.variant_a)
            m = max(2, S // args.bundle_size)
            print(f"\n##### seed {seed}  |Omega|={S}  m={m}: {inst.summary()}", flush=True)

            # ---------------- deterministic equivalent under the budget
            de = dict(de_ub="", de_lb="", de_proved="", de_time="", de_gap_pct="")
            if not args.skip_de:
                r = solve_deterministic_equivalent(inst, get_solver(args.solver, time_limit=args.time_limit,
                                                                    threads=args.parallel))
                gap = (100 * (r.objective - r.bound) / abs(r.objective)
                       if np.isfinite(r.objective) and r.bound is not None else float("inf"))
                de = dict(de_ub=round(r.objective, 4) if np.isfinite(r.objective) else "",
                          de_lb=round(r.bound, 4) if r.bound is not None else "",
                          de_proved=int(r.proved_optimal), de_time=round(r.seconds, 1),
                          de_gap_pct=round(gap, 4) if np.isfinite(gap) else "")
                print(f"  DE: UB={de['de_ub']} LB={de['de_lb']} proved={r.proved_optimal} "
                      f"gap={de['de_gap_pct']}%  {r.seconds:.0f}s", flush=True)

            # ---------------- decomposition under the same budget
            lp_solver = get_solver(args.solver)
            t0 = time.time()
            B = (stratified_bundles(inst, m, seed=seed, solver=lp_solver, parallel=args.parallel)
                 if args.partition == "stratified" else random_bundles(inst, m, seed=seed))
            t_part = time.time() - t0
            cache = {}
            ld = BundledLagrangian(inst, B, get_solver(args.solver, time_limit=args.bundle_time_limit),
                                   step_rule="polyak", theta=1.0, tau=0.5, max_iter=100000,
                                   time_limit=max(1.0, args.time_limit - t_part - args.polish_time),
                                   gap_tol=1e-3,
                                   parallel=args.parallel, verbose=False, ls_max_evals=args.ls_evals,
                                   recovery_every=args.recovery_every, recourse_solver=lp_solver,
                                   eval_cache=cache, theta_warm=args.theta_warm or None)
            cands, names = [], []
            if policy is not None:
                cands, names = [policy.multipliers(inst, B, solver=lp_solver)], ["ml"]
            res = ld.run(cands, names)
            for h in res.history:
                append(H_csv, dict(seed=seed, S=S, t=h["t"], L=round(h["L"], 4), LB=round(h["best_lb"], 4),
                                   UB=round(h["best_ub"], 4) if np.isfinite(h["best_ub"]) else "",
                                   time=round(h["time"] + t_part, 2)))
            # ---------------- primal polishing: DE restricted to the services the bundles opened
            ub, gap_ld = res.best_ub, res.gap
            pol = dict(polish_ub="", polish_time="", polish_services="", polish_proved="")
            elapsed = res.seconds + t_part
            if args.polish_time > 0 and elapsed < args.time_limit - 5:
                allowed = {a for j, a in enumerate(inst.service_ids) if ld.open_count[j] > 0}
                if res.best_y:
                    allowed |= {a for a, v in res.best_y.items() if v > 0.5}
                r_p = solve_restricted_de(inst, allowed, get_solver(args.solver, threads=args.parallel,
                                                                   time_limit=args.time_limit - elapsed))
                p_ub = r_p.objective
                if np.isfinite(p_ub):        # exact cost of the polished design (all scenarios)
                    p_ub = min(p_ub, ld._eval_design(r_p.y).expected_total_cost)
                pol = dict(polish_ub=round(p_ub, 4) if np.isfinite(p_ub) else "",
                           polish_time=round(time.time() - t0 - elapsed, 1),
                           polish_services=len(allowed), polish_proved=int(r_p.proved_optimal))
                if p_ub < ub:
                    ub = p_ub
                gap_ld = (ub - res.best_lb) / abs(ub) if np.isfinite(ub) else float("inf")
                print(f"  polish: {len(allowed)}/{len(inst.service_ids)} services, UB={p_ub:.2f} "
                      f"proved={r_p.proved_optimal}  {pol['polish_time']}s", flush=True)
            row = dict(seed=seed, S=S, m=m, n_services=len(inst.service_ids), **de,
                       ld_lb=round(res.best_lb, 4),
                       ld_ub_before_polish=round(res.best_ub, 4) if np.isfinite(res.best_ub) else "",
                       ld_ub=round(ub, 4) if np.isfinite(ub) else "",
                       ld_gap_pct=round(100 * gap_ld, 4) if np.isfinite(gap_ld) else "",
                       **pol,
                       ld_iters=res.iterations, ld_time=round(time.time() - t0, 1),
                       partition_time=round(t_part, 1), start=res.start_source, theta_warm=args.theta_warm,
                       unproved_bundles=res.n_unproved_bundle_solves)
            # best LB / UB reached by each checkpoint (LD)
            for T in args.checkpoints:
                hh = [h for h in res.history if h["time"] + t_part <= T]
                row[f"ld_lb_{int(T)}s"] = round(max(h["best_lb"] for h in hh), 4) if hh else ""
            append(S_csv, row)
            print(f"  LD: LB={res.best_lb:.2f} UB={ub:.2f} gap={100 * gap_ld:.2f}% "
                  f"iters={res.iterations} unproved={res.n_unproved_bundle_solves}  "
                  f"{time.time() - t0:.0f}s total", flush=True)
    print(f"\nresults in {args.out}/")


if __name__ == "__main__":
    main()
