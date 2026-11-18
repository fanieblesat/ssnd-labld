"""Pilot on SND_V1 instances (IDSR benchmark): Gurobi on the extensive form vs bundled Lagrangian
decomposition, with the same wall-clock budget on the same machine.

    python run_snd.py --instances ../idsr/instances/snd/scaling/snd_10_60_10_500_seed1.txt \
        --time-limit 1800 --parallel 10 --bundle-size 10 --out results/snd_pilot

For every instance:
  DE  gurobipy extensive form (MIP gap 1e-4, as in the IDSR protocol), time limit including model
      building; records the incumbent (UB), Gurobi's proven bound (LB) and whether it proved optimality.
  LD  stratified (or random) partition into bundles of --bundle-size scenarios, persistent Gurobi
      bundle MILPs and recourse LPs (grb_backend), zero start, Polyak steps; the budget covers the
      partition, model building and the iterations. Bundle values are Gurobi bounds, so LB is valid.
Output: snd.csv (one row per instance and method), history.csv (LD trajectory), designs.json (the
design of each method, as a list of open arcs). Resumable: finished (instance, method) pairs are skipped.
Run each instance in a fresh process if memory is tight (--methods de, then --methods ld).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np

from ssnd_labld import BundledLagrangian, random_bundles, stratified_bundles
from ssnd_labld.grb_backend import GurobiBackend, solve_de_gurobi
from ssnd_labld.snd_io import read_snd


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
        return {(r["instance"], r["method"]) for r in csv.DictReader(f)}


def save_design(path, name, method, y):
    data = json.load(open(path)) if os.path.exists(path) else {}
    data.setdefault(name, {})[method] = sorted(int(a) for a, v in y.items() if v > 0.5)
    json.dump(data, open(path, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", nargs="+", required=True)
    ap.add_argument("--methods", nargs="+", default=["de", "ld"], choices=["de", "ld"])
    ap.add_argument("--time-limit", type=float, default=1800.0)
    ap.add_argument("--parallel", type=int, default=8, help="threads for the DE / workers for LD")
    ap.add_argument("--bundle-size", type=int, default=10)
    ap.add_argument("--partition", choices=["stratified", "random"], default="stratified")
    ap.add_argument("--basis", choices=["costsig", "features"], default="costsig")
    ap.add_argument("--bundle-time-limit", type=float, default=300.0)
    ap.add_argument("--recovery-every", type=int, default=3)
    ap.add_argument("--ls-evals", type=int, default=8)
    ap.add_argument("--no-strong-linking", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.out, "snd_args.json"), "w"), indent=1)
    out, hist, des = (os.path.join(args.out, f) for f in ("snd.csv", "history.csv", "designs.json"))
    finished = done(out)
    strong = not args.no_strong_linking

    for path in args.instances:
        inst = read_snd(path)
        name, N = os.path.basename(path), len(inst.scenarios)
        print(f"\n##### {name}: |V|={len(inst.nodes)} |A|={len(inst.arcs)} |K|={len(inst.commodities)} N={N}",
              flush=True)

        if "de" in args.methods and (name, "de") not in finished:
            r = solve_de_gurobi(inst, time_limit=args.time_limit, threads=args.parallel, strong_linking=strong)
            gap = 100 * (r.objective - r.bound) / abs(r.objective) if np.isfinite(r.objective) else ""
            append(out, dict(instance=name, N=N, method="de", lb=round(r.bound, 4),
                             ub=round(r.objective, 4) if np.isfinite(r.objective) else "",
                             gap_pct=round(gap, 4) if gap != "" else "", proved=int(r.proved_optimal),
                             seconds=round(r.seconds, 1), build_seconds=round(r.build_seconds, 1),
                             iterations="", bundles="", unproved_bundles=""))
            if np.isfinite(r.objective):
                save_design(des, name, "de", r.y)
            print(f"  DE: UB={r.objective:.2f} LB={r.bound:.2f} proved={r.proved_optimal} "
                  f"({r.seconds:.0f}s, build {r.build_seconds:.0f}s)", flush=True)

        if "ld" in args.methods and (name, "ld") not in finished:
            t0 = time.time()
            m = max(2, N // args.bundle_size)
            be = GurobiBackend(inst, parallel=args.parallel, bundle_time_limit=args.bundle_time_limit,
                               strong_linking=strong, verbose=True)
            B = (stratified_bundles(inst, m, seed=args.seed, basis=args.basis, evaluator=be.evaluate_design)
                 if args.partition == "stratified" else random_bundles(inst, m, seed=args.seed))
            be.set_bundles(B)
            t_prep = time.time() - t0
            ld = BundledLagrangian(inst, B, step_rule="polyak", theta=1.0, tau=0.5, max_iter=100000,
                                   time_limit=max(1.0, args.time_limit - t_prep), gap_tol=1e-4,
                                   parallel=args.parallel, verbose=False, ls_max_evals=args.ls_evals,
                                   recovery_every=args.recovery_every, backend=be)
            res = ld.run()
            for h in res.history:
                append(hist, dict(instance=name, t=h["t"], L=round(h["L"], 4), LB=round(h["best_lb"], 4),
                                  UB=round(h["best_ub"], 4) if np.isfinite(h["best_ub"]) else "",
                                  time=round(h["time"] + t_prep, 2)))
            append(out, dict(instance=name, N=N, method="ld", lb=round(res.best_lb, 4),
                             ub=round(res.best_ub, 4) if np.isfinite(res.best_ub) else "",
                             gap_pct=round(100 * res.gap, 4) if np.isfinite(res.gap) else "",
                             proved=int(res.gap <= 1e-4), seconds=round(res.seconds + t_prep, 1),
                             build_seconds=round(t_prep, 1), iterations=res.iterations, bundles=m,
                             unproved_bundles=res.n_unproved_bundle_solves))
            if res.best_y is not None:
                save_design(des, name, "ld", res.best_y)
            print(f"  LD: LB={res.best_lb:.2f} UB={res.best_ub:.2f} gap={100 * res.gap:.3f}% "
                  f"iters={res.iterations} m={m} prep={t_prep:.0f}s total={res.seconds + t_prep:.0f}s", flush=True)
            be.close()
    print(f"\nresults in {args.out}/")


if __name__ == "__main__":
    main()
