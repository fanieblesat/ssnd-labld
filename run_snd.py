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
The budget is a hard wall-clock limit for LD: each bundle MILP gets at most the remaining time, and an
iteration interrupted by the deadline is discarded (the bounds of the last complete iteration are kept).
  BD  multi-cut Benders (ssnd_labld/benders.py): Kelley rounds on the master LP, then branch-and-cut
      with lazy cuts; mean-demand block in the master unless --bd-no-ev; same budget, model building included.
Output: snd.csv (one row per instance and method), history.csv (LD trajectory), designs.json (the
design of each method, as a list of open arcs). lb0 is the LD bound at zero multipliers (the bundle /
group-subproblem bound). --tag labels a variant (method "ld_<tag>" / "de_<tag>"), so several settings
can share one output folder. Resumable: finished (instance, method) pairs are skipped.
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
from ssnd_labld.benders import solve_benders
from ssnd_labld.grb_backend import GurobiBackend, solve_de_gurobi
from ssnd_labld.snd_io import read_snd


def append(path, row):
    new = not os.path.exists(path)
    if not new:                                      # refuse to append under a different header
        with open(path) as f:
            header = f.readline().strip().split(",")
        if header != list(row.keys()):
            raise RuntimeError(f"{path}: header {header} differs from {list(row.keys())}; "
                               f"repair the file or use a new --out folder")
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


SND_COLS = ["instance", "N", "method", "lb", "lb0", "ub", "gap_pct", "proved", "seconds", "build_seconds",
            "iterations", "bundles", "unproved_bundles"]
HIST_COLS = ["instance", "method", "t", "L", "LB", "UB", "time"]


def check_header(path, cols):
    """Fail at startup (not after a 30-minute solve) if an existing CSV has another header."""
    if os.path.exists(path):
        with open(path) as f:
            header = f.readline().strip().split(",")
        if header != cols:
            raise SystemExit(f"{path}: header {header} differs from {cols}; repair it or use a new --out folder")


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
    ap.add_argument("--methods", nargs="+", default=["de", "ld"], choices=["de", "ld", "bd"])
    ap.add_argument("--time-limit", type=float, default=1800.0)
    ap.add_argument("--parallel", type=int, default=8, help="threads for the DE / workers for LD")
    ap.add_argument("--bundle-size", type=int, default=10)
    ap.add_argument("--partition", choices=["stratified", "random"], default="stratified")
    ap.add_argument("--basis", choices=["costsig", "features"], default="costsig")
    ap.add_argument("--bundle-time-limit", type=float, default=300.0)
    ap.add_argument("--bundle-gap", type=float, default=1e-6, help="MIP gap of the bundle MILPs (bound stays valid)")
    ap.add_argument("--theta", type=float, default=1.0, help="Polyak factor")
    ap.add_argument("--bd-no-ev", action="store_true", help="Benders without the mean-demand block")
    ap.add_argument("--bd-phase1-share", type=float, default=0.25, help="budget share of the LP cut rounds")
    ap.add_argument("--save-lambda", default=None, help="save LD's best multipliers to this .npy file")
    ap.add_argument("--start-lambda", default=None, help="start LD from multipliers in this .npy file")
    ap.add_argument("--no-zero-guard", action="store_true", help="with --start-lambda/--start-policy: skip the zero start")
    ap.add_argument("--start-policy", default=None, help="start LD from a learned DualStartPolicy (.pkl)")
    ap.add_argument("--theta-warm", type=float, default=None, help="Polyak factor after an accepted non-zero start")
    ap.add_argument("--tag", default="", help="variant label appended to the method name")
    ap.add_argument("--recovery-every", type=int, default=3)
    ap.add_argument("--ls-evals", type=int, default=8)
    ap.add_argument("--no-strong-linking", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.out, "snd_args.json"), "w"), indent=1)
    out, hist, des = (os.path.join(args.out, f) for f in ("snd.csv", "history.csv", "designs.json"))
    check_header(out, SND_COLS)
    check_header(hist, HIST_COLS)
    finished = done(out)
    strong = not args.no_strong_linking
    sfx = f"_{args.tag}" if args.tag else ""
    m_de, m_ld, m_bd = "de" + sfx, "ld" + sfx, "bd" + sfx

    for path in args.instances:
        inst = read_snd(path)
        # the label is unique across configurations (file names repeat between main/ and scaling/)
        name, N = (inst.meta.get("label") or os.path.basename(path)), len(inst.scenarios)
        print(f"\n##### {name}: |V|={len(inst.nodes)} |A|={len(inst.arcs)} |K|={len(inst.commodities)} N={N}",
              flush=True)

        if "de" in args.methods and (name, m_de) not in finished:
            r = solve_de_gurobi(inst, time_limit=args.time_limit, threads=args.parallel, strong_linking=strong)
            gap = 100 * (r.objective - r.bound) / abs(r.objective) if np.isfinite(r.objective) else ""
            append(out, dict(instance=name, N=N, method=m_de, lb=round(r.bound, 4), lb0="",
                             ub=round(r.objective, 4) if np.isfinite(r.objective) else "",
                             gap_pct=round(gap, 4) if gap != "" else "", proved=int(r.proved_optimal),
                             seconds=round(r.seconds, 1), build_seconds=round(r.build_seconds, 1),
                             iterations="", bundles="", unproved_bundles=""))
            if np.isfinite(r.objective):
                save_design(des, name, m_de, r.y)
            print(f"  {m_de.upper()}: UB={r.objective:.2f} LB={r.bound:.2f} proved={r.proved_optimal} "
                  f"({r.seconds:.0f}s, build {r.build_seconds:.0f}s)", flush=True)

        if "ld" in args.methods and (name, m_ld) not in finished:
            t0 = time.time()
            m = max(2, N // args.bundle_size)
            be = GurobiBackend(inst, parallel=args.parallel, bundle_time_limit=args.bundle_time_limit,
                               bundle_gap=args.bundle_gap, strong_linking=strong, verbose=True)
            B = (stratified_bundles(inst, m, seed=args.seed, basis=args.basis, evaluator=be.evaluate_design)
                 if args.partition == "stratified" else random_bundles(inst, m, seed=args.seed))
            be.set_bundles(B)
            cands, names = None, None
            if args.start_lambda:
                cands, names = [np.load(args.start_lambda)], ["given"]
            elif args.start_policy:                   # the policy's LP features count in the budget (t_prep)
                from ssnd_labld.learning import load_models
                policy, _, _ = load_models(args.start_policy)
                _, y_lp = be.bundle_lp_designs()
                cands, names = [policy.multipliers(inst, B, y_lp=y_lp)], ["learned"]
            t_prep = time.time() - t0
            ld = BundledLagrangian(inst, B, step_rule="polyak", theta=args.theta, tau=0.5, max_iter=100000,
                                   time_limit=max(1.0, args.time_limit - t_prep), gap_tol=1e-4,
                                   parallel=args.parallel, verbose=False, ls_max_evals=args.ls_evals,
                                   recovery_every=args.recovery_every, backend=be, theta_warm=args.theta_warm)
            res = ld.run(candidates=cands, candidate_names=names, include_zero=not args.no_zero_guard)
            if args.save_lambda:
                np.save(args.save_lambda, res.lam_final)
            for h in res.history:
                append(hist, dict(instance=name, method=m_ld, t=h["t"], L=round(h["L"], 4), LB=round(h["best_lb"], 4),
                                  UB=round(h["best_ub"], 4) if np.isfinite(h["best_ub"]) else "",
                                  time=round(h["time"] + t_prep, 2)))
            append(out, dict(instance=name, N=N, method=m_ld, lb=round(res.best_lb, 4),
                             lb0=round(res.history[0]["best_lb"], 4),
                             ub=round(res.best_ub, 4) if np.isfinite(res.best_ub) else "",
                             gap_pct=round(100 * res.gap, 4) if np.isfinite(res.gap) else "",
                             proved=int(res.gap <= 1e-4), seconds=round(res.seconds + t_prep, 1),
                             build_seconds=round(t_prep, 1), iterations=res.iterations, bundles=m,
                             unproved_bundles=res.n_unproved_bundle_solves))
            if res.best_y is not None:
                save_design(des, name, m_ld, res.best_y)
            print(f"  {m_ld.upper()}: LB0={res.history[0]['best_lb']:.2f} LB={res.best_lb:.2f} UB={res.best_ub:.2f} gap={100 * res.gap:.3f}% "
                  f"iters={res.iterations} m={m} prep={t_prep:.0f}s total={res.seconds + t_prep:.0f}s", flush=True)
            be.close()

        if "bd" in args.methods and (name, m_bd) not in finished:
            r = solve_benders(inst, time_limit=args.time_limit, threads=args.parallel, strong_linking=strong,
                              ev_block=not args.bd_no_ev, phase1_share=args.bd_phase1_share)
            for h in r.history:
                append(hist, dict(instance=name, method=m_bd, t=h["round"], L="", LB=round(h["lb"], 4),
                                  UB=round(h["ub"], 4) if np.isfinite(h["ub"]) else "", time=round(h["time"], 2)))
            append(out, dict(instance=name, N=N, method=m_bd, lb=round(r.lb, 4), lb0=round(r.lp_bound, 4),
                             ub=round(r.ub, 4) if np.isfinite(r.ub) else "",
                             gap_pct=round(100 * r.gap, 4) if np.isfinite(r.gap) else "",
                             proved=int(r.proved_optimal), seconds=round(r.seconds, 1), build_seconds="",
                             iterations=r.phase1_rounds, bundles=r.candidates, unproved_bundles=r.cuts))
            if r.y is not None:
                save_design(des, name, m_bd, r.y)
            print(f"  {m_bd.upper()}: LP={r.lp_bound:.2f} LB={r.lb:.2f} UB={r.ub:.2f} gap={100 * r.gap:.3f}% "
                  f"rounds={r.phase1_rounds} candidates={r.candidates} cuts={r.cuts} ({r.seconds:.0f}s)", flush=True)
    print(f"\nresults in {args.out}/")


if __name__ == "__main__":
    main()
