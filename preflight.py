"""Pre-flight check at full scale, before a long evaluation: every method must stay inside its budget and
return consistent bounds. Uses a tuning-seed instance (never an evaluation instance).

    python preflight.py --instance <file> --budget 300

Checks, per method (DE, LD with the tuned settings, Benders without the mean-demand block):
  1. wall-clock time <= budget + slack (slack = 2 % of the budget, at least 10 s)
  2. LB <= UB (own bounds)
  3. across methods: every LB <= every UB (all bounds are valid for the same instance)
Prints PASS or FAIL per check and exits with status 1 on any failure.
"""
import argparse
import sys
import time

import numpy as np

from ssnd_labld import BundledLagrangian, stratified_bundles
from ssnd_labld.benders import solve_benders
from ssnd_labld.grb_backend import GurobiBackend, solve_de_gurobi
from ssnd_labld.snd_io import read_snd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True)
    ap.add_argument("--budget", type=float, default=300.0)
    ap.add_argument("--parallel", type=int, default=10)
    ap.add_argument("--bundle-size", type=int, default=20)
    a = ap.parse_args()
    inst = read_snd(a.instance)
    N, B = len(inst.scenarios), a.budget
    slack = max(10.0, 0.02 * B)
    res, ok = {}, True
    print(f"instance {inst.meta.get('label')}  N={N}  budget {B:.0f}s  slack {slack:.0f}s", flush=True)

    t = time.time()
    r = solve_de_gurobi(inst, time_limit=B, threads=a.parallel)
    res["DE"] = (r.bound, r.objective, time.time() - t)

    t = time.time()
    be = GurobiBackend(inst, parallel=a.parallel, bundle_time_limit=300.0, bundle_gap=1e-4)
    m = max(2, N // a.bundle_size)
    bundles = stratified_bundles(inst, m, seed=0, basis="costsig", evaluator=be.evaluate_design)
    be.set_bundles(bundles)
    prep = time.time() - t
    ld = BundledLagrangian(inst, bundles, step_rule="polyak", theta=1.0, tau=0.5, max_iter=100000,
                           time_limit=max(1.0, B - prep), gap_tol=1e-4, parallel=a.parallel,
                           ls_max_evals=8, recovery_every=3, backend=be)
    r = ld.run()
    res["LD"] = (r.best_lb, r.best_ub, time.time() - t)
    be.close()

    t = time.time()
    r = solve_benders(inst, time_limit=B, threads=a.parallel, ev_block=False)
    res["BD"] = (r.lb, r.ub, time.time() - t)

    for k, (lb, ub, sec) in res.items():
        c1 = sec <= B + slack
        c2 = (not np.isfinite(ub)) or lb <= ub + 1e-6 * abs(ub)
        ok &= c1 and c2
        print(f"  {k}: LB={lb:.2f} UB={ub:.2f} time={sec:.0f}s  "
              f"[{'PASS' if c1 else 'FAIL'} budget] [{'PASS' if c2 else 'FAIL'} LB<=UB]", flush=True)
    max_lb = max(v[0] for v in res.values())
    min_ub = min(v[1] for v in res.values())
    c3 = max_lb <= min_ub + 1e-6 * abs(min_ub)
    ok &= c3
    print(f"  cross-method: max LB {max_lb:.2f} <= min UB {min_ub:.2f}  [{'PASS' if c3 else 'FAIL'}]")
    print("PREFLIGHT " + ("PASSED" if ok else "FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
