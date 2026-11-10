"""Sustainability and resilience study (RQ5).

For every seed, disruption setting and carbon price kappa:
  1. generate one network with |Omega_train| + |Omega_test| scenarios and split them, so the design is
     planned on the training scenarios and evaluated out of sample on independent test scenarios
     of the same network (the first |Omega_train| scenarios are identical to Study B for the same seed);
  2. solve the deterministic equivalent on the training scenarios (time limit);
  3. evaluate the design on the test scenarios: operating cost without the carbon charge, CO2e,
     emissions from emergency capacity, service reliability, emergency capacity, unserved demand,
     tonne-km shares by mode, and services opened by mode.

    python run_sustainability.py --seeds 1 2 3 4 5 --kappas 0 0.5 1 2 4 --settings base stressed \
        --solver HiGHS --parallel 6 --out results/sustainability

Resumable: finished (seed, setting, kappa) rows are skipped.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np

from ssnd_labld import generate_instance, get_solver, solve_deterministic_equivalent
from ssnd_labld.generator import REGIME_PROBS, REGIME_PROBS_STRESSED
from ssnd_labld.models import solve_recourse

MODES = ("road", "rail", "barge")


def split(inst, n_train):
    tr, te = inst.scenarios[:n_train], inst.scenarios[n_train:]
    renorm = lambda L: [replace(s, prob=1.0 / len(L)) for s in L]
    return (replace(inst, scenarios=renorm(tr), name=inst.name + "_train"),
            replace(inst, scenarios=renorm(te), name=inst.name + "_test"))


def evaluate_out_of_sample(inst, y, solver, parallel):
    with ThreadPoolExecutor(max_workers=parallel) as ex:
        per = list(ex.map(lambda s: solve_recourse(inst, y, s, solver, keep_flows=True), inst.scenarios))
    arcs = {a.id: a for a in inst.arcs}
    p = np.array([s.prob for s in inst.scenarios])
    fixed = sum(a.F * y[a.id] for a in inst.service_arcs)
    tkm = defaultdict(float)
    spot_em = 0.0
    for w, r in zip(p, per):
        for (a, k), f in r.flows.items():
            if arcs[a].is_service:
                tkm[arcs[a].mode] += w * f * arcs[a].dist
        spot_em += w * sum(arcs[a].e_r * v for a, v in r.spot_by_arc.items())
    em = float(p @ [r.emissions for r in per])
    total = fixed + float(p @ [r.cost for r in per])
    dem = float(p @ [r.demand for r in per])
    short = float(p @ [r.shortage for r in per])
    tk = sum(tkm.values()) or 1.0
    opened = {m: sum(1 for a in inst.service_arcs if a.mode == m and y[a.id] > 0.5) for m in MODES}
    regimes = sorted({s.regime for s in inst.scenarios})
    rel_by_regime = {}
    for g in ("nominal", "severe"):
        idx = [i for i, s in enumerate(inst.scenarios) if s.regime == g]
        if idx:
            d = sum(per[i].demand for i in idx)
            rel_by_regime[g] = 1.0 - sum(per[i].shortage for i in idx) / d if d > 0 else ""
    return dict(
        total_cost=total, operating_cost=total - inst.kappa * em, fixed_cost=fixed,
        emissions=em, spot_emissions=spot_em, reliability=1.0 - short / dem if dem > 0 else 1.0,
        reliability_nominal=rel_by_regime.get("nominal", ""), reliability_severe=rel_by_regime.get("severe", ""),
        spot=float(p @ [r.spot for r in per]), unserved_share=short / dem if dem > 0 else 0.0,
        **{f"share_{m}": tkm[m] / tk for m in MODES}, **{f"open_{m}": opened[m] for m in MODES},
        tonne_km=sum(tkm.values()), n_test=len(inst.scenarios))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--kappas", type=float, nargs="+", default=[0, 0.5, 1, 2, 4])
    ap.add_argument("--settings", nargs="+", default=["base", "stressed"], choices=["base", "stressed"])
    ap.add_argument("--terminals", type=int, default=6)
    ap.add_argument("--periods", type=int, default=6)
    ap.add_argument("--commodities", type=int, default=10)
    ap.add_argument("--train-scenarios", type=int, default=30)
    ap.add_argument("--test-scenarios", type=int, default=100)
    ap.add_argument("--spot-emission-factor", type=float, default=1.0,
                    help="emergency capacity emits this multiple of road CO2e per unit and distance")
    ap.add_argument("--time-limit", type=float, default=900.0, help="DE time limit (s)")
    ap.add_argument("--solver", type=str, default="HiGHS")
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--variant-a", action="store_true")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.out, "sustainability_args.json"), "w"), indent=1)
    out = os.path.join(args.out, "sustainability.csv")
    done = set()
    if os.path.exists(out):
        with open(out) as f:
            done = {(r["seed"], r["setting"], r["kappa"]) for r in csv.DictReader(f)}
    lp = get_solver(args.solver)
    for setting in args.settings:
        probs = REGIME_PROBS if setting == "base" else REGIME_PROBS_STRESSED
        for seed in args.seeds:
            for kappa in args.kappas:
                if (str(seed), setting, str(kappa)) in done:
                    continue
                full = generate_instance(n_terminals=args.terminals, n_periods=args.periods,
                                         n_commodities=args.commodities,
                                         n_scenarios=args.train_scenarios + args.test_scenarios,
                                         kappa=kappa, seed=seed, regime_probs=probs,
                                         spot_emission_factor=args.spot_emission_factor,
                                         spot_requires_activation=not args.variant_a)
                train, test = split(full, args.train_scenarios)
                t0 = time.time()
                de = solve_deterministic_equivalent(
                    train, get_solver(args.solver, time_limit=args.time_limit, threads=args.parallel))
                gap = 100 * (de.objective - de.bound) / abs(de.objective) if de.bound is not None else ""
                ev = evaluate_out_of_sample(test, de.y, lp, args.parallel)
                row = dict(seed=seed, setting=setting, kappa=kappa, de_obj=round(de.objective, 4),
                           de_proved=int(de.proved_optimal), de_gap_pct=round(gap, 4) if gap != "" else "",
                           de_time=round(de.seconds, 1),
                           **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in ev.items()},
                           eval_time=round(time.time() - t0 - de.seconds, 1))
                new = not os.path.exists(out)
                with open(out, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(row.keys()))
                    if new:
                        w.writeheader()
                    w.writerow(row)
                print(f"{setting:8s} seed {seed} kappa {kappa:<4}: CO2e={ev['emissions']:.1f} "
                      f"op.cost={ev['operating_cost']:.1f} rel={ev['reliability']:.4f} "
                      f"shares road/rail/barge={ev['share_road']:.2f}/{ev['share_rail']:.2f}/{ev['share_barge']:.2f} "
                      f"(DE {de.seconds:.0f}s, proved={de.proved_optimal})", flush=True)
    print(f"\nresults in {out}")


if __name__ == "__main__":
    main()
