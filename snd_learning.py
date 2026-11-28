"""Learning components on SND_V1 instances (IDSR benchmark), with the persistent Gurobi backend.

    # 1. bundling study: bound at zero multipliers L(0) for several partition rules
    python snd_learning.py bundling --instances <files> --out results/learn

    # 2. train the dual start policy on tuning instances; needs long LD runs made with
    #    run_snd.py --tag long (their rows give L(0) and LB* per instance)
    python snd_learning.py train --instances <files> --long-csv results/learn/snd.csv \
        --out models/snd_policy.pkl

The learned start is then evaluated with run_snd.py --start-policy models/snd_policy.pkl
(guarded, or --no-zero-guard), and the oracle start with run_snd.py --start-lambda <lam_long.npy>.

Partition rules of the bundling study (all with m = N / bundle_size bundles of equal size):
  random1..3     random partitions
  homog_feat     k-means on demand features, capacitated to equal sizes (similar scenarios together)
  homog_cost     k-means on cost signatures, capacitated to equal sizes
  strat_feat     stratified on demand features (each bundle spans the strata)
  strat_cost     stratified on cost signatures (the rule used by LD)
For every partition: L(0) (sum of bundle MILP bounds, cap per bundle as in LD), the number of bundle
problems not proved optimal, the time, the LP proxy (sum of bundle LP values) and partition features.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np

from ssnd_labld.bundling import (_kmeans, _labels_to_bundles, balance_labels, cost_signatures,
                                 random_bundles, scenario_features, standardize, stratified_bundles)
from ssnd_labld.grb_backend import GurobiBackend
from ssnd_labld.learning import (PART_FEATURES, DualStartPolicy, partition_features, policy_directions,
                                 save_models)
from ssnd_labld.snd_io import read_snd


def append(path, row):
    new = not os.path.exists(path)
    if not new:
        with open(path) as f:
            if f.readline().strip().split(",") != list(row.keys()):
                raise SystemExit(f"{path}: header differs; use a new --out folder")
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def done(path, key_cols):
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {tuple(r[k] for k in key_cols) for r in csv.DictReader(f)}


def stratified_from(X: np.ndarray, ids, m: int, seed: int):
    """Stratified (anti-clustered) partition from a standardised feature matrix (as stratified_bundles)."""
    q = max(1, min(int(np.ceil(len(X) / m)), len(X)))
    lab = _kmeans(X, q, seed)
    order = []
    for j in sorted(set(int(l) for l in lab)):
        idx = np.where(lab == j)[0]
        c = X[idx].mean(axis=0)
        order.extend(idx[np.argsort(((X[idx] - c) ** 2).sum(axis=1))].tolist())
    bundles = [[] for _ in range(m)]
    for pos, i in enumerate(order):
        bundles[pos % m].append(int(ids[i]))
    return [sorted(b) for b in bundles if b]


def homogeneous_from(X: np.ndarray, ids, m: int, seed: int):
    size = int(np.ceil(len(X) / m))
    return _labels_to_bundles(balance_labels(X, _kmeans(X, m, seed), size), ids)


# ------------------------------------------------------------------------------------------ bundling
def cmd_bundling(a):
    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, "bundling.csv")
    fin = done(out, ["instance", "partition"])
    for path in a.instances:
        inst = read_snd(path)
        name, N = inst.meta.get("label"), len(inst.scenarios)
        m = max(2, N // a.bundle_size)
        print(f"##### {name}: N={N} m={m}", flush=True)
        be = GurobiBackend(inst, parallel=a.parallel, bundle_time_limit=a.bundle_time_limit,
                           bundle_gap=1e-4, verbose=False)
        ids = [s.id for s in inst.scenarios]
        X_sig = standardize(cost_signatures(inst, seed=0, evaluator=be.evaluate_design))
        Phi = standardize(scenario_features(inst))
        parts = [(f"random{k}", random_bundles(inst, m, seed=1000 + k)) for k in (1, 2, 3)]
        parts += [("homog_feat", homogeneous_from(Phi, ids, m, 0)),
                  ("homog_cost", homogeneous_from(X_sig, ids, m, 0)),
                  ("strat_feat", stratified_bundles(inst, m, seed=0, basis="features")),
                  ("strat_cost", stratified_from(X_sig, ids, m, 0))]
        for pname, B in parts:
            if (name, pname) in fin:
                continue
            assert sorted(w for b in B for w in b) == sorted(ids), "not a partition"
            be.set_bundles(B)
            t0 = time.time()
            vals, _, proved, complete = be.solve_bundles(np.zeros((len(B), len(inst.service_ids))))
            secs = time.time() - t0
            lp_vals, _ = be.bundle_lp_designs()
            f = partition_features(inst, B, X_sig, Phi)
            row = dict(instance=name, N=N, partition=pname, m=len(B), L0=round(float(vals.sum()), 4),
                       unproved=int((~proved).sum()), complete=int(complete), seconds=round(secs, 1),
                       lp_proxy=round(float(lp_vals.sum()), 4))
            row.update({k: round(float(v), 6) for k, v in zip(PART_FEATURES, f)})
            append(out, row)
            print(f"  {pname:11s} L0={row['L0']:.2f}  unproved={row['unproved']}  {secs:.0f}s", flush=True)
        be.close()


# ------------------------------------------------------------------------------------------ train
def cmd_train(a):
    long = {}
    with open(a.long_csv) as f:
        for r in csv.DictReader(f):
            if r["method"] == a.long_method:
                long[r["instance"]] = (float(r["lb0"]), float(r["lb"]))
    cases, backends = [], []
    for path in a.instances:
        inst = read_snd(path)
        name, N = inst.meta.get("label"), len(inst.scenarios)
        if name not in long:
            raise SystemExit(f"no long run for {name} in {a.long_csv} (method {a.long_method})")
        m = max(2, N // a.bundle_size)
        be = GurobiBackend(inst, parallel=a.parallel, bundle_time_limit=a.bundle_time_limit,
                           bundle_gap=1e-4, verbose=False)
        B = stratified_bundles(inst, m, seed=0, basis="costsig", evaluator=be.evaluate_design)
        be.set_bundles(B)
        _, y_lp = be.bundle_lp_designs()
        L0_run, LBstar = long[name]
        # L(0) is re-evaluated with the training cap, so that L(0) and L(lambda(w)) are computed alike
        vals0, _, _, ok0 = be.solve_bundles(np.zeros((len(B), len(inst.service_ids))))
        L0 = float(vals0.sum()) if ok0 else L0_run
        LBstar = max(LBstar, L0)
        cases.append(dict(inst=inst, bundles=B, D=policy_directions(inst, B, y_lp), L0=L0, LBstar=LBstar,
                          be=be, name=name))
        backends.append(be)
        print(f"case {name}: L0={L0:.2f} (run {L0_run:.2f}) LB*={LBstar:.2f} headroom={LBstar - L0:.2f} m={m}",
              flush=True)

    def evaluate(case, lam):
        vals, _, _, complete = case["be"].solve_bundles(lam)
        return float(vals.sum()) if complete else -np.inf

    policy = DualStartPolicy()
    t0 = time.time()
    info = policy.fit(cases, evaluate, max_evals=a.max_evals, verbose=True, scale="bundle")
    info["seconds"] = round(time.time() - t0, 1)
    info["train_instances"] = [c["name"] for c in cases]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    save_models(a.out, policy, None, meta=dict(info=info, bundle_size=a.bundle_size))
    print(json.dumps(info, indent=1))
    for be in backends:
        be.close()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("bundling", "train"):
        p = sub.add_parser(c)
        p.add_argument("--instances", nargs="+", required=True)
        p.add_argument("--parallel", type=int, default=10)
        p.add_argument("--bundle-size", type=int, default=20)
        p.add_argument("--bundle-time-limit", type=float, default=300.0)
        p.add_argument("--out", required=True)
        if c == "train":
            p.add_argument("--long-csv", required=True)
            p.add_argument("--long-method", default="ld_long")
            p.add_argument("--max-evals", type=int, default=40)
    a = ap.parse_args()
    {"bundling": cmd_bundling, "train": cmd_train}[a.cmd](a)


if __name__ == "__main__":
    main()
