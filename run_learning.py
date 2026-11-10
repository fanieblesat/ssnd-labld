"""Learning components: collect training data, train, evaluate.

    # 1. training data on small instances (separate seeds from the test seeds)
    python run_learning.py collect --seeds 101-140 --terminals 4 --periods 4 --commodities 6 \
        --scenarios 12 --bundles 4 --solver HiGHS --out data/train_small

    # 2. train both models (grouped cross-validation is printed)
    python run_learning.py train --data data/train_small --out models/small.pkl

    # 3. evaluate on the Study A instances (z* re-used from Study A's de.csv)
    python run_learning.py eval --models models/small.pkl --seeds 1-5 --terminals 5 --periods 5 \
        --commodities 8 --scenarios 20 --bundles 5 --solver HiGHS --de-csv results/studyA/de.csv \
        --out results/learnA

collect   per instance: a pool of candidate partitions with L(0) (and the LP proxy), and, for the
          partitions in --mult-parts, a long Lagrangian run whose best multiplier lambda* is the
          training target of the warm-start model. Resumable (finished seeds are skipped).
train     fits MultiplierPredictor and PartitionScorer; prints grouped-CV R^2 / Spearman / regret.
eval      warm start: zero vs learned (vs LP-dual, vs oracle lambda*) on the same partition;
          partition choice: random, stratified, learned selection, LP-proxy selection.
          Writes warmstart.csv, bundling.csv, history.csv. Resumable.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time

import numpy as np

from ssnd_labld import (BundledLagrangian, generate_instance, get_solver, lp_dual_multipliers,
                        random_bundles, stratified_bundles)
from ssnd_labld.lagrangian import oracle_headroom_capture, time_to_target
from ssnd_labld.learning import (PART_FEATURES, DualStartPolicy, rescale_policy, MultiplierPredictor, PartitionScorer, WarmTransfer,
                                 bundle_lp_designs, policy_directions,
                                 candidate_partitions, instance_signature_data, load_models,
                                 lp_proxy_scores, multiplier_features, multiplier_targets,
                                 partition_features, save_models)


# --------------------------------------------------------------------------- helpers
def parse_seeds(tokens):
    out = []
    for t in tokens:
        for part in str(t).split(","):
            if "-" in part:
                a, b = part.split("-")
                out.extend(range(int(a), int(b) + 1))
            elif part:
                out.append(int(part))
    return out


def append(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def rows(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def make_instance(args, seed):
    return generate_instance(n_terminals=args.terminals, n_periods=args.periods,
                             n_commodities=args.commodities, n_scenarios=args.scenarios,
                             kappa=args.kappa, seed=seed, spot_requires_activation=not args.variant_a)


def make_ld(args, inst, bundles, solver, lp_solver, max_iter, cache):
    return BundledLagrangian(inst, bundles, solver, step_rule="polyak", theta=1.0, tau=0.5,
                             max_iter=max_iter, gap_tol=1e-3, parallel=args.parallel, verbose=False,
                             ls_max_evals=args.ls_evals, recovery_every=args.recovery_every,
                             recourse_solver=lp_solver, theta_patience=args.theta_patience,
                             eval_cache=cache, theta_warm=getattr(args, "theta_warm", 0.0) or None)


def solvers(args):
    tl = None if not args.bundle_time_limit else args.bundle_time_limit
    return get_solver(args.solver, time_limit=tl), get_solver(args.solver)


def common_args(ap):
    ap.add_argument("--seeds", nargs="+", required=True, help="e.g. 101-140 or 1 2 3")
    ap.add_argument("--terminals", type=int, default=4)
    ap.add_argument("--periods", type=int, default=4)
    ap.add_argument("--commodities", type=int, default=6)
    ap.add_argument("--scenarios", type=int, default=12)
    ap.add_argument("--bundles", type=int, default=4)
    ap.add_argument("--kappa", type=float, default=0.0)
    ap.add_argument("--variant-a", action="store_true")
    ap.add_argument("--solver", type=str, default="HiGHS")
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--bundle-time-limit", type=float, default=600.0)
    ap.add_argument("--recovery-every", type=int, default=2)
    ap.add_argument("--ls-evals", type=int, default=12)
    ap.add_argument("--theta-patience", type=int, default=4)
    ap.add_argument("--candidates", type=int, default=20, help="size of the candidate-partition pool")
    ap.add_argument("--out", type=str, required=True)


# --------------------------------------------------------------------------- collect
def cmd_collect(args):
    os.makedirs(args.out, exist_ok=True)
    P_csv = os.path.join(args.out, "partitions.csv")
    solver, lp_solver = solvers(args)
    mult_parts = [p.strip() for p in args.mult_parts.split(",") if p.strip()]
    json.dump(vars(args), open(os.path.join(args.out, "collect_args.json"), "w"), indent=1)
    done_seeds = {r["seed"] for r in rows(P_csv)}
    for seed in parse_seeds(args.seeds):
        inst = make_instance(args, seed)
        print(f"\n##### seed {seed}: {inst.summary()}", flush=True)
        cache = {}
        t0 = time.time()
        if str(seed) not in done_seeds and not args.skip_pool:
            X_sig, Phi = instance_signature_data(inst, lp_solver, cache, args.parallel, seed)
            pool = candidate_partitions(inst, args.bundles, args.candidates, seed, lp_solver, cache, args.parallel)
            proxy = lp_proxy_scores(inst, [B for _, B in pool], lp_solver)
            for (name, B), lpx in zip(pool, proxy):
                ld = make_ld(args, inst, B, solver, lp_solver, 1, cache)
                L0, _ = ld.evaluate(np.zeros((len(B), len(inst.service_ids))))
                f = partition_features(inst, B, X_sig, Phi)
                row = dict(seed=seed, name=name, m=len(B), S=len(inst.scenarios), L0=round(L0, 6),
                           lp_proxy=round(float(lpx), 6), unproved=ld.n_unproved)
                row.update({k: round(float(v), 6) for k, v in zip(PART_FEATURES, f)})
                append(P_csv, row)
            print(f"  partition pool: {len(pool)} candidates  ({time.time() - t0:.0f}s)", flush=True)
        for part in mult_parts:
            path = os.path.join(args.out, f"mult_seed{seed}_{part}.npz")
            if os.path.exists(path):
                continue
            t1 = time.time()
            B = (random_bundles(inst, args.bundles, seed=seed) if part == "random" else
                 stratified_bundles(inst, args.bundles, seed=seed, solver=lp_solver, cache=cache,
                                    parallel=args.parallel))
            res = make_ld(args, inst, B, solver, lp_solver, args.iters_long, cache).run()
            _, y_lp = bundle_lp_designs(inst, B, lp_solver)
            X = multiplier_features(inst, B, y_lp)
            y = multiplier_targets(inst, res.lam_final)
            np.savez(path, X=X, y=y, seed=seed, lb=res.best_lb, ub=res.best_ub, L0=res.history[0]["L"],
                     bundles=json.dumps(B), part=part)
            print(f"  lambda* ({part}): L(0)={res.history[0]['L']:.2f} -> LB*={res.best_lb:.2f} "
                  f"UB*={res.best_ub:.2f} gap={100 * res.gap:.2f}%  ({time.time() - t1:.0f}s)", flush=True)
    print(f"\ndata in {args.out}/")


# --------------------------------------------------------------------------- train
def _training_cases(args, files):
    """Rebuild the training instances and partitions of the lambda* runs for the policy search."""
    ca = argparse.Namespace(**json.load(open(os.path.join(args.data, "collect_args.json"))))
    solver, lp_solver = get_solver(args.solver), get_solver(args.solver)
    cases = []
    for fpath in files[: args.policy_cases]:
        d = np.load(fpath, allow_pickle=False)
        seed = int(d["seed"])
        inst = make_instance(ca, seed)
        if "bundles" in d.files:
            B = json.loads(str(d["bundles"]))
        else:   # data from an early v5 build: regenerate the partition deterministically
            part = "random" if "_random" in fpath else "stratified"
            B = (random_bundles(inst, ca.bundles, seed=seed) if part == "random" else
                 stratified_bundles(inst, ca.bundles, seed=seed, solver=lp_solver, parallel=args.parallel))
        _, y_lp = bundle_lp_designs(inst, B, lp_solver)
        ld = BundledLagrangian(inst, B, solver, verbose=False, parallel=args.parallel)
        cases.append(dict(inst=inst, bundles=B, D=policy_directions(inst, B, y_lp), ld=ld,
                          L0=float(d["L0"]), LBstar=float(d["lb"])))
    return cases


def cmd_train(args):
    import glob
    P = rows(os.path.join(args.data, "partitions.csv"))
    scorer = None
    if P:
        X = np.array([[float(r[k]) for k in PART_FEATURES] for r in P])
        L0 = np.array([float(r["L0"]) for r in P])
        lpx = np.array([float(r["lp_proxy"]) for r in P])
        g = np.array([int(r["seed"]) for r in P])
        scorer = PartitionScorer(use_lp_proxy=(args.scorer == "hybrid"))
        info = scorer.fit(X, L0, g, cv_folds=args.folds, lp_proxy=lpx)
        print("partition scorer:", json.dumps(info, indent=1), flush=True)
    files = sorted(glob.glob(os.path.join(args.data, "mult_seed*.npz")))
    mult = None
    if files and args.ws_model == "regression":
        Xs, ys, gs = [], [], []
        for fpath in files:
            d = np.load(fpath)
            Xs.append(d["X"]); ys.append(d["y"]); gs.append(np.full(len(d["y"]), int(d["seed"])))
        mult = MultiplierPredictor()
        info = mult.fit(np.vstack(Xs), np.concatenate(ys), np.concatenate(gs), cv_folds=args.folds)
        print("multiplier regression:", json.dumps(info, indent=1), flush=True)
    elif files:
        t0 = time.time()
        rng = np.random.default_rng(0)
        files = list(rng.permutation(files))
        n_val = max(1, int(round(args.val_share * len(files)))) if args.val_share > 0 else 0
        val_files, train_files = files[:n_val], files[n_val:]
        args.policy_cases = len(train_files) if args.policy_cases <= 0 else args.policy_cases
        cases = _training_cases(args, train_files)
        print(f"policy search on {len(cases)} training runs ...", flush=True)
        evaluate = lambda c, lam: c["ld"].evaluate(lam)[0]
        ca = json.load(open(os.path.join(args.data, "collect_args.json")))
        w0, step0 = None, args.step0
        if args.init_from:   # start the search at an earlier policy, converted to the same scale
            old, _, meta0 = load_models(args.init_from)
            if args.scale == "bundle" and getattr(old, "scale", "none") != "bundle":
                old = rescale_policy(old, args.init_m or _m_train(meta0))
            w0 = np.asarray(old.w, dtype=float)
            print(f"policy search starts at {args.init_from}: w={np.round(w0, 4)} (scale={args.scale})", flush=True)
        if step0 <= 0:       # automatic: a quarter of the largest initial weight, else the v5 step
            step0 = (0.25 * float(np.abs(w0).max()) if w0 is not None and np.abs(w0).max() > 0
                     else 0.02 * (int(ca["bundles"]) if args.scale == "bundle" else 1))
        mult = DualStartPolicy()
        info = mult.fit(cases, evaluate, max_evals=args.max_evals, scale=args.scale, w0=w0,
                        step0=step0, step_min=step0 / 8)
        if val_files:
            keep = args.policy_cases
            args.policy_cases = len(val_files)
            vcases = [c for c in _training_cases(args, val_files)
                      if c["LBstar"] - c["L0"] > 1e-6 * max(1.0, abs(c["L0"]))]
            args.policy_cases = keep
            clo = []
            for c in vcases:
                lam = mult.multipliers(c["inst"], c["bundles"], solver=get_solver(args.solver))
                clo.append((evaluate(c, lam) - c["L0"]) / (c["LBstar"] - c["L0"]))
            info["validation_closure_mean"] = float(np.mean(clo)) if clo else None
            info["validation_closure_min"] = float(np.min(clo)) if clo else None
            info["validation_cases"] = len(vcases)
        info["seconds"] = round(time.time() - t0, 1)
        print("dual start policy:", json.dumps(info, indent=1), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if scorer is None and args.scorer_from:   # keep the partition scorer of an earlier model file
        _, scorer, _ = load_models(args.scorer_from)
        print(f"partition scorer taken from {args.scorer_from}")
    save_models(args.out, mult, scorer, meta={"data": args.data, "ws_model": args.ws_model,
                                              "scorer": args.scorer, "scale": args.scale,
                                              "init_from": args.init_from, "scorer_from": args.scorer_from})
    print(f"models saved to {args.out}")


def _m_train(meta):
    """Number of bundles of the training data of a model file (from its collect_args.json)."""
    path = os.path.join(meta.get("data", ""), "collect_args.json")
    if not os.path.exists(path):
        raise SystemExit(f"cannot find {path}; pass --m-train (--init-m) explicitly")
    return int(json.load(open(path))["bundles"])


def cmd_rescale(args):
    mult, scorer, meta = load_models(args.models)
    if not isinstance(mult, DualStartPolicy):
        raise SystemExit("rescale needs a DualStartPolicy model")
    m = args.m_train or _m_train(meta)
    new = rescale_policy(mult, m)
    meta = dict(meta); meta.update(scale="bundle", rescaled_from=args.models, rescaled_m=m)
    save_models(args.out, new, scorer, meta=meta)
    print(f"w (scale none, m={m}) = {np.round(np.asarray(mult.w, dtype=float), 5)}  ->  "
          f"w (scale bundle) = {np.round(new.w, 5)};  saved to {args.out}")


# --------------------------------------------------------------------------- eval
def load_transfer(path, solver, parallel):
    """WarmTransfer bank from the lambda* runs of a `collect` directory."""
    import glob
    ca = argparse.Namespace(**json.load(open(os.path.join(path, "collect_args.json"))))
    bank = WarmTransfer()
    for fpath in sorted(glob.glob(os.path.join(path, "mult_seed*.npz"))):
        d = np.load(fpath, allow_pickle=False)
        seed = int(d["seed"])
        inst = make_instance(ca, seed)
        if "bundles" in d.files:
            B = json.loads(str(d["bundles"]))
        else:
            part = "random" if "_random" in fpath else "stratified"
            B = (random_bundles(inst, ca.bundles, seed=seed) if part == "random" else
                 stratified_bundles(inst, ca.bundles, seed=seed, solver=solver, parallel=parallel))
        bank.add(inst, B, d["y"], name=os.path.basename(fpath))
    print(f"warm-transfer bank: {len(bank)} runs from {path}")
    return bank


def read_zstar(path, seed):
    for r in rows(path) if path else []:
        if r["seed"] == str(seed) and r.get("de_proved") == "1":
            return float(r["de_obj"])
    return None


def cmd_eval(args):
    os.makedirs(args.out, exist_ok=True)
    W_csv, B_csv, H_csv = (os.path.join(args.out, f) for f in ("warmstart.csv", "bundling.csv", "history.csv"))
    mult, scorer, _ = load_models(args.models)
    solver, lp_solver = solvers(args)
    bank = load_transfer(args.transfer_data, lp_solver, args.parallel) if args.transfer_data else None
    starts = [s.strip() for s in args.starts.split(",") if s.strip()]
    json.dump(vars(args), open(os.path.join(args.out, "eval_args.json"), "w"), indent=1)

    def log_history(seed, exp, label, res):
        for h in res.history:
            append(H_csv, dict(seed=seed, exp=exp, label=label, t=h["t"], L=round(h["L"], 4),
                               LB=round(h["best_lb"], 4),
                               UB=round(h["best_ub"], 4) if np.isfinite(h["best_ub"]) else "",
                               time=round(h["time"], 2)))

    for seed in parse_seeds(args.seeds):
        inst = make_instance(args, seed)
        zstar = read_zstar(args.de_csv, seed)
        print(f"\n##### seed {seed}: {inst.summary()}  z*={zstar}", flush=True)
        cache = {}
        gap = lambda v: round(100 * (zstar - v) / zstar, 4) if zstar else ""

        # ---------------- warm start on a fixed partition
        done_w = {(r["seed"], r["start"]) for r in rows(W_csv)}
        if args.ws_partition == "random":
            B = random_bundles(inst, args.bundles, seed=seed)
        elif args.ws_partition == "stratified":
            B = stratified_bundles(inst, args.bundles, seed=seed, solver=lp_solver, cache=cache, parallel=args.parallel)
        else:   # learned / lpproxy: select from the candidate pool (full LABLD when combined with 'ml')
            Xs, Ph = instance_signature_data(inst, lp_solver, cache, args.parallel, seed)
            pool = candidate_partitions(inst, args.bundles, args.candidates, seed, lp_solver, cache, args.parallel)
            prx = lp_proxy_scores(inst, [P for _, P in pool], lp_solver)
            if args.ws_partition == "learned":
                if scorer is None:
                    raise SystemExit("--ws-partition learned needs a partition scorer in --models")
                i, _ = scorer.select(inst, pool, Xs, Ph, lp_proxy=prx if scorer.use_lp_proxy else None)
            else:
                i = int(np.argmax(prx))
            B = pool[i][1]
            print(f"  warm-start partition ({args.ws_partition}): {pool[i][0]}", flush=True)
        results = {}
        for st in starts:
            if (str(seed), st) in done_w:
                continue
            t0 = time.time()
            cands, names = [], []
            if st == "ml":
                if mult is None:
                    print("  no multiplier model in --models; skipping 'ml'"); continue
                lam_ml = (mult.multipliers(inst, B, solver=lp_solver) if isinstance(mult, DualStartPolicy)
                          else mult.predict(inst, B, solver=lp_solver))
                cands, names = [lam_ml], ["ml"]
            elif st == "transfer":
                if bank is None:
                    print("  no --transfer-data; skipping 'transfer'"); continue
                lam_tr, src = bank.multipliers(inst, B)
                if lam_tr is None:
                    print(f"  transfer: {src}; skipping"); continue
                cands, names = [lam_tr], ["transfer"]
            elif st == "lpdual":
                lp = lp_dual_multipliers(inst, B, lp_solver)
                cands, names = [lp, -lp], ["+LPdual", "-LPdual"]
            elif st == "oracle":
                long = make_ld(args, inst, B, solver, lp_solver, args.iters_long, cache).run()
                cands, names = [long.lam_final], ["oracle"]
            prep = time.time() - t0
            # fresh evaluation cache per run: otherwise later runs reuse earlier design evaluations
            # and look faster than they are
            res = make_ld(args, inst, B, solver, lp_solver, args.iters, {}).run(cands, names)
            results[st] = res
            log_history(seed, "warmstart", st, res)
            append(W_csv, dict(seed=seed, partition=args.ws_partition, start=st, chosen=res.start_source,
                               zstar=zstar or "", L_start=round(res.history[0]["L"], 4),
                               LB=round(res.best_lb, 4), UB=round(res.best_ub, 4), gap_pct=round(100 * res.gap, 4),
                               L_start_gap_pct=gap(res.history[0]["L"]), LB_gap_pct=gap(res.best_lb),
                               iters=res.iterations, prep_time=round(prep, 2), ld_time=round(res.seconds, 2)))
            print(f"  start={st:<7} chosen={res.start_source:<8} L(start)={res.history[0]['L']:.2f} "
                  f"LB*={res.best_lb:.2f} UB*={res.best_ub:.2f}  prep {prep:.0f}s + {res.seconds:.0f}s", flush=True)
        if {"zero", "oracle"} <= set(results):
            tgt = results["zero"].best_lb
            T = {k: time_to_target(r.history, tgt, key="t") for k, r in results.items()}
            ohc = {k: oracle_headroom_capture(T["zero"], T[k], T["oracle"]) for k in results
                   if k not in ("zero", "oracle")}
            print(f"  iterations to reach LB(zero)={tgt:.2f}: {T}  OHC={ohc}", flush=True)

        # ---------------- partition choice
        if args.skip_bundling or str(seed) in {r["seed"] for r in rows(B_csv)}:
            continue
        t0 = time.time()
        X_sig, Phi = instance_signature_data(inst, lp_solver, cache, args.parallel, seed)
        pool = candidate_partitions(inst, args.bundles, args.candidates, seed, lp_solver, cache, args.parallel)
        t_pool = time.time() - t0
        picks = {"random": random_bundles(inst, args.bundles, seed=seed),
                 "stratified": stratified_bundles(inst, args.bundles, seed=seed, solver=lp_solver,
                                                  cache=cache, parallel=args.parallel)}
        t1 = time.time()
        proxy = lp_proxy_scores(inst, [P for _, P in pool], lp_solver)
        picks["lpproxy"] = pool[int(np.argmax(proxy))][1]
        t_proxy = time.time() - t1
        t1 = time.time()
        if scorer is not None:
            i, _ = scorer.select(inst, pool, X_sig, Phi, lp_proxy=proxy if scorer.use_lp_proxy else None)
            picks["learned"] = pool[i][1]
            print(f"  learned selection: {pool[i][0]}   LP-proxy selection: {pool[int(np.argmax(proxy))][0]}",
                  flush=True)
        t_learn = time.time() - t1 + (t_proxy if scorer is not None and scorer.use_lp_proxy else 0.0)
        pool_best = ""
        if args.pool_l0:   # L(0) of every candidate: the best attainable choice from the pool
            vals = [make_ld(args, inst, P, solver, lp_solver, 1, cache).evaluate(
                np.zeros((len(P), len(inst.service_ids))))[0] for _, P in pool]
            pool_best = round(max(vals), 4)
        for name, P in picks.items():
            res = make_ld(args, inst, P, solver, lp_solver, args.iters, {}).run()
            log_history(seed, "bundling", name, res)
            sel_time = {"learned": t_pool + t_learn, "lpproxy": t_pool + t_proxy}.get(name, 0.0)
            append(B_csv, dict(seed=seed, choice=name, zstar=zstar or "", L0=round(res.history[0]["L"], 4),
                               pool_best_L0=pool_best, LB=round(res.best_lb, 4), UB=round(res.best_ub, 4),
                               gap_pct=round(100 * res.gap, 4), L0_gap_pct=gap(res.history[0]["L"]),
                               LB_gap_pct=gap(res.best_lb), select_time=round(sel_time, 2),
                               ld_time=round(res.seconds, 2),
                               bundle_sizes="-".join(map(str, sorted(len(b) for b in P)))))
            print(f"  choice={name:<10} L(0)={res.history[0]['L']:.2f} LB*={res.best_lb:.2f} "
                  f"UB*={res.best_ub:.2f}  ({res.seconds:.0f}s)", flush=True)
    print(f"\nresults in {args.out}/")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect"); common_args(c)
    c.add_argument("--iters-long", type=int, default=60, help="iterations of the run that yields lambda*")
    c.add_argument("--mult-parts", type=str, default="random,stratified")
    c.add_argument("--skip-pool", action="store_true",
                   help="only the long lambda* runs (e.g. transfer data at the test size), no partition pool")
    t = sub.add_parser("train")
    t.add_argument("--data", required=True); t.add_argument("--out", required=True)
    t.add_argument("--folds", type=int, default=5)
    t.add_argument("--ws-model", choices=["policy", "regression"], default="policy")
    t.add_argument("--scorer", choices=["hybrid", "struct"], default="hybrid",
                   help="hybrid: structural features + LP proxy; struct: structural features only (no solves)")
    t.add_argument("--solver", type=str, default="HiGHS")
    t.add_argument("--parallel", type=int, default=4)
    t.add_argument("--policy-cases", type=int, default=0, help="max training runs for the policy search (0 = all)")
    t.add_argument("--val-share", type=float, default=0.25, help="share of the lambda* runs held out")
    t.add_argument("--max-evals", type=int, default=120)
    t.add_argument("--scale", choices=["bundle", "none"], default="bundle",
                   help="bundle: multipliers proportional to P_b F (valid for any m); none: v5-v8 behaviour")
    t.add_argument("--init-from", type=str, default="", help="model file whose policy starts the search")
    t.add_argument("--init-m", type=int, default=0, help="training m of --init-from (default: from its data)")
    t.add_argument("--step0", type=float, default=0.0, help="initial search step (0 = automatic)")
    t.add_argument("--scorer-from", type=str, default="",
                   help="copy the partition scorer of this model file (when --data has no partition pool)")
    r = sub.add_parser("rescale", help="convert a v5-v8 policy to the bundle scale (exact at its training m)")
    r.add_argument("--models", required=True); r.add_argument("--out", required=True)
    r.add_argument("--m-train", type=int, default=0, help="training m (default: from the model's data)")
    e = sub.add_parser("eval"); common_args(e)
    e.add_argument("--models", required=True)
    e.add_argument("--iters", type=int, default=30)
    e.add_argument("--iters-long", type=int, default=60)
    e.add_argument("--starts", type=str, default="zero,ml,oracle",
                   help="subset of zero,ml,transfer,lpdual,oracle")
    e.add_argument("--transfer-data", type=str, default="",
                   help="collect directory with lambda* runs of the test size (warm-transfer baseline)")
    e.add_argument("--ws-partition", type=str, default="stratified",
                   choices=["stratified", "random", "learned", "lpproxy"],
                   help="partition for the warm-start comparison; learned + start 'ml' = full LABLD")
    e.add_argument("--de-csv", type=str, default="", help="de.csv of a previous study (z* for gaps)")
    e.add_argument("--pool-l0", action="store_true", help="also evaluate L(0) of every pool candidate")
    e.add_argument("--skip-bundling", action="store_true")
    e.add_argument("--theta-warm", type=float, default=0.0,
                   help="initial Polyak factor after an accepted non-zero start (0 = same as the zero start)")
    args = ap.parse_args()
    {"collect": cmd_collect, "train": cmd_train, "eval": cmd_eval, "rescale": cmd_rescale}[args.cmd](args)


if __name__ == "__main__":
    main()
