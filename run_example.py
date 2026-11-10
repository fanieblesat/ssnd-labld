"""End-to-end example + numerical validity checks.

    python run_example.py [--scenarios 12] [--bundles 4] [--iters 25] [--solver CBC]

What it does
  1. generates a small multimodal corridor instance with joint demand/disruption scenarios
  2. solves the deterministic equivalent (reference optimum z*)
  3. checks Proposition 1 on random balanced multipliers:      L(lambda) <= z*
  4. checks Proposition 2 numerically (coarser partition => bound no weaker)
  5. runs the bundled Lagrangian method with singleton / random / k-means / solution-similarity
     partitions, guarded start over {0, +LP-dual, -LP-dual}
  6. asserts LB* <= z* <= UB* for every run, prints a comparison table, and demonstrates the
     time-to-target / oracle-headroom-capture diagnostic
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from ssnd_labld import (BundledLagrangian, cost_signature_bundles, generate_instance, get_solver, kmeans_bundles,
                        lp_dual_multipliers, random_bundles, singleton_bundles,
                        solution_similarity_bundles, solve_deterministic_equivalent,
                        evaluate_design)
from ssnd_labld.bundling import describe_bundles
from ssnd_labld.lagrangian import oracle_headroom_capture, time_to_target
from ssnd_labld.warmstart import project_balanced


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--terminals", type=int, default=4)
    ap.add_argument("--periods", type=int, default=4)
    ap.add_argument("--commodities", type=int, default=6)
    ap.add_argument("--scenarios", type=int, default=12)
    ap.add_argument("--bundles", type=int, default=4)
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--kappa", type=float, default=0.0)
    ap.add_argument("--solver", type=str, default="CBC")
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--de-time-limit", type=float, default=600.0,
                    help="seconds for the deterministic equivalent (0 = no limit). If hit, z* is replaced by "
                         "the incumbent (UB) and the LP bound; validity checks use what is provable")
    ap.add_argument("--bundle-time-limit", type=float, default=120.0,
                    help="seconds per bundle MILP (0 = no limit); if hit, the LP-relaxation value is used "
                         "so L(lambda) stays a valid bound")
    ap.add_argument("--recovery-every", type=int, default=1,
                    help="primal recovery (the expensive recourse evaluations) every k-th iteration")
    ap.add_argument("--ls-evals", type=int, default=12, help="max design evaluations per recovery")
    ap.add_argument("--max-bundle-size", type=int, default=-1,
                    help="cap on scenarios per bundle for kmeans/solsim/costsig; -1 = ceil(1.5*S/m), 0 = no cap")
    ap.add_argument("--theta-patience", type=int, default=4,
                    help="halve the Polyak theta after this many iterations without LB improvement")
    ap.add_argument("--no-checks", action="store_true", help="skip the Prop.1/Prop.2 numerical checks")
    ap.add_argument("--skip-de", action="store_true", help="do not solve the DE at all (large instances)")
    ap.add_argument("--partitions", type=str, default="singleton,random,kmeans,solsim,costsig",
                    help="comma list from: singleton,random,kmeans,solsim,costsig")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--spot-requires-activation", action="store_true",
                    help="model variant B: spot capacity only on activated services (audit 4.1)")
    args = ap.parse_args()
    verbose = not args.quiet
    tl = lambda v: None if not v else v
    solver = get_solver(args.solver, msg=False, time_limit=tl(args.bundle_time_limit))   # bundle MILPs
    lp_solver = get_solver(args.solver, msg=False)                                        # LPs: no limit
    de_solver = get_solver(args.solver, msg=False, time_limit=tl(args.de_time_limit))

    inst = generate_instance(n_terminals=args.terminals, n_periods=args.periods,
                             n_commodities=args.commodities, n_scenarios=args.scenarios,
                             kappa=args.kappa, seed=args.seed,
                             spot_requires_activation=args.spot_requires_activation)
    print("Instance:", inst.summary())
    print("Regimes:", [s.regime for s in inst.scenarios])

    # ---- 1. deterministic equivalent ---------------------------------------
    class _DE:  # placeholder when skipped
        objective, bound, proved_optimal, seconds, y = float("nan"), None, False, 0.0, None
    if args.skip_de:
        de = _DE()
        print("\n[DE] skipped")
    else:
        de = solve_deterministic_equivalent(inst, de_solver)
        tag = "PROVED OPTIMAL" if de.proved_optimal else f"TIME LIMIT: incumbent only, LP bound {de.bound:.3f}"
        print(f"\n[DE] {tag}  incumbent={de.objective:.3f}  ({de.seconds:.1f}s)  "
              f"active services={int(sum(de.y.values()))}/{len(inst.service_ids)}")
        ev = evaluate_design(inst, de.y, lp_solver, parallel=args.parallel)
        print(f"[DE] re-evaluated expected cost={ev.expected_total_cost:.3f}  EM={ev.expected_emissions:.2f}  "
              f"REL={ev.reliability:.4f}  spot={ev.expected_spot:.2f}")
        assert abs(ev.expected_total_cost - de.objective) < 1e-4 * max(1, abs(de.objective)), \
            "DE objective and re-evaluation disagree -> model inconsistency"
    zref = de.objective                                  # reference: z* if proved, else best known UB
    z_ub = zref if not args.skip_de else float("inf")    # valid UB on z*
    z_exact = (not args.skip_de) and de.proved_optimal

    import math
    cap = (math.ceil(1.5 * args.scenarios / args.bundles) if args.max_bundle_size < 0
           else (args.max_bundle_size or None))
    part = kmeans_bundles(inst, args.bundles, seed=args.seed, max_size=cap)
    F = np.array([inst.arcs[a].F for a in inst.service_ids])
    rng = np.random.default_rng(args.seed)
    if not args.no_checks and z_exact:
        # ---- 2. Proposition 1 on random balanced multipliers ----------------
        ld = BundledLagrangian(inst, part, solver, verbose=False, parallel=args.parallel)
        print("\n[Prop.1] random lambda in Lambda:")
        for trial in range(3):
            lam = project_balanced(rng.normal(scale=0.5 * F.mean(), size=(len(part), len(F))))
            L, _ = ld.evaluate(lam)
            ok = L <= de.objective + 1e-6
            print(f"   trial {trial}: L(lambda)={L:.3f} <= z*={de.objective:.3f}  {'OK' if ok else 'VIOLATED'}")
            assert ok, "Proposition 1 violated - implementation bug"
        # ---- 3. Proposition 2 -----------------------------------------------
        fine = singleton_bundles(inst)
        ld_fine = BundledLagrangian(inst, fine, solver, verbose=False, parallel=args.parallel)
        lam_fine = project_balanced(rng.normal(scale=0.5 * F.mean(), size=(len(fine), len(F))))
        L_fine, _ = ld_fine.evaluate(lam_fine)
        mu = np.vstack([lam_fine[[fine.index([w]) for w in B]].sum(axis=0) for B in part])
        L_coarse, _ = ld.evaluate(mu)
        print(f"\n[Prop.2] L_fine(lambda)={L_fine:.3f} <= L_coarse(mu)={L_coarse:.3f}  "
              f"{'OK' if L_coarse >= L_fine - 1e-6 else 'VIOLATED'}")
        assert L_coarse >= L_fine - 1e-6
    else:
        print("\n[Prop.1/2] skipped (--no-checks, or DE not proved optimal)")

    # ---- 4. Lagrangian runs with different partitions -----------------------
    want = [w.strip() for w in args.partitions.split(",") if w.strip()]
    makers = {
        "singleton": lambda: ("singleton", singleton_bundles(inst)),
        "random": lambda: (f"random(m={args.bundles})", random_bundles(inst, args.bundles, seed=args.seed)),
        "kmeans": lambda: (f"kmeans(m={args.bundles})", part),
        "solsim": lambda: (f"solsim(m={args.bundles})",
                           solution_similarity_bundles(inst, args.bundles, solver=solver, max_size=cap)),
        "costsig": lambda: (f"costsig(m={args.bundles})",
                            cost_signature_bundles(inst, args.bundles, seed=args.seed, solver=lp_solver,
                                                   parallel=args.parallel, max_size=cap)),
    }
    configs = dict(makers[w]() for w in want)
    results = {}
    for name, bundles in configs.items():
        print(f"\n=== {name} ===")
        if verbose:
            print(describe_bundles(inst, bundles))
        lp_lam = lp_dual_multipliers(inst, bundles, lp_solver)
        ld = BundledLagrangian(inst, bundles, solver, step_rule="polyak", theta=1.0, tau=0.5,
                               max_iter=args.iters, gap_tol=1e-3, parallel=args.parallel,
                               verbose=verbose, ls_max_evals=args.ls_evals, recovery_every=args.recovery_every,
                               recourse_solver=lp_solver, theta_patience=args.theta_patience)
        res = ld.run(candidates=[lp_lam, -lp_lam], candidate_names=["+LPdual", "-LPdual"])
        print(res.summary())
        tol = 1e-6 * max(1, abs(zref)) if not args.skip_de else 0.0
        lb_ok = args.skip_de or res.best_lb <= z_ub + tol          # LB <= z* <= incumbent: always provable
        ub_ok = (not z_exact) or res.best_ub >= de.objective - tol  # UB >= z* only checkable if z* proved
        print(f"   validity: LB*={res.best_lb:.3f} <= z*{'' if z_exact else '(<=incumbent)'}"
              f"={zref:.3f} <= UB*={res.best_ub:.3f}  {'OK' if lb_ok and ub_ok else 'VIOLATED'}")
        assert lb_ok and ub_ok, "bound validity violated - implementation bug"
        results[name] = res

    # ---- 5. summary table ----------------------------------------------------
    print("\n" + "=" * 106)
    print(f"{'partition':<18}{'start':>10}{'LB*':>12}{'UB*':>12}{'gap%':>8}{'iters':>7}{'bundle solves':>15}{'designs':>9}{'time s':>9}")
    print("-" * 106)
    for name, r in results.items():
        print(f"{name:<18}{r.start_source:>10}{r.best_lb:>12.2f}{r.best_ub:>12.2f}{100 * r.gap:>8.2f}"
              f"{r.iterations:>7}{r.n_bundle_solves:>15}{r.n_design_evals:>9}{r.seconds:>9.1f}")
    if not args.skip_de:
        lo = de.objective if de.proved_optimal else de.bound
        print(f"{'DE (z*)' if de.proved_optimal else 'DE (LPbd/inc)':<18}{'':>10}{lo:>12.2f}{de.objective:>12.2f}"
              f"{100 * (de.objective - lo) / abs(de.objective):>8.2f}{'':>7}{'':>15}{'':>9}{de.seconds:>9.1f}")
    print("=" * 106)

    # ---- 6. oracle-headroom diagnostic (demonstration) ----------------------
    r = next(iter(results.values()))
    target = r.best_lb - 1e-6            # target bound: the best bound this run reached
    T_zero = time_to_target(r.history, target, key="t")
    print(f"\n[diagnostic] iterations to reach LB target {target:.2f} from the guarded start: {T_zero}")
    print("   (to compute OHC: rerun with start=0, start=ML prediction, start=oracle multipliers, and\n"
          "    feed the three iteration counts to oracle_headroom_capture(T0, Tml, Toracle))")
    print("\nAll validity checks passed.")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"total wall time {time.time() - t0:.1f}s")
