"""Numerical checks of the bound properties on a small instance (CBC, ~1 min)."""
import numpy as np
import pytest

from ssnd_labld import (BundledLagrangian, generate_instance, get_solver, kmeans_bundles,
                        singleton_bundles, solve_deterministic_equivalent, evaluate_design)
from ssnd_labld.warmstart import project_balanced


@pytest.fixture(scope="module")
def setup():
    inst = generate_instance(n_terminals=3, n_periods=3, n_commodities=4, n_scenarios=6, seed=3)
    solver = get_solver("CBC")
    de = solve_deterministic_equivalent(inst, solver)
    return inst, solver, de


def test_de_matches_reevaluation(setup):
    inst, solver, de = setup
    ev = evaluate_design(inst, de.y, solver)
    assert abs(ev.expected_total_cost - de.objective) < 1e-5 * max(1.0, abs(de.objective))


def test_proposition1_lower_bound(setup):
    inst, solver, de = setup
    part = kmeans_bundles(inst, 2, seed=0)
    ld = BundledLagrangian(inst, part, solver, verbose=False)
    rng = np.random.default_rng(0)
    F = np.array([inst.arcs[a].F for a in inst.service_ids])
    for _ in range(3):
        lam = project_balanced(rng.normal(scale=0.5 * F.mean(), size=(len(part), len(F))))
        L, _ = ld.evaluate(lam)
        assert L <= de.objective + 1e-6


def test_proposition2_coarsening(setup):
    inst, solver, de = setup
    fine, coarse = singleton_bundles(inst), kmeans_bundles(inst, 2, seed=0)
    ld_f = BundledLagrangian(inst, fine, solver, verbose=False)
    ld_c = BundledLagrangian(inst, coarse, solver, verbose=False)
    rng = np.random.default_rng(1)
    F = np.array([inst.arcs[a].F for a in inst.service_ids])
    lam = project_balanced(rng.normal(scale=0.5 * F.mean(), size=(len(fine), len(F))))
    Lf, _ = ld_f.evaluate(lam)
    mu = np.vstack([lam[[fine.index([w]) for w in B]].sum(axis=0) for B in coarse])
    Lc, _ = ld_c.evaluate(mu)
    assert Lc >= Lf - 1e-6


def test_lagrangian_run_bounds(setup):
    inst, solver, de = setup
    ld = BundledLagrangian(inst, kmeans_bundles(inst, 2, seed=0), solver, max_iter=6,
                           verbose=False, ls_max_evals=6)
    res = ld.run()
    tol = 1e-6 * max(1.0, abs(de.objective))
    assert res.best_lb <= de.objective + tol
    assert res.best_ub >= de.objective - tol
    assert np.allclose(res.lam_final.sum(axis=0), 0.0, atol=1e-9)   # stays in Lambda
