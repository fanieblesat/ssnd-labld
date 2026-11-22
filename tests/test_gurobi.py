import pytest

from ssnd_labld import BundledLagrangian, generate_instance, get_solver, solve_deterministic_equivalent, stratified_bundles

gp = pytest.importorskip("gurobipy")


def _gurobi_works():
    try:
        return get_solver("GUROBI").available()
    except Exception:
        return False


@pytest.mark.skipif(not _gurobi_works(), reason="Gurobi not available")
def test_gurobi_matches_highs_and_runs_in_parallel():
    inst = generate_instance(n_terminals=3, n_periods=3, n_commodities=3, n_scenarios=4, seed=1)
    a = solve_deterministic_equivalent(inst, get_solver("HiGHS", time_limit=60))
    b = solve_deterministic_equivalent(inst, get_solver("GUROBI", time_limit=60, threads=2))
    assert b.proved_optimal and abs(a.objective - b.objective) < 1e-4
    B = stratified_bundles(inst, 2, seed=1, solver=get_solver("GUROBI"))
    ld = BundledLagrangian(inst, B, get_solver("GUROBI", time_limit=30), max_iter=4, parallel=2,
                           verbose=False, recourse_solver=get_solver("GUROBI"))
    res = ld.run()
    assert res.best_lb <= b.objective + 1e-4 <= res.best_ub + 2e-4
