import numpy as np
import pytest

from ssnd_labld import BundledLagrangian, get_solver, solve_deterministic_equivalent, stratified_bundles
from ssnd_labld.instance import Arc, Commodity, Scenario, SSNDInstance
from ssnd_labld.snd_io import read_snd, write_snd

gp = pytest.importorskip("gurobipy")


def _small_snd(N=6, seed=0):
    rng = np.random.default_rng(seed)
    nodes = [str(i) for i in range(4)]
    pairs = [(0, 1), (1, 2), (2, 3), (0, 2), (1, 3), (3, 0), (2, 0), (0, 3)]
    arcs = [Arc(id=i, tail=str(a), head=str(b), is_service=True, mode="arc", F=float(rng.uniform(50, 120)),
                U=float(rng.uniform(8, 15)), c=float(rng.uniform(1, 4))) for i, (a, b) in enumerate(pairs)]
    comms = [Commodity(id=0, origin="0", dest="3", penalty=40.0, nominal_demand=10.0),
             Commodity(id=1, origin="1", dest="0", penalty=30.0, nominal_demand=6.0)]
    scens = [Scenario(id=i, prob=1.0 / N, demand={0: float(rng.uniform(4, 16)), 1: float(rng.uniform(2, 10))}, xi={})
             for i in range(N)]
    return SSNDInstance(name="small_snd", nodes=nodes, arcs=arcs, commodities=comms, scenarios=scens)


def _gurobi_ok():
    try:
        e = gp.Env(); e.dispose(); return True
    except Exception:
        return False


def test_snd_roundtrip(tmp_path):
    inst = _small_snd()
    p = tmp_path / "snd_small.txt"
    write_snd(inst, str(p))
    back = read_snd(str(p))
    assert len(back.arcs) == 8 and len(back.scenarios) == 6
    assert abs(back.arcs[3].F - inst.arcs[3].F) < 1e-9
    assert abs(back.scenarios[5].demand[1] - inst.scenarios[5].demand[1]) < 1e-9


@pytest.mark.skipif(not _gurobi_ok(), reason="Gurobi not available")
def test_backend_matches_pulp_and_bounds_are_valid():
    from ssnd_labld.grb_backend import GurobiBackend, solve_de_gurobi
    inst = _small_snd()
    de_pulp = solve_deterministic_equivalent(inst, get_solver("HiGHS"))
    de_fast = solve_de_gurobi(inst, threads=1, mip_gap=1e-9)
    assert abs(de_pulp.objective - de_fast.objective) < 1e-4 * abs(de_pulp.objective)
    be = GurobiBackend(inst, parallel=2)
    # recourse evaluation equals the DE objective at the optimal design
    ev = be.evaluate_design(de_fast.y)
    assert abs(ev.expected_total_cost - de_fast.objective) < 1e-4 * abs(de_fast.objective)
    B = stratified_bundles(inst, 3, seed=0, evaluator=be.evaluate_design)
    be.set_bundles(B)
    ld = BundledLagrangian(inst, B, max_iter=15, parallel=2, verbose=False, backend=be)
    r = ld.run()
    tol = 1e-6 * abs(de_fast.objective)
    assert r.best_lb <= de_fast.objective + tol and de_fast.objective <= r.best_ub + tol
    be.close()


@pytest.mark.skipif(not _gurobi_ok(), reason="Gurobi not available")
def test_deadline_keeps_bounds_valid():
    """A deadline that expires inside an iteration discards that iteration; bounds stay valid."""
    import time
    from ssnd_labld.grb_backend import GurobiBackend, solve_de_gurobi
    inst = _small_snd(N=8, seed=1)
    de = solve_de_gurobi(inst, threads=1, mip_gap=1e-9)
    be = GurobiBackend(inst, parallel=2)
    B = stratified_bundles(inst, 4, seed=0, evaluator=be.evaluate_design)
    be.set_bundles(B)
    lam = np.zeros((4, len(inst.service_ids)))
    vals, ys, proved, complete = be.solve_bundles(lam, deadline=time.time() - 1.0)
    assert not complete
    ld = BundledLagrangian(inst, B, max_iter=10000, time_limit=0.3, parallel=2, backend=be)
    r = ld.run()
    tol = 1e-6 * abs(de.objective)
    assert r.best_lb <= de.objective + tol and de.objective <= r.best_ub + tol
    assert r.seconds < 5.0
    be.close()
