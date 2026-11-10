"""Safety properties of the learning components (HiGHS or CBC, ~20 s)."""
import numpy as np
import pytest

from ssnd_labld import (BundledLagrangian, MultiplierPredictor, PartitionScorer, candidate_partitions,
                        generate_instance, get_solver, multiplier_features, partition_features,
                        random_bundles, solve_deterministic_equivalent)
from ssnd_labld.learning import (DualStartPolicy, bundle_lp_designs, instance_signature_data,
                                 multiplier_targets, policy_directions)


@pytest.fixture(scope="module")
def setup():
    inst = generate_instance(n_terminals=3, n_periods=3, n_commodities=4, n_scenarios=6, seed=3)
    solver = get_solver("HiGHS")
    return inst, solver, solve_deterministic_equivalent(inst, solver)


def test_learned_start_is_valid_and_guarded(setup):
    inst, solver, de = setup
    B = random_bundles(inst, 2, seed=0)
    ld = BundledLagrangian(inst, B, solver, max_iter=8, verbose=False, ls_max_evals=6)
    ref = ld.run()
    _, y_lp = bundle_lp_designs(inst, B, solver)
    X, y = multiplier_features(inst, B, y_lp), multiplier_targets(inst, ref.lam_final)
    mp = MultiplierPredictor()
    mp.fit(X, y)                                   # fitted on this instance only: a functional test
    lam = mp.predict(inst, B, y_lp=y_lp)
    assert np.allclose(lam.sum(axis=0), 0.0, atol=1e-8)          # in Lambda
    ld2 = BundledLagrangian(inst, B, solver, max_iter=4, verbose=False, ls_max_evals=6)
    res = ld2.run(candidates=[lam], candidate_names=["ml"])
    L0, _ = ld2.evaluate(np.zeros_like(lam))
    assert res.history[0]["L"] >= L0 - 1e-6                      # guard: never worse than zero
    assert res.best_lb <= de.objective + 1e-6 * max(1, abs(de.objective))


def test_partition_scorer_selects_valid_partition(setup):
    inst, solver, _ = setup
    X_sig, Phi = instance_signature_data(inst, solver)
    pool = candidate_partitions(inst, 2, n_candidates=8, seed=0, solver=solver)
    ids = sorted(s.id for s in inst.scenarios)
    for _, P in pool:
        assert sorted(w for b in P for w in b) == ids
    F = np.vstack([partition_features(inst, P, X_sig, Phi) for _, P in pool])
    sc = PartitionScorer(use_lp_proxy=False)
    sc.fit(F, np.arange(len(pool), dtype=float), np.zeros(len(pool), dtype=int))
    i, s = sc.select(inst, pool, X_sig, Phi)
    assert 0 <= i < len(pool) and len(s) == len(pool)


def test_policy_search_never_below_zero_start(setup):
    inst, solver, de = setup
    B = random_bundles(inst, 2, seed=1)
    ld = BundledLagrangian(inst, B, solver, max_iter=8, verbose=False, ls_max_evals=6)
    ref = ld.run()
    _, y_lp = bundle_lp_designs(inst, B, solver)
    case = dict(inst=inst, bundles=B, D=policy_directions(inst, B, y_lp), L0=ref.history[0]["L"],
                LBstar=ref.best_lb + 1e-3)
    pol = DualStartPolicy()
    info = pol.fit([case], lambda c, lam: ld.evaluate(lam)[0], max_evals=10, verbose=False)
    assert info["train_closure_mean"] >= 0.0                     # search starts at w = 0
    lam = pol.multipliers(inst, B, y_lp=y_lp)
    assert np.allclose(lam.sum(axis=0), 0.0, atol=1e-8)
    assert ld.evaluate(lam)[0] <= de.objective + 1e-6 * max(1, abs(de.objective))


def test_warm_transfer_is_balanced_and_needs_same_network(setup):
    from ssnd_labld.learning import WarmTransfer
    inst, solver, _ = setup
    other = generate_instance(n_terminals=3, n_periods=3, n_commodities=4, n_scenarios=6, seed=4)
    B = random_bundles(inst, 2, seed=0)
    Bo = random_bundles(other, 2, seed=0)
    bank = WarmTransfer()
    bank.add(other, Bo, np.random.default_rng(0).normal(size=(2, len(other.service_ids))), "other")
    lam, src = bank.multipliers(inst, B)
    assert lam is not None and src == "other"
    assert np.allclose(lam.sum(axis=0), 0.0, atol=1e-8)
    small = generate_instance(n_terminals=2, n_periods=3, n_commodities=3, n_scenarios=6, seed=1)
    assert bank.multipliers(small, random_bundles(small, 2, seed=0))[0] is None   # different network
