import numpy as np

from ssnd_labld import generate_instance, get_solver, stratified_bundles
from ssnd_labld.learning import DualStartPolicy, bundle_lp_designs, rescale_policy


def _setup(S, m):
    inst = generate_instance(n_terminals=4, n_periods=4, n_commodities=6, n_scenarios=S, seed=3)
    B = stratified_bundles(inst, m, seed=3, solver=get_solver("HiGHS"))
    _, y_lp = bundle_lp_designs(inst, B, get_solver("HiGHS"))
    return inst, B, y_lp


def test_rescale_exact_at_training_m():
    old = DualStartPolicy(w=np.array([0.04, 0.09, 0.01, 0.0]))
    new = rescale_policy(old, 4)
    assert new.scale == "bundle" and getattr(old, "scale", "none") == "none"
    inst, B, y = _setup(12, 4)
    assert np.allclose(old.multipliers(inst, B, y_lp=y), new.multipliers(inst, B, y_lp=y))


def test_bundle_scale_shrinks_with_m():
    new = rescale_policy(DualStartPolicy(w=np.array([0.04, 0.09, 0.01, 0.0])), 4)
    inst, B, y = _setup(24, 8)
    old_lam = DualStartPolicy(w=np.array([0.04, 0.09, 0.01, 0.0])).multipliers(inst, B, y_lp=y)
    assert np.allclose(new.multipliers(inst, B, y_lp=y), old_lam * 4 / 8)
    assert np.allclose(new.multipliers(inst, B, y_lp=y).sum(axis=0), 0.0)
