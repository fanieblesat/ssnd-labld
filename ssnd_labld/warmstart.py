"""Dual warm starts ("learn to dualize" hooks).

* project_balanced(lam)          Pi: unweighted projection onto Lambda = {sum_b lambda_b = 0}
* zero_multipliers(m, n)         the conventional start
* lp_dual_multipliers(...)       problem-informed, non-learning warm start from the LP relaxation
                                 of the bundle reformulation (duals of the consensus constraints)
* any external predictor (e.g. a trained GNN) just needs to return an (m x n) array; pass it to
  BundledLagrangian.run(candidates=[...]) and the guard of eq. (guard) selects the best start.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pulp

from .instance import SSNDInstance
from .models import add_scenario_block, get_solver


def project_balanced(lam: np.ndarray) -> np.ndarray:
    """Pi(lambda)_b = lambda_b - (1/m) sum_c lambda_c   (rows = bundles, columns = service arcs)."""
    lam = np.asarray(lam, dtype=float)
    return lam - lam.mean(axis=0, keepdims=True)


def zero_multipliers(m: int, n: int) -> np.ndarray:
    return np.zeros((m, n))


def lp_dual_multipliers(inst: SSNDInstance, bundles: Sequence[Sequence[int]], solver=None,
                        sign: float = 1.0) -> np.ndarray:
    """Multipliers from the LP relaxation of the bundle reformulation (eq. (bundleprimal)-(consensus)).

    Variables: y^b in [0,1]^{A_S} per bundle, consensus z in [0,1]^{A_S}, constraints y^b_a - z_a = 0.
    The duals pi of the consensus constraints are LP-optimal Lagrangian multipliers of the *relaxed*
    problem; we return sign * Pi(pi). Solver dual-sign conventions differ, so evaluate both signs
    through the guard (see run_example.py) rather than trusting one.
    """
    solver = solver or get_solver()
    sids = inst.service_ids
    m, n = len(bundles), len(sids)
    prob = pulp.LpProblem("SSND_bundle_LP", pulp.LpMinimize)
    z = {a: pulp.LpVariable(f"z_{a}", lowBound=0, upBound=1) for a in sids}
    obj = 0
    cons = {}
    for b, bundle in enumerate(bundles):
        P_b = sum(inst.scenarios[w].prob for w in bundle)
        y = {a: pulp.LpVariable(f"y_{b}_{a}", lowBound=0, upBound=1) for a in sids}
        obj += pulp.lpSum(P_b * inst.arcs[a].F * y[a] for a in sids)
        for a in sids:
            c = pulp.LpConstraint(y[a] - z[a], sense=pulp.LpConstraintEQ, rhs=0, name=f"na_{b}_{a}")
            prob += c
            cons[(b, a)] = c
        for w in bundle:
            s = inst.scenarios[w]
            blk = add_scenario_block(prob, inst, s, y, f"b{b}s{s.id}")
            obj += s.prob * blk.cost
    prob += obj
    prob.solve(solver)
    lam = np.zeros((m, n))
    for b in range(m):
        for j, a in enumerate(sids):
            pi = cons[(b, a)].pi
            lam[b, j] = float(pi) if pi is not None else 0.0
    return sign * project_balanced(lam)
