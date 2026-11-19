"""Multi-cut Benders decomposition for the SND / SSND model with continuous recourse (Gurobi).

Master:   min  F'y + sum_s p_s theta_s,   y binary,  theta_s >= 0,  Benders cuts
Cuts:     theta_s >= Q_s(y_bar) + g_s'(y - y_bar),
          where Q_s is the recourse LP value of scenario s at y_bar and g_s its subgradient in y.
          The recourse LPs are the persistent models of GurobiBackend with y fixed through its
          bounds; the reduced costs of y are a subgradient of the LP value function Q_s, which is
          convex in y, so every cut is valid for all y in [0,1]^n.

Two phases (as in the accelerated Benders literature, e.g. Rahmaniani et al. 2017):
  1. Kelley cutting planes on the LP relaxation of the master (fractional y) until the bound
     stalls or the phase-1 time share is used: cheap, strong cuts at the root.
  2. Branch-and-cut on the master with lazy constraints: at every integer candidate, all
     scenario LPs are solved (in parallel) and violated cuts are added. The full-set cost of each
     candidate is an upper bound, so the incumbent is tracked independently of the master's theta.

Optional "expected-value" strengthening (ev_block=True): the recourse block of the mean-demand
scenario is added to the master with sum_s p_s theta_s >= cost of that block. Valid because the
recourse value is convex in the demand (right-hand side), so Q(y, E d) <= E Q(y, d) (Jensen).
The lower bound is Gurobi's ObjBound of the master, valid also when the time limit stops the search.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .grb_backend import GurobiBackend, _new_env
from .instance import Scenario, SSNDInstance

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:          # pragma: no cover
    gp = None


@dataclass
class BendersResult:
    lb: float
    ub: float
    y: Optional[Dict[int, float]]
    seconds: float
    proved_optimal: bool
    lp_bound: float                      # master LP bound after phase 1
    phase1_rounds: int
    cuts: int
    candidates: int                      # integer candidates evaluated in phase 2
    history: List[Dict] = field(default_factory=list)

    @property
    def gap(self) -> float:
        return (self.ub - self.lb) / max(abs(self.ub), 1e-9) if np.isfinite(self.ub) else np.inf


def _recourse_and_subgradients(be: GurobiBackend, yv: np.ndarray):
    """Q_s(yv) and subgradients g_s (S x n) from the persistent recourse LPs (parallel threads)."""
    from concurrent.futures import ThreadPoolExecutor
    S, n = len(be.rmodels), len(yv)
    Q, G = np.zeros(S), np.zeros((S, n))

    def work(w):
        for i in range(w, S, be.P):
            m, yy, _, _ = be.rmodels[i]
            yy.LB = yv
            yy.UB = yv
            m.optimize()
            Q[i] = m.ObjVal
            G[i] = yy.RC

    with ThreadPoolExecutor(max_workers=be.P) as ex:
        list(ex.map(work, range(be.P)))
    return Q, G


def solve_benders(inst: SSNDInstance, time_limit: float = 1800.0, threads: int = 8,
                  mip_gap: float = 1e-4, strong_linking: bool = True, ev_block: bool = True,
                  phase1_share: float = 0.25, phase1_max_rounds: int = 200, phase1_tol: float = 1e-5,
                  cut_tol: float = 1e-6, backend: Optional[GurobiBackend] = None,
                  verbose: bool = False) -> BendersResult:
    if gp is None:
        raise ImportError("gurobipy is required")
    t0 = time.time()
    deadline = t0 + time_limit
    own_be = backend is None
    be = backend or GurobiBackend(inst, parallel=threads, strong_linking=strong_linking)
    T = be.T
    S, n = len(inst.scenarios), len(T.sids)
    p = np.array([s.prob for s in inst.scenarios])

    env = _new_env(threads)
    m = gp.Model(env=env)
    y = m.addMVar(n, lb=0.0, ub=1.0, obj=T.F)                 # continuous in phase 1
    th = m.addMVar(S, lb=0.0, obj=p)
    if ev_block:
        dbar = {k.id: float(np.mean([s.demand.get(k.id, 0.0) for s in inst.scenarios]))
                for k in inst.commodities}
        sbar = Scenario(id=-1, prob=1.0, demand=dbar, xi={})
        q, u = T.add_scenario(m, sbar, y, 0.0)
        m.addConstr(p @ th >= (T.cost * q).sum() + T.pen @ u)
    m.update()

    ncuts = 0
    best_ub, best_y = np.inf, None
    history: List[Dict] = []

    ylist, thlist = y.tolist(), th.tolist()

    def add_cuts(yv, Q, G, theta_val, adder):
        nonlocal ncuts
        added = 0
        rhs0 = Q - G @ yv
        for s in np.nonzero(theta_val < Q - cut_tol * np.maximum(1.0, np.abs(Q)))[0]:
            nz = np.nonzero(np.abs(G[s]) > 1e-12)[0]
            expr = gp.LinExpr([1.0] + [-float(G[s, j]) for j in nz], [thlist[s]] + [ylist[j] for j in nz])
            adder(expr >= float(rhs0[s]))
            added += 1
        ncuts += added
        return added

    # ---------------- phase 1: Kelley on the LP relaxation
    lp_bound, rounds, stall = -np.inf, 0, 0
    t1_end = t0 + phase1_share * time_limit
    while rounds < phase1_max_rounds and time.time() < t1_end:
        m.Params.TimeLimit = max(1.0, t1_end - time.time())
        m.optimize()
        if m.Status != GRB.OPTIMAL:
            break
        rounds += 1
        yv, tv, val = y.X.copy(), th.X.copy(), m.ObjVal
        Q, G = _recourse_and_subgradients(be, yv)
        if add_cuts(yv, Q, G, tv, m.addConstr) == 0:
            lp_bound = val
            break
        stall = stall + 1 if val <= lp_bound + phase1_tol * max(1.0, abs(val)) else 0
        lp_bound = max(lp_bound, val)
        history.append(dict(phase=1, round=rounds, lb=lp_bound, ub=best_ub, time=time.time() - t0))
        if verbose:
            print(f"  benders LP round {rounds}: bound {val:.2f}, cuts {ncuts}", flush=True)
        if stall >= 5:
            break
    if not np.isfinite(lp_bound):
        lp_bound = 0.0

    # ---------------- phase 2: branch-and-cut with lazy cuts
    y.VType = GRB.BINARY
    m.Params.LazyConstraints = 1
    m.Params.MIPGap = mip_gap
    m.Params.TimeLimit = max(1.0, deadline - time.time())
    state = dict(cands=0)

    def cb(model, where):
        nonlocal best_ub, best_y
        if where != GRB.Callback.MIPSOL:
            return
        yv = np.round(np.array(model.cbGetSolution(ylist)))
        tv = np.array(model.cbGetSolution(thlist))
        Q, G = _recourse_and_subgradients(be, yv)
        state["cands"] += 1
        full = float(T.F @ yv + p @ Q)
        if full < best_ub:
            best_ub, best_y = full, yv.copy()
            history.append(dict(phase=2, round=state["cands"],
                                lb=model.cbGet(GRB.Callback.MIPSOL_OBJBND), ub=best_ub,
                                time=time.time() - t0))
        add_cuts(yv, Q, G, tv, model.cbLazy)

    if verbose:
        m.Params.OutputFlag = 1
    m.optimize(cb)
    lb = float(m.ObjBound) if m.SolCount > 0 or m.Status == GRB.OPTIMAL else lp_bound
    lb = max(lb, lp_bound)
    proved = m.Status == GRB.OPTIMAL
    if m.SolCount > 0:                                   # master incumbent: evaluate it exactly as well
        yv = np.round(y.X)
        full = float(T.F @ yv + p @ _recourse_and_subgradients(be, yv)[0])
        if full < best_ub:
            best_ub, best_y = full, yv
    m.dispose(); env.dispose()
    if own_be:
        be.close()
    yd = {a: float(v) for a, v in zip(T.sids, best_y)} if best_y is not None else None
    return BendersResult(lb=lb, ub=best_ub, y=yd, seconds=time.time() - t0, proved_optimal=proved,
                         lp_bound=lp_bound, phase1_rounds=rounds, cuts=ncuts, candidates=state["cands"],
                         history=history)
