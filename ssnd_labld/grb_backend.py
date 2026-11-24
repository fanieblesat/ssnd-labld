"""Persistent Gurobi models for large scenario sets.

With hundreds or thousands of scenarios, rebuilding every bundle MILP and every recourse LP through
PuLP in each iteration dominates the run time (and is serialised by Python's GIL). This backend
builds each model once with gurobipy's matrix API and only updates it afterwards:

  * bundle MILPs: the objective coefficients of y^b (P_b F + lambda_b) change between iterations;
  * recourse LPs: one LP per scenario, with the design y fixed through variable bounds.

Workers are threads, each with its own Gurobi environment and its own share of the models, so the
models solve in parallel (gurobipy releases the GIL while optimizing). Bundle values are Gurobi's
ObjBound, a valid lower bound on phi_b also when the solve stops at the gap or time limit.

Supports the SSND model with service arcs (design y), optional uncapacitated or capacitated
auxiliary arcs, capacity availabilities xi, unmet demand, and kappa = 0 or > 0. Spot capacity
(rbar > 0) is not supported by this backend.

Optional strengthening (as in the IDSR benchmark): q_aks <= min(xi U_a, d_ks) y_a.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import scipy.sparse as sp

from .instance import SSNDInstance
from .models import DesignEvaluation, DEResult, RecourseResult, design_key

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:          # pragma: no cover
    gp = None


def _new_env(threads: int = 1):
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Threads", threads)
    env.start()
    return env


class _Template:
    """Instance data in matrix form, shared by all models."""

    def __init__(self, inst: SSNDInstance, strong_linking: bool):
        if any(a.rbar > 0 for a in inst.service_arcs):
            raise NotImplementedError("grb_backend does not support spot capacity (rbar > 0)")
        self.inst = inst
        self.strong = strong_linking
        self.A = len(inst.arcs)
        self.K = len(inst.commodities)
        self.sids = inst.service_ids
        self.svc = np.array(self.sids, dtype=int)
        node_idx = {n: i for i, n in enumerate(inst.nodes)}
        rows, cols, vals = [], [], []
        for a in inst.arcs:
            rows += [node_idx[a.tail], node_idx[a.head]]
            cols += [a.id, a.id]
            vals += [1.0, -1.0]
        self.inc = sp.csr_matrix((vals, (rows, cols)), shape=(len(inst.nodes), self.A))
        # net supply direction e_o - e_t per commodity
        self.ot = np.zeros((self.K, len(inst.nodes)))
        for k in inst.commodities:
            self.ot[k.id, node_idx[k.origin]] += 1.0
            self.ot[k.id, node_idx[k.dest]] -= 1.0
        self.cost = np.array([[a.cost(k.id) + inst.kappa * a.emis(k.id) for k in inst.commodities]
                              for a in inst.arcs])                      # A x K
        self.U = np.array([a.U for a in inst.arcs])
        self.F = np.array([inst.arcs[a].F for a in self.sids])
        self.pen = np.array([k.penalty for k in inst.commodities])
        # all cost coefficients nonnegative => every bundle value at lambda = 0 is >= 0 (used as a fallback bound)
        self.nonneg = bool((self.F >= 0).all() and (self.cost >= 0).all() and (self.pen >= 0).all())
        self.aux_cap = [(a.id, a.cap) for a in inst.arcs if not a.is_service and a.cap is not None]

    def add_scenario(self, m, s, y, weight: float):
        """Add the recourse block of scenario s to model m; y is an MVar over service arcs."""
        A, K, inst = self.A, self.K, self.inst
        d = np.array([s.demand.get(k.id, 0.0) for k in inst.commodities])
        q = m.addMVar((A, K), lb=0.0, obj=weight * self.cost)
        u = m.addMVar(K, lb=0.0, ub=d, obj=weight * self.pen)
        for k in range(K):
            if d[k] > 0:
                m.addConstr(self.inc @ q[:, k] + self.ot[k] * u[k] == self.ot[k] * d[k])
            else:
                m.addConstr(self.inc @ q[:, k] == 0.0)
        xi = np.array([s.xi.get(a, 1.0) for a in self.sids])
        capU = xi * self.U[self.svc]
        m.addConstr(q[self.svc, :].sum(axis=1) - np.diag(capU) @ y <= 0.0)
        for aid, cap in self.aux_cap:
            m.addConstr(q[aid, :].sum() <= cap)
        if self.strong:
            for k in range(K):
                if d[k] > 0:
                    m.addConstr(q[self.svc, k] - np.diag(np.minimum(capU, d[k])) @ y <= 0.0)
                else:
                    m.addConstr(q[self.svc, k] == 0.0)
        return q, u


@dataclass
class _BundleModel:
    model: object
    y: object
    P: float


class GurobiBackend:
    """Persistent bundle MILPs and recourse LPs for BundledLagrangian(backend=...)."""

    def __init__(self, inst: SSNDInstance, bundles: Optional[Sequence[Sequence[int]]] = None,
                 parallel: int = 4, bundle_time_limit: Optional[float] = None, bundle_gap: float = 1e-6,
                 strong_linking: bool = True, build_recourse: bool = True, verbose: bool = False):
        if gp is None:
            raise ImportError("gurobipy is required for GurobiBackend")
        t0 = time.time()
        self.T = _Template(inst, strong_linking)
        self.inst = inst
        self.P = max(1, parallel)
        self.verbose = verbose
        self.bundle_time_limit, self.bundle_gap = bundle_time_limit, bundle_gap
        self.envs = [_new_env(1) for _ in range(self.P)]
        n = len(self.T.sids)
        self.rmodels = []
        if build_recourse:
            for i, s in enumerate(inst.scenarios):
                m = gp.Model(env=self.envs[i % self.P])
                y = m.addMVar(n, lb=0.0, ub=1.0)
                q, u = self.T.add_scenario(m, s, y, 1.0)
                m.update()
                self.rmodels.append((m, y, u, float(sum(s.demand.values()))))
        self.recourse_build_seconds = time.time() - t0
        if verbose:
            print(f"  GurobiBackend: {len(self.rmodels)} recourse LPs built in {self.recourse_build_seconds:.1f}s",
                  flush=True)
        self.bundles, self.bmodels = [], []
        if bundles is not None:
            self.set_bundles(bundles)

    def set_bundles(self, bundles: Sequence[Sequence[int]]):
        """(Re)build the persistent bundle MILPs for a partition of the scenarios."""
        t0 = time.time()
        for bm in self.bmodels:
            bm.model.dispose()
        self.bundles = [list(b) for b in bundles]
        n = len(self.T.sids)
        self.bmodels = []
        for b, bundle in enumerate(self.bundles):
            m = gp.Model(env=self.envs[b % self.P])
            y = m.addMVar(n, vtype=GRB.BINARY)
            for w in bundle:
                self.T.add_scenario(m, self.inst.scenarios[w], y, self.inst.scenarios[w].prob)
            m.Params.MIPGap = self.bundle_gap
            if self.bundle_time_limit is not None:
                m.Params.TimeLimit = self.bundle_time_limit
            m.update()
            self.bmodels.append(_BundleModel(m, y, sum(self.inst.scenarios[w].prob for w in bundle)))
        self.bundle_build_seconds = time.time() - t0
        if self.verbose:
            print(f"  GurobiBackend: {len(self.bmodels)} bundle MILPs built in {self.bundle_build_seconds:.1f}s",
                  flush=True)

    # ------------------------------------------------------------------ bundles
    def solve_bundles(self, lam: np.ndarray, deadline: Optional[float] = None, partial_ok: bool = False):
        """Returns (values, ys, proved, complete): values[b] is a valid lower bound on phi_b(lam_b).

        With a deadline (absolute time.time()), each worker splits its remaining time equally over its
        remaining bundles (capped by bundle_time_limit), so a round of bundle solves ends by the deadline.
        A bundle that gets no time or no finite bound makes the result incomplete, unless partial_ok is set
        and lam = 0 on that bundle with nonnegative costs: then 0 is a valid bound for it (zero start only)."""
        m = len(self.bmodels)
        vals, ys, proved = np.zeros(m), np.zeros((m, len(self.T.sids))), np.zeros(m, dtype=bool)
        ok = np.ones(m, dtype=bool)
        zero_ok = partial_ok and self.T.nonneg

        def missing(b):
            if zero_ok and not np.any(lam[b]):
                vals[b] = 0.0
            else:
                ok[b] = False

        def work(w):
            mine = list(range(w, m, self.P))
            for k, b in enumerate(mine):
                bm = self.bmodels[b]
                if deadline is not None:
                    rem = deadline - time.time()
                    share = rem / (len(mine) - k)
                    if rem <= 0.5 or share <= 0.5:
                        missing(b)
                        continue
                    lim = share if self.bundle_time_limit is None else min(self.bundle_time_limit, share)
                    bm.model.Params.TimeLimit = lim
                bm.y.Obj = bm.P * self.T.F + lam[b]
                bm.model.optimize()
                bound = bm.model.ObjBound
                if not np.isfinite(bound) or abs(bound) >= GRB.INFINITY:
                    missing(b)
                    continue
                vals[b] = bound
                ys[b] = (bm.y.X > 0.5).astype(float) if bm.model.SolCount > 0 else 0.0
                proved[b] = bm.model.Status == GRB.OPTIMAL
                if bm.model.SolCount == 0 and not zero_ok:
                    ok[b] = False

        with ThreadPoolExecutor(max_workers=self.P) as ex:
            list(ex.map(work, range(self.P)))
        return vals, ys, proved, bool(ok.all())

    def resolve_bundles(self, idx: Sequence[int], deadline: Optional[float] = None):
        """Give the bundle MILPs in idx more time at their current objective (no model change).

        Gurobi may resume the search or restart it; either way the returned ObjBound is a valid bound, and
        the caller keeps the larger of the old and the new one. Returns (vals, ys, proved) for idx, with
        vals[i] = -inf where no bound could be obtained before the deadline."""
        idx = list(idx)
        vals = np.full(len(idx), -np.inf)
        ys = np.zeros((len(idx), len(self.T.sids)))
        proved = np.zeros(len(idx), dtype=bool)

        def work(w):
            for j in range(w, len(idx), self.P):
                bm = self.bmodels[idx[j]]
                lim = GRB.INFINITY if self.bundle_time_limit is None else self.bundle_time_limit
                if deadline is not None:
                    rem = deadline - time.time()
                    if rem <= 0.5:
                        continue
                    lim = min(lim, rem)
                bm.model.Params.TimeLimit = lim
                bm.model.optimize()
                bound = bm.model.ObjBound
                if np.isfinite(bound) and abs(bound) < GRB.INFINITY:
                    vals[j] = bound
                if bm.model.SolCount > 0:
                    ys[j] = (bm.y.X > 0.5).astype(float)
                proved[j] = bm.model.Status == GRB.OPTIMAL

        with ThreadPoolExecutor(max_workers=self.P) as ex:
            list(ex.map(work, range(self.P)))
        return vals, ys, proved

    # ------------------------------------------------------------------ designs
    def evaluate_design(self, y: Dict[int, float], cache: Optional[Dict] = None) -> DesignEvaluation:
        key = design_key(self.inst, y)
        if cache is not None and key in cache:
            return cache[key]
        yv = np.array([float(y[a]) for a in self.T.sids])
        S = len(self.rmodels)
        cost, short, dem = np.zeros(S), np.zeros(S), np.zeros(S)

        def work(w):
            for i in range(w, S, self.P):
                m, yy, u, d = self.rmodels[i]
                yy.LB = yv
                yy.UB = yv
                m.optimize()
                cost[i] = m.ObjVal
                short[i] = float(u.X.sum())
                dem[i] = d

        with ThreadPoolExecutor(max_workers=self.P) as ex:
            list(ex.map(work, range(self.P)))
        p = np.array([s.prob for s in self.inst.scenarios])
        fixed = float(self.T.F @ yv)
        per = [RecourseResult(cost=float(cost[i]), emissions=0.0, shortage=float(short[i]), spot=0.0,
                              demand=float(dem[i]), status="Optimal") for i in range(S)]
        exp_rec = float(p @ cost)
        exp_dem = float(p @ dem)
        ev = DesignEvaluation(expected_total_cost=fixed + exp_rec, fixed_cost=fixed, expected_recourse=exp_rec,
                              expected_emissions=0.0,
                              reliability=1.0 - (float(p @ short) / exp_dem if exp_dem > 0 else 0.0),
                              expected_spot=0.0, per_scenario=per)
        if cache is not None:
            cache[key] = ev
        return ev

    def scenario_costs(self, y: Dict[int, float]) -> np.ndarray:
        return np.array([r.cost for r in self.evaluate_design(y).per_scenario])

    def close(self):
        for bm in self.bmodels:
            bm.model.dispose()
        for m, *_ in self.rmodels:
            m.dispose()
        for e in self.envs:
            e.dispose()


def solve_de_gurobi(inst: SSNDInstance, time_limit: Optional[float] = None, threads: int = 4,
                    mip_gap: float = 1e-4, strong_linking: bool = True, verbose: bool = False) -> DEResult:
    """Extensive form with gurobipy (matrix API). bound = Gurobi's ObjBound (valid lower bound)."""
    if gp is None:
        raise ImportError("gurobipy is required")
    t0 = time.time()
    T = _Template(inst, strong_linking)
    env = _new_env(threads)
    m = gp.Model(env=env)
    y = m.addMVar(len(T.sids), vtype=GRB.BINARY, obj=T.F)
    for s in inst.scenarios:
        T.add_scenario(m, s, y, s.prob)
    m.Params.MIPGap = mip_gap
    if time_limit is not None:
        m.Params.TimeLimit = max(1.0, time_limit - (time.time() - t0))   # model building counts
    if verbose:
        m.Params.OutputFlag = 1
    build = time.time() - t0
    m.optimize()
    has = m.SolCount > 0
    yd = {a: (float(v > 0.5) if has else 0.0) for a, v in zip(T.sids, y.X if has else np.zeros(len(T.sids)))}
    res = DEResult(objective=float(m.ObjVal) if has else float("inf"), y=yd,
                   status=str(m.Status), seconds=time.time() - t0,
                   bound=float(m.ObjBound), proved_optimal=(m.Status == GRB.OPTIMAL))
    res.build_seconds = build
    m.dispose(); env.dispose()
    return res
