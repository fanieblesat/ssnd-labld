"""PuLP models for the stochastic service network design problem.

Three building blocks share one scenario block:

* solve_recourse(inst, y, scenario)       -> Q_omega(y), the second-stage LP for a fixed design
* solve_deterministic_equivalent(inst)    -> z*, y*  (extensive form MILP)
* solve_bundle_subproblem(inst, B_b, lam) -> phi_b(lambda_b), y^b  (bundle MILP of eq. (bundlesubproblem))

Scenario block (for all i in N, k in K, a in A):
    flow conservation   sum_{a in out(i)} x_ak - sum_{a in in(i)} x_ak = b_ik(d_k - u_k)
    service capacity    sum_k x_ak <= xi_a U_a y_a + r_a               (a in A_S)
    auxiliary capacity  sum_k x_ak <= cap_a                            (a in A_0, if capacitated)
    spot capacity       0 <= r_a <= rbar_a           [variant A]
                        0 <= r_a <= rbar_a y_a       [variant B, spot_requires_activation=True]
    shortage            0 <= u_k <= d_k
    cost                sum_{a,k} (c_ak + kappa e_ak) x_ak + sum_a (c_a^r + kappa e_a^r) r_a + sum_k pi_k u_k
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pulp

from .instance import Scenario, SSNDInstance

YLike = Dict[int, Union[float, pulp.LpVariable]]


# --------------------------------------------------------------------------- #
# Solver factory
# --------------------------------------------------------------------------- #
def get_solver(name: Optional[str] = None, msg: bool = False,
               time_limit: Optional[float] = None, gap_rel: Optional[float] = None,
               threads: Optional[int] = None):
    """Return a PuLP solver. `name` in {None/'CBC', 'GUROBI', 'GUROBI_CMD', 'CPLEX_CMD', 'HiGHS', ...}.

    The default is the CBC binary bundled with PuLP so that everything runs without a licence;
    swap to Gurobi/CPLEX on the host chair's machines by passing the PuLP solver name.
    """
    name = (name or "PULP_CBC_CMD").upper()
    if name in ("CBC", "PULP_CBC_CMD") and not _cbc_available():
        # PuLP >= 3 no longer ships the CBC binary on every platform: fall back to HiGHS (pip install highspy)
        warnings.warn("CBC is not available in this PuLP installation; using HiGHS instead", RuntimeWarning)
        name = "HIGHS"
    if name in ("CBC", "PULP_CBC_CMD"):
        kw = dict(msg=msg)
        if time_limit is not None:
            kw["timeLimit"] = time_limit
        if gap_rel is not None:
            kw["gapRel"] = gap_rel
        if threads is not None:
            kw["threads"] = threads
        return pulp.PULP_CBC_CMD(**kw)
    kw = dict(msg=msg)
    if time_limit is not None:
        kw["timeLimit"] = time_limit
    if gap_rel is not None:
        kw["gapRel"] = gap_rel
    if threads is not None and name in ("HIGHS", "HIGHS_CMD"):
        kw["threads"] = threads
    return pulp.getSolver("HiGHS" if name == "HIGHS" else name, **kw)


def _cbc_available() -> bool:
    cls = getattr(pulp, "PULP_CBC_CMD", None)
    if cls is None:
        return False
    try:
        return bool(cls(msg=False).available())
    except Exception:
        return False


def _mip_dual_bound(prob: pulp.LpProblem) -> Optional[float]:
    """Best proven lower bound of a MIP stopped early (HiGHS through PuLP's highspy interface).
    Returns None when the solver does not expose it."""
    m = getattr(prob, "solverModel", None)
    if m is None or not hasattr(m, "getInfo"):
        return None
    try:
        v = float(m.getInfo().mip_dual_bound)
    except Exception:
        return None
    return v if np.isfinite(v) else None


def _proved_optimal(prob: pulp.LpProblem) -> bool:
    """True only if the solver proved optimality (not merely 'a feasible solution was found')."""
    return prob.status == pulp.LpStatusOptimal and prob.sol_status == pulp.LpSolutionOptimal


# --------------------------------------------------------------------------- #
# Scenario block
# --------------------------------------------------------------------------- #
@dataclass
class ScenarioBlock:
    x: Dict[Tuple[int, int], pulp.LpVariable]
    r: Dict[int, pulp.LpVariable]
    u: Dict[int, pulp.LpVariable]
    cost: pulp.LpAffineExpression
    emissions: pulp.LpAffineExpression


def add_scenario_block(prob: pulp.LpProblem, inst: SSNDInstance, scen: Scenario,
                       y: YLike, tag: str) -> ScenarioBlock:
    """Append the recourse variables/constraints of one scenario to `prob`.

    `y` maps service-arc id -> LpVariable (design is a decision) or float (design fixed).
    """
    K = inst.commodities
    x = {(a.id, k.id): pulp.LpVariable(f"x_{tag}_{a.id}_{k.id}", lowBound=0)
         for a in inst.arcs for k in K}
    r = {a.id: pulp.LpVariable(f"r_{tag}_{a.id}", lowBound=0, upBound=a.rbar)
         for a in inst.service_arcs}
    u = {k.id: pulp.LpVariable(f"u_{tag}_{k.id}", lowBound=0, upBound=scen.demand.get(k.id, 0.0))
         for k in K}

    # flow conservation
    for node in inst.nodes:
        out_a = inst.out_arcs(node)
        in_a = inst.in_arcs(node)
        for k in K:
            d = scen.demand.get(k.id, 0.0)
            lhs = pulp.lpSum(x[(a.id, k.id)] for a in out_a) - pulp.lpSum(x[(a.id, k.id)] for a in in_a)
            if node == k.origin:
                prob += (lhs == d - u[k.id]), f"flow_{tag}_{node}_{k.id}"
            elif node == k.dest:
                prob += (lhs == -(d - u[k.id])), f"flow_{tag}_{node}_{k.id}"
            else:
                prob += (lhs == 0), f"flow_{tag}_{node}_{k.id}"

    # capacities
    for a in inst.arcs:
        load = pulp.lpSum(x[(a.id, k.id)] for k in K)
        if a.is_service:
            xi = scen.xi.get(a.id, 1.0)
            prob += (load <= xi * a.U * y[a.id] + r[a.id]), f"cap_{tag}_{a.id}"
            if inst.spot_requires_activation:
                prob += (r[a.id] <= a.rbar * y[a.id]), f"spot_{tag}_{a.id}"
        elif a.cap is not None:
            prob += (load <= a.cap), f"auxcap_{tag}_{a.id}"

    cost = (pulp.lpSum((a.cost(k.id) + inst.kappa * a.emis(k.id)) * x[(a.id, k.id)]
                       for a in inst.arcs for k in K)
            + pulp.lpSum((a.c_r + inst.kappa * a.e_r) * r[a.id] for a in inst.service_arcs)
            + pulp.lpSum(k.penalty * u[k.id] for k in K))
    emissions = (pulp.lpSum(a.emis(k.id) * x[(a.id, k.id)] for a in inst.arcs for k in K)
                 + pulp.lpSum(a.e_r * r[a.id] for a in inst.service_arcs))
    return ScenarioBlock(x=x, r=r, u=u, cost=cost, emissions=emissions)


# --------------------------------------------------------------------------- #
# Recourse LP and design evaluation
# --------------------------------------------------------------------------- #
@dataclass
class RecourseResult:
    cost: float
    emissions: float
    shortage: float
    spot: float
    demand: float
    status: str
    flows: Dict[Tuple[int, int], float] = field(default_factory=dict)
    spot_by_arc: Dict[int, float] = field(default_factory=dict)


def solve_recourse(inst: SSNDInstance, y: Dict[int, float], scen: Scenario,
                   solver=None, keep_flows: bool = False) -> RecourseResult:
    """Q_omega(y): second-stage LP for a fixed 0/1 design y (dict service-arc id -> 0/1)."""
    solver = solver or get_solver()
    prob = pulp.LpProblem(f"recourse_{scen.id}", pulp.LpMinimize)
    blk = add_scenario_block(prob, inst, scen, {a: float(y[a]) for a in inst.service_ids}, f"s{scen.id}")
    prob += blk.cost
    prob.solve(solver)
    status = pulp.LpStatus[prob.status]
    val = lambda v: float(v.value() or 0.0)
    res = RecourseResult(
        cost=float(pulp.value(prob.objective)),
        emissions=float(pulp.value(blk.emissions)),
        shortage=sum(val(v) for v in blk.u.values()),
        spot=sum(val(v) for v in blk.r.values()),
        demand=sum(scen.demand.values()),
        status=status,
    )
    if keep_flows:
        res.flows = {key: val(v) for key, v in blk.x.items() if val(v) > 1e-9}
        res.spot_by_arc = {a: val(v) for a, v in blk.r.items() if val(v) > 1e-9}
    return res


@dataclass
class DesignEvaluation:
    expected_total_cost: float      # sum F y + sum p Q
    fixed_cost: float
    expected_recourse: float
    expected_emissions: float       # EM(y)
    reliability: float              # REL(y)
    expected_spot: float
    per_scenario: List[RecourseResult]


def design_key(inst: SSNDInstance, y: Dict[int, float]) -> Tuple[int, ...]:
    """Hashable 0/1 key of a design (ordered as inst.service_ids)."""
    return tuple(int(round(float(y[a]))) for a in inst.service_ids)


def evaluate_design(inst: SSNDInstance, y: Dict[int, float], solver=None,
                    cache: Optional[Dict] = None, parallel: int = 1) -> DesignEvaluation:
    """Exact expected cost of a design: all |Omega| recourse LPs (valid primal bound if feasible).

    cache    optional dict; evaluations are stored/looked up by the 0/1 design, so a design that is
             proposed twice (very common in the local search) is solved once.
    parallel number of recourse LPs solved concurrently (CBC/HiGHS-CMD run as subprocesses).
    """
    key = design_key(inst, y)
    if cache is not None and key in cache:
        return cache[key]
    solver = solver or get_solver()
    fixed = sum(a.F * float(y[a.id]) for a in inst.service_arcs)
    if parallel > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=parallel) as ex:
            per = list(ex.map(lambda s: solve_recourse(inst, y, s, solver), inst.scenarios))
    else:
        per = [solve_recourse(inst, y, s, solver) for s in inst.scenarios]
    exp_rec = sum(s.prob * r.cost for s, r in zip(inst.scenarios, per))
    exp_em = sum(s.prob * r.emissions for s, r in zip(inst.scenarios, per))
    exp_short = sum(s.prob * r.shortage for s, r in zip(inst.scenarios, per))
    exp_dem = sum(s.prob * r.demand for s, r in zip(inst.scenarios, per))
    exp_spot = sum(s.prob * r.spot for s, r in zip(inst.scenarios, per))
    ev = DesignEvaluation(
        expected_total_cost=fixed + exp_rec, fixed_cost=fixed, expected_recourse=exp_rec,
        expected_emissions=exp_em, reliability=1.0 - (exp_short / exp_dem if exp_dem > 0 else 0.0),
        expected_spot=exp_spot, per_scenario=per)
    if cache is not None:
        cache[key] = ev
    return ev


# --------------------------------------------------------------------------- #
# Deterministic equivalent (extensive form)
# --------------------------------------------------------------------------- #
@dataclass
class DEResult:
    objective: float             # best integer solution found (= z* if proved_optimal)
    y: Dict[int, float]
    status: str
    seconds: float
    bound: Optional[float] = None    # valid lower bound on z* (== objective if proved optimal)
    proved_optimal: bool = True


def solve_deterministic_equivalent(inst: SSNDInstance, solver=None,
                                   scenarios: Optional[Sequence[Scenario]] = None,
                                   relax: bool = False) -> DEResult:
    """Extensive-form MILP over all scenarios (or the subset `scenarios`, with probabilities as given).

    relax=True solves the LP relaxation (used by the LP-dual warm start).

    If the solver stops on a time limit before proving optimality, `proved_optimal` is False,
    `objective` is the incumbent (an upper bound on z*) and `bound` is the LP-relaxation value
    (a valid, weaker lower bound on z*), so that gaps can still be reported honestly.
    """
    solver = solver or get_solver()
    scenarios = list(scenarios) if scenarios is not None else inst.scenarios
    t0 = time.time()

    def build(cat):
        prob = pulp.LpProblem("SSND_DE", pulp.LpMinimize)
        y = {a.id: pulp.LpVariable(f"y_{a.id}", lowBound=0, upBound=1, cat=cat) for a in inst.service_arcs}
        obj = pulp.lpSum(a.F * y[a.id] for a in inst.service_arcs)
        for s in scenarios:
            blk = add_scenario_block(prob, inst, s, y, f"s{s.id}")
            obj += s.prob * blk.cost
        prob += obj
        return prob, y

    prob, y = build(pulp.LpContinuous if relax else pulp.LpBinary)
    prob.solve(solver)
    status = pulp.LpStatus[prob.status]
    val = pulp.value(prob.objective)
    obj = float(val) if val is not None else float("inf")     # no incumbent within the time limit
    yv = {a: float(v.value() or 0.0) for a, v in y.items()}
    if relax:
        return DEResult(objective=obj, y=yv, status=status, seconds=time.time() - t0, bound=obj)
    yv = {a: (1.0 if v > 0.5 else 0.0) for a, v in yv.items()}
    proved = _proved_optimal(prob)
    bound = obj
    if not proved:
        # valid lower bound: the solver's MIP dual bound if exposed (HiGHS), else the LP relaxation
        mb = _mip_dual_bound(prob)
        lp, _ = build(pulp.LpContinuous)
        lp.solve(get_solver())
        bound = float(pulp.value(lp.objective))
        if mb is not None:
            bound = max(bound, mb)
    return DEResult(objective=obj, y=yv, status=status, seconds=time.time() - t0,
                    bound=bound, proved_optimal=proved)


# --------------------------------------------------------------------------- #
# Bundle subproblem  phi_b(lambda_b)
# --------------------------------------------------------------------------- #
@dataclass
class BundleResult:
    value: float                 # valid lower bound on phi_b(lambda_b) (== phi_b if proved)
    y: np.ndarray                # y^b over service arcs (0/1), ordered as inst.service_ids
    status: str
    seconds: float
    proved: bool = True          # False: time limit hit; value is then the LP-relaxation bound


def solve_bundle_subproblem(inst: SSNDInstance, bundle: Sequence[int], lam_b: np.ndarray,
                            solver=None) -> BundleResult:
    """phi_b(lambda_b) = min_{y^b, x, r, u}  P_b F^T y^b + lambda_b^T y^b + sum_{omega in B_b} p_omega c_omega

    `bundle` is a list of scenario ids; `lam_b` is a vector over inst.service_ids.
    Scenario probabilities p_omega are the *original* ones (not renormalised), so that
    L(lambda) = sum_b phi_b(lambda_b) is directly comparable with the DE objective.

    Validity under a time limit: an incumbent value is an UPPER bound on phi_b and would make
    L(lambda) an invalid dual bound. If optimality is not proved, the returned `value` is the
    LP-relaxation value of the same bundle problem (a valid lower bound on phi_b) while `y`
    stays the incumbent design (still usable as a subgradient / vote).
    """
    solver = solver or get_solver()
    t0 = time.time()
    sids = inst.service_ids
    P_b = sum(inst.scenarios[w].prob for w in bundle)

    def build(cat):
        prob = pulp.LpProblem(f"bundle_{'_'.join(map(str, bundle[:4]))}", pulp.LpMinimize)
        y = {a: pulp.LpVariable(f"y_{a}", lowBound=0, upBound=1, cat=cat) for a in sids}
        obj = pulp.lpSum((P_b * inst.arcs[a].F + float(lam_b[j])) * y[a] for j, a in enumerate(sids))
        for w in bundle:
            s = inst.scenarios[w]
            blk = add_scenario_block(prob, inst, s, y, f"s{s.id}")
            obj += s.prob * blk.cost
        prob += obj
        return prob, y

    prob, y = build(pulp.LpBinary)
    prob.solve(solver)
    yv = np.array([1.0 if (y[a].value() or 0.0) > 0.5 else 0.0 for a in sids])
    proved = _proved_optimal(prob)
    if proved:
        value = float(pulp.value(prob.objective))
    else:
        mb = _mip_dual_bound(prob)
        lp, _ = build(pulp.LpContinuous)
        lp.solve(get_solver())
        value = float(pulp.value(lp.objective))
        if mb is not None:                     # HiGHS MIP dual bound: valid and at least as strong
            value = max(value, mb)
    return BundleResult(value=value, y=yv, status=pulp.LpStatus[prob.status],
                        seconds=time.time() - t0, proved=proved)


def solve_bundle_lp(inst: SSNDInstance, bundle: Sequence[int], lam_b: Optional[np.ndarray] = None,
                    solver=None) -> Tuple[float, np.ndarray]:
    """LP relaxation of the bundle subproblem phi_b(lambda_b) (y^b in [0,1]).

    Returns (value, y_frac). Cheap compared with the MILP; used as an input feature for the learned
    multiplier predictor and as the non-learning proxy for partition selection (L_LP(0)).
    """
    solver = solver or get_solver()
    sids = inst.service_ids
    lam_b = np.zeros(len(sids)) if lam_b is None else np.asarray(lam_b, dtype=float)
    P_b = sum(inst.scenarios[w].prob for w in bundle)
    prob = pulp.LpProblem("bundle_lp", pulp.LpMinimize)
    y = {a: pulp.LpVariable(f"y_{a}", lowBound=0, upBound=1) for a in sids}
    obj = pulp.lpSum((P_b * inst.arcs[a].F + float(lam_b[j])) * y[a] for j, a in enumerate(sids))
    for w in bundle:
        s = inst.scenarios[w]
        blk = add_scenario_block(prob, inst, s, y, f"s{s.id}")
        obj += s.prob * blk.cost
    prob += obj
    prob.solve(solver)
    yv = np.array([float(y[a].value() or 0.0) for a in sids])
    return float(pulp.value(prob.objective)), yv


def solve_restricted_de(inst: SSNDInstance, allowed, solver=None) -> DEResult:
    """Primal polishing: the deterministic equivalent with every service outside `allowed` fixed
    closed. Any feasible solution is feasible for the full problem, so its objective is a valid
    upper bound on z*; the bound reported is the restricted problem's own (not a bound on z*)."""
    solver = solver or get_solver()
    allowed = set(allowed)
    t0 = time.time()
    prob = pulp.LpProblem("SSND_restricted_DE", pulp.LpMinimize)
    y = {a.id: pulp.LpVariable(f"y_{a.id}", lowBound=0, upBound=(1 if a.id in allowed else 0),
                               cat=pulp.LpBinary) for a in inst.service_arcs}
    obj = pulp.lpSum(a.F * y[a.id] for a in inst.service_arcs)
    for s in inst.scenarios:
        blk = add_scenario_block(prob, inst, s, y, f"s{s.id}")
        obj += s.prob * blk.cost
    prob += obj
    prob.solve(solver)
    val = pulp.value(prob.objective)
    yv = {a: (1.0 if (v.value() or 0.0) > 0.5 else 0.0) for a, v in y.items()}
    return DEResult(objective=float(val) if val is not None else float("inf"), y=yv,
                    status=pulp.LpStatus[prob.status], seconds=time.time() - t0,
                    bound=None, proved_optimal=_proved_optimal(prob))
