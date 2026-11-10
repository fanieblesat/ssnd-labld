"""Synthetic multimodal time-space instance generator with joint demand/disruption scenarios.

Structure (deliberately simple and fully reproducible):

* A linear corridor of `n_terminals` terminals and `n_periods` time periods.
  Node (i, t) is terminal i at period t.
* Service arcs: for each mode in `modes`, each adjacent terminal pair (both
  directions) and each departure period t with t + travel_time <= n_periods-1.
  Modes differ in travel time, capacity, fixed cost, unit cost and emissions
  (road: fast/expensive/dirty, rail: medium, barge: slow/cheap/clean).
* Auxiliary holding arcs (i, t) -> (i, t+1): uncapacitated, small holding cost,
  never disrupted, no activation decision (audit 4.3).
* Commodities: OD pairs with origin node (o, 0) and destination (d, n_periods-1).
* Scenarios omega = (d^omega, xi^omega) are generated *jointly*: demand from a
  correlated log-normal model (one common factor + idiosyncratic noise), and
  capacity availability xi from disruption regimes
      nominal | localized | corridor | severe
  so that severe disruptions hit groups of services rather than isolated arcs.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .instance import Arc, Commodity, Scenario, SSNDInstance

MODE_PARAMS = {
    # travel_time (periods), capacity, fixed cost, unit cost, unit CO2e, spot cost multiplier
    "road":  dict(tt=1, U=40.0,  F=60.0,  c=2.0, e=1.00, spot=1.6),
    "rail":  dict(tt=2, U=120.0, F=180.0, c=0.9, e=0.30, spot=1.9),
    "barge": dict(tt=3, U=220.0, F=260.0, c=0.5, e=0.12, spot=2.3),
}

REGIME_PROBS = {"nominal": 0.45, "localized": 0.25, "corridor": 0.20, "severe": 0.10}
# stressed setting for the resilience analysis (more corridor-wide and severe disruptions)
REGIME_PROBS_STRESSED = {"nominal": 0.20, "localized": 0.25, "corridor": 0.30, "severe": 0.25}
# spot_emission_factor: emergency (spot) capacity emits this multiple of road emissions per unit
# and distance on top of the flow's own emissions (0 = previous behaviour, used in all other studies)


def _node(i: int, t: int) -> str:
    return f"T{i}_t{t}"


def generate_instance(
    n_terminals: int = 4,
    n_periods: int = 4,
    modes: Optional[List[str]] = None,
    n_commodities: int = 6,
    n_scenarios: int = 12,
    demand_cv: float = 0.35,
    demand_corr: float = 0.5,
    kappa: float = 0.0,
    spot_fraction: float = 0.5,
    penalty_multiplier: float = 8.0,
    holding_cost: float = 0.05,
    spot_requires_activation: bool = False,
    regime_probs: Optional[Dict[str, float]] = None,
    spot_emission_factor: float = 0.0,
    seed: int = 0,
    name: Optional[str] = None,
) -> SSNDInstance:
    rng = np.random.default_rng(seed)
    modes = modes or ["road", "rail", "barge"]
    regime_probs = regime_probs or REGIME_PROBS

    nodes = [_node(i, t) for i in range(n_terminals) for t in range(n_periods)]
    arcs: List[Arc] = []
    aid = 0

    # --- service arcs -------------------------------------------------------
    for mode in modes:
        mp = MODE_PARAMS[mode]
        for i in range(n_terminals - 1):
            for (u, v) in ((i, i + 1), (i + 1, i)):
                for t in range(n_periods - mp["tt"]):
                    dist = 1.0 + 0.2 * rng.random()   # segment length (arbitrary units)
                    arcs.append(Arc(
                        id=aid, tail=_node(u, t), head=_node(v, t + mp["tt"]),
                        is_service=True, mode=mode,
                        F=mp["F"] * dist, U=mp["U"],
                        c=mp["c"] * dist, e=mp["e"] * dist,
                        rbar=spot_fraction * mp["U"], c_r=mp["spot"] * mp["c"] * dist,
                        e_r=spot_emission_factor * MODE_PARAMS["road"]["e"] * dist, dist=dist,
                    ))
                    aid += 1

    # --- auxiliary holding arcs ---------------------------------------------
    for i in range(n_terminals):
        for t in range(n_periods - 1):
            arcs.append(Arc(id=aid, tail=_node(i, t), head=_node(i, t + 1),
                            is_service=False, mode="hold", cap=None,
                            c=holding_cost, e=0.0, dist=0.0))
            aid += 1

    # --- commodities --------------------------------------------------------
    max_unit_cost = max(a.cost(0) for a in arcs if a.is_service) * (n_terminals - 1)
    commodities: List[Commodity] = []
    pairs = [(o, d) for o in range(n_terminals) for d in range(n_terminals) if o != d]
    rng.shuffle(pairs)
    for k in range(n_commodities):
        o, d = pairs[k % len(pairs)]
        nominal = float(rng.uniform(15.0, 45.0))
        commodities.append(Commodity(
            id=k, origin=_node(o, 0), dest=_node(d, n_periods - 1),
            penalty=penalty_multiplier * max_unit_cost, nominal_demand=nominal))

    # --- scenarios (joint demand + disruption) ------------------------------
    service_ids = [a.id for a in arcs if a.is_service]
    regimes = list(regime_probs.keys())
    rprobs = np.array([regime_probs[r] for r in regimes], dtype=float)
    rprobs /= rprobs.sum()
    sigma = np.sqrt(np.log(1.0 + demand_cv ** 2))     # log-normal with given CV
    scenarios: List[Scenario] = []
    for w in range(n_scenarios):
        common = rng.normal()
        demand = {}
        for k in commodities:
            z = np.sqrt(demand_corr) * common + np.sqrt(1.0 - demand_corr) * rng.normal()
            demand[k.id] = float(k.nominal_demand * np.exp(sigma * z - 0.5 * sigma ** 2))
        regime = regimes[int(rng.choice(len(regimes), p=rprobs))]
        xi = {a: 1.0 for a in service_ids}
        if regime == "localized":
            hit = rng.choice(service_ids, size=max(1, len(service_ids) // 12), replace=False)
            for a in hit:
                xi[int(a)] = float(rng.choice([0.0, 0.5]))
        elif regime == "corridor":
            mode = rng.choice(modes)
            seg = int(rng.integers(0, n_terminals - 1))
            for a in arcs:
                if a.is_service and a.mode == mode and (
                        a.tail.startswith(f"T{seg}_") or a.tail.startswith(f"T{seg + 1}_")):
                    xi[a.id] = float(rng.uniform(0.0, 0.5))
        elif regime == "severe":
            for a in service_ids:
                if rng.random() < 0.4:
                    xi[a] = float(rng.uniform(0.0, 0.6))
        scenarios.append(Scenario(id=w, prob=1.0 / n_scenarios, demand=demand, xi=xi, regime=regime))

    inst = SSNDInstance(
        name=name or f"corridor_T{n_terminals}_P{n_periods}_K{n_commodities}_S{n_scenarios}_seed{seed}",
        nodes=nodes, arcs=arcs, commodities=commodities, scenarios=scenarios,
        kappa=kappa, spot_requires_activation=spot_requires_activation,
        meta=dict(seed=seed, modes=modes, demand_cv=demand_cv, demand_corr=demand_corr),
    )
    inst.validate()
    return inst
