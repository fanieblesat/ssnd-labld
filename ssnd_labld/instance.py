"""Data structures for the two-stage stochastic multimodal service network design model.

The arc set is split as recommended in the audit (section 4.3):

    A = A_S  (scheduled *service* arcs, with activation y_a, fixed cost F_a,
              nominal capacity U_a, availability xi_a^omega, spot capacity r_a)
      + A_0  (auxiliary arcs: transfer / storage / waiting; no activation
              decision, optional fixed capacity, never disrupted)

Costs c_ak and emissions e_ak may be commodity-specific; scalars are broadcast.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

Number = Union[int, float]


@dataclass
class Arc:
    id: int
    tail: str
    head: str
    is_service: bool
    mode: str = "aux"
    F: float = 0.0                 # fixed activation cost (service arcs)
    U: float = 0.0                 # nominal capacity (service arcs)
    cap: Optional[float] = None    # fixed capacity for auxiliary arcs (None = uncapacitated)
    c: Union[float, Dict[int, float]] = 0.0   # unit routing cost (scalar or per commodity)
    e: Union[float, Dict[int, float]] = 0.0   # unit CO2e (scalar or per commodity)
    rbar: float = 0.0              # max spot / emergency capacity (service arcs)
    c_r: float = 0.0               # unit cost of spot capacity
    e_r: float = 0.0               # optional emissions surcharge per unit of spot capacity (audit 4.2)
    dist: float = 1.0              # distance (used by the generator / GNN features)

    def cost(self, k: int) -> float:
        return self.c[k] if isinstance(self.c, dict) else float(self.c)

    def emis(self, k: int) -> float:
        return self.e[k] if isinstance(self.e, dict) else float(self.e)


@dataclass
class Commodity:
    id: int
    origin: str
    dest: str
    penalty: float                 # pi_k, cost per unit of unserved demand
    nominal_demand: float = 0.0


@dataclass
class Scenario:
    id: int
    prob: float
    demand: Dict[int, float]       # k -> d_k^omega
    xi: Dict[int, float]           # service arc id -> availability in [0,1]
    regime: str = "nominal"        # generator label (nominal / localized / corridor / severe)


@dataclass
class SSNDInstance:
    name: str
    nodes: List[str]
    arcs: List[Arc]
    commodities: List[Commodity]
    scenarios: List[Scenario]
    kappa: float = 0.0                       # carbon valuation applied to emissions
    spot_requires_activation: bool = False   # audit 4.1: variant B if True
    meta: Dict = field(default_factory=dict)

    # ---- convenience views -------------------------------------------------
    @property
    def service_arcs(self) -> List[Arc]:
        return [a for a in self.arcs if a.is_service]

    @property
    def aux_arcs(self) -> List[Arc]:
        return [a for a in self.arcs if not a.is_service]

    @property
    def service_ids(self) -> List[int]:
        return [a.id for a in self.arcs if a.is_service]

    def out_arcs(self, node: str) -> List[Arc]:
        return [a for a in self.arcs if a.tail == node]

    def in_arcs(self, node: str) -> List[Arc]:
        return [a for a in self.arcs if a.head == node]

    def validate(self) -> None:
        p = sum(s.prob for s in self.scenarios)
        assert abs(p - 1.0) < 1e-9, f"scenario probabilities sum to {p}, not 1"
        ids = [a.id for a in self.arcs]
        assert ids == list(range(len(ids))), "arc ids must be 0..|A|-1 in order"
        kids = [k.id for k in self.commodities]
        assert kids == list(range(len(kids))), "commodity ids must be 0..|K|-1 in order"
        for s in self.scenarios:
            for a in self.service_arcs:
                assert 0.0 <= s.xi.get(a.id, 1.0) <= 1.0
            for k in self.commodities:
                assert s.demand.get(k.id, 0.0) >= 0.0

    def summary(self) -> str:
        return (f"{self.name}: |N|={len(self.nodes)} |A_S|={len(self.service_arcs)} "
                f"|A_0|={len(self.aux_arcs)} |K|={len(self.commodities)} "
                f"|Omega|={len(self.scenarios)} kappa={self.kappa} "
                f"spot_requires_activation={self.spot_requires_activation}")
