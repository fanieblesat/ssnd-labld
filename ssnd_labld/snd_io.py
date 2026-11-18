"""Reader for the SND_V1 instance format (multicommodity network design with stochastic demand).

The format and model are those of the IDSR benchmark (instances/snd/snd_format.txt):
arcs are design arcs with capacity U_a, unit cost c_a and fixed cost f_a; commodities have an
origin, a destination, and a per-unit penalty for unmet demand; scenarios are equiprobable demand
vectors. The model is the special case of our SSND model with service arcs only, no spot capacity,
no capacity disruptions and kappa = 0, so the instance maps one-to-one onto SSNDInstance.
"""
from __future__ import annotations

import math
from typing import Dict, List

from .instance import Arc, Commodity, Scenario, SSNDInstance


def read_snd(path: str) -> SSNDInstance:
    with open(path) as f:
        tok = f.read().split()
    pos = 0

    def nxt() -> str:
        nonlocal pos
        pos += 1
        return tok[pos - 1]

    def expect(word: str) -> None:
        w = nxt()
        if w != word:
            raise ValueError(f"{path}: expected '{word}' at token {pos}, found '{w}'")

    expect("SND_V1")
    meta: Dict = {"format": "SND_V1", "source": path}
    expect("label"); meta["label"] = nxt()
    expect("seed"); meta["seed"] = int(nxt())
    expect("factors"); meta["factors"] = int(nxt())
    expect("correlation"); meta["rho"] = float(nxt())
    expect("cv"); meta["cv"] = float(nxt())

    expect("nodes"); V = int(nxt())
    coords = {}
    for _ in range(V):
        expect("node"); i = int(nxt()); coords[i] = (float(nxt()), float(nxt()))
    nodes = [str(i) for i in range(V)]

    expect("arcs"); A = int(nxt())
    arcs: List[Arc] = []
    for _ in range(A):
        expect("arc")
        aid, tail, head = int(nxt()), int(nxt()), int(nxt())
        U, c, F = float(nxt()), float(nxt()), float(nxt())
        (x1, y1), (x2, y2) = coords[tail], coords[head]
        arcs.append(Arc(id=aid, tail=str(tail), head=str(head), is_service=True, mode="arc",
                        F=F, U=U, c=c, e=0.0, rbar=0.0, c_r=0.0, dist=math.hypot(x2 - x1, y2 - y1)))

    expect("commodities"); C = int(nxt())
    comms: List[Commodity] = []
    factor_of = {}
    for _ in range(C):
        expect("commodity")
        kid, o, t = int(nxt()), int(nxt()), int(nxt())
        base, fac, pen = float(nxt()), int(nxt()), float(nxt())
        comms.append(Commodity(id=kid, origin=str(o), dest=str(t), penalty=pen, nominal_demand=base))
        factor_of[kid] = fac
    meta["factor_of"] = factor_of

    expect("scenarios"); N = int(nxt()); expect("equiprobable")
    scens: List[Scenario] = []
    for i in range(N):
        expect("scenario"); sid = int(nxt())
        if sid != i + 1:
            raise ValueError(f"{path}: scenario ids must be 1..N in order (found {sid} at position {i + 1})")
        dem = {k: float(nxt()) for k in range(C)}
        scens.append(Scenario(id=i, prob=1.0 / N, demand=dem, xi={}))
    if pos != len(tok):
        raise ValueError(f"{path}: {len(tok) - pos} unread tokens at the end of the file")

    inst = SSNDInstance(name=meta["label"], nodes=nodes, arcs=arcs, commodities=comms, scenarios=scens,
                        kappa=0.0, spot_requires_activation=False, meta=meta)
    inst.validate()
    return inst


def write_snd(inst: SSNDInstance, path: str, seed: int = 0) -> None:
    """Write an SSNDInstance with service arcs only in SND_V1 format (used by the tests)."""
    with open(path, "w") as f:
        f.write("SND_V1\n")
        f.write(f"label {inst.name}\nseed {seed}\nfactors 1 correlation 0 cv 0\n")
        f.write(f"nodes {len(inst.nodes)}\n")
        for i, _ in enumerate(inst.nodes):
            f.write(f"node {i} {float(i)} 0\n")
        idx = {n: i for i, n in enumerate(inst.nodes)}
        f.write(f"arcs {len(inst.arcs)}\n")
        for a in inst.arcs:
            f.write(f"arc {a.id} {idx[a.tail]} {idx[a.head]} {a.U} {a.cost(0)} {a.F}\n")
        f.write(f"commodities {len(inst.commodities)}\n")
        for k in inst.commodities:
            f.write(f"commodity {k.id} {idx[k.origin]} {idx[k.dest]} {k.nominal_demand} 0 {k.penalty}\n")
        f.write(f"scenarios {len(inst.scenarios)} equiprobable\n")
        for i, s in enumerate(inst.scenarios):
            f.write(f"scenario {i + 1} " + " ".join(str(s.demand[k.id]) for k in inst.commodities) + "\n")
