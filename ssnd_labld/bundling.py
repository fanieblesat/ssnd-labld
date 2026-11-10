"""Scenario partitions ("learn to decompose").

Bundles are lists of scenario ids; a partition is a list of bundles covering all scenarios.

* singleton_bundles            one scenario per bundle (scenario-wise Lagrangian decomposition)
* random_bundles               random balanced partition (control baseline)
* kmeans_bundles               demand--disruption bundling on the feature vector of eq. (scenfeatures)
* cost_signature_bundles       k-means on the recourse-cost vector of each scenario at a few candidate
                               designs (the cost-based reduction idea of Spieckermann & Minner, used here
                               to *bundle* rather than to discard scenarios)
* stratified_bundles           anti-clustering: cluster scenarios into similar groups (strata) and
                               deal each stratum across the bundles, so every bundle is diverse
* solution_similarity_bundles  hierarchical clustering on the Hamming distance between the
                               first-stage solutions y^omega of singleton subproblems at lambda = 0
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .instance import SSNDInstance

Bundles = List[List[int]]


def scenario_features(inst: SSNDInstance, include_mode_stats: bool = True) -> np.ndarray:
    """phi(omega) = [total demand, d_1..d_K, ||d - mean d||, mean xi, q10, q50, q90 of xi, (mode-wise mean xi)]."""
    K = [k.id for k in inst.commodities]
    S = inst.service_ids
    D = np.array([[s.demand.get(k, 0.0) for k in K] for s in inst.scenarios])
    dbar = D.mean(axis=0)
    Xi = np.array([[s.xi.get(a, 1.0) for a in S] for s in inst.scenarios])
    feats = [D.sum(axis=1, keepdims=True), D,
             np.linalg.norm(D - dbar, axis=1, keepdims=True),
             Xi.mean(axis=1, keepdims=True),
             np.quantile(Xi, 0.10, axis=1, keepdims=True),
             np.quantile(Xi, 0.50, axis=1, keepdims=True),
             np.quantile(Xi, 0.90, axis=1, keepdims=True)]
    if include_mode_stats:
        modes = sorted({inst.arcs[a].mode for a in S})
        for m in modes:
            cols = [j for j, a in enumerate(S) if inst.arcs[a].mode == m]
            feats.append(Xi[:, cols].mean(axis=1, keepdims=True))
    return np.hstack(feats)


def standardize(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-12] = 1.0
    return (X - mu) / sd


def singleton_bundles(inst: SSNDInstance) -> Bundles:
    return [[s.id] for s in inst.scenarios]


def random_bundles(inst: SSNDInstance, m: int, seed: int = 0) -> Bundles:
    rng = np.random.default_rng(seed)
    ids = np.array([s.id for s in inst.scenarios])
    rng.shuffle(ids)
    return [sorted(int(i) for i in chunk) for chunk in np.array_split(ids, m) if len(chunk) > 0]


def _kmeans(X: np.ndarray, m: int, seed: int, n_init: int = 10, iters: int = 100) -> np.ndarray:
    try:
        from sklearn.cluster import KMeans  # type: ignore
        return KMeans(n_clusters=m, n_init=n_init, random_state=seed).fit_predict(X)
    except Exception:  # pragma: no cover - fallback without sklearn
        rng = np.random.default_rng(seed)
        best, best_inertia = None, np.inf
        for _ in range(n_init):
            C = X[rng.choice(len(X), size=m, replace=False)]
            for _ in range(iters):
                lab = np.argmin(((X[:, None, :] - C[None, :, :]) ** 2).sum(-1), axis=1)
                Cn = np.array([X[lab == j].mean(axis=0) if np.any(lab == j) else C[j] for j in range(m)])
                if np.allclose(Cn, C):
                    break
                C = Cn
            inertia = ((X - C[lab]) ** 2).sum()
            if inertia < best_inertia:
                best, best_inertia = lab, inertia
        return best


def _labels_to_bundles(labels: np.ndarray, ids: Sequence[int]) -> Bundles:
    bundles: Dict[int, List[int]] = {}
    for lab, sid in zip(labels, ids):
        bundles.setdefault(int(lab), []).append(int(sid))
    return [sorted(b) for _, b in sorted(bundles.items())]


def balance_labels(X: np.ndarray, labels: np.ndarray, max_size: Optional[int]) -> np.ndarray:
    """Capacitated reassignment: keep the cluster centroids, then assign scenario-centroid pairs in order
    of increasing distance while the cluster has room (at most `max_size` scenarios per bundle).
    Huge bundles make the bundle MILPs almost as hard as the DE; this caps them."""
    if not max_size:
        return labels
    labs = np.unique(labels)
    m = len(labs)
    if max_size * m < len(X):
        raise ValueError(f"max_size={max_size} too small for {len(X)} scenarios in {m} bundles")
    C = np.array([X[labels == l].mean(axis=0) for l in labs])
    D = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
    new = -np.ones(len(X), dtype=int)
    room = np.full(m, max_size)
    for flat in np.argsort(D, axis=None):
        i, j = divmod(int(flat), m)
        if new[i] < 0 and room[j] > 0:
            new[i], room[j] = j, room[j] - 1
    return new


def kmeans_bundles(inst: SSNDInstance, m: int, seed: int = 0, max_size: Optional[int] = None) -> Bundles:
    """Demand--disruption bundling: k-means on standardized scenario features."""
    X = standardize(scenario_features(inst))
    labels = balance_labels(X, _kmeans(X, m, seed), max_size)
    return _labels_to_bundles(labels, [s.id for s in inst.scenarios])


def solution_similarity_bundles(inst: SSNDInstance, m: int, y_hat: Optional[np.ndarray] = None,
                                solver=None, linkage: str = "average",
                                max_size: Optional[int] = None) -> Bundles:
    """Cluster scenarios by the Hamming distance between their singleton first-stage solutions.

    y_hat: optional (|Omega| x |A_S|) 0/1 matrix of singleton solutions at lambda = 0. If None, the
    singleton subproblems are solved here (this is the "warm-up pass" of the paper).
    """
    from scipy.cluster.hierarchy import fcluster, linkage as sp_linkage
    from scipy.spatial.distance import pdist
    if y_hat is None:
        from .models import solve_bundle_subproblem
        zero = np.zeros(len(inst.service_ids))
        y_hat = np.array([solve_bundle_subproblem(inst, [s.id], zero, solver).y for s in inst.scenarios])
    D = pdist(y_hat, metric="hamming")               # = (1/|A_S|) sum_a |y_a^w - y_a^w'|, eq. (solutiondistance)
    Z = sp_linkage(D, method=linkage)
    labels = balance_labels(y_hat.astype(float), fcluster(Z, t=m, criterion="maxclust"), max_size)
    return _labels_to_bundles(labels, [s.id for s in inst.scenarios])


def cost_signatures(inst: SSNDInstance, designs: Optional[Sequence[Dict[int, float]]] = None,
                    n_designs: int = 4, seed: int = 0, solver=None, cache: Optional[Dict] = None,
                    parallel: int = 1) -> np.ndarray:
    """Signature matrix (|Omega| x D): Q_omega(y_d) for D candidate designs y_d.

    Default designs: all services off, all on, and random designs with activation prob. 0.35 / 0.65
    (cheap, diverse proxies for 'good designs'; pass `designs` to use e.g. the LP-rounded design or
    an elite set). Recourse LPs are cached by design, so reusing `cache` in the Lagrangian run
    makes the signatures free if those designs are visited again.
    """
    from .models import evaluate_design
    sids = inst.service_ids
    if designs is None:
        rng = np.random.default_rng(seed)
        designs = [{a: 0.0 for a in sids}, {a: 1.0 for a in sids}]
        probs = [0.35, 0.65, 0.5, 0.2, 0.8]
        for d in range(max(0, n_designs - 2)):
            p = probs[d % len(probs)]
            designs.append({a: float(rng.random() < p) for a in sids})
    cols = []
    for y in designs:
        ev = evaluate_design(inst, y, solver, cache=cache, parallel=parallel)
        cols.append([r.cost for r in ev.per_scenario])
    return np.array(cols).T


def cost_signature_bundles(inst: SSNDInstance, m: int, seed: int = 0, designs=None, n_designs: int = 4,
                           solver=None, cache: Optional[Dict] = None, parallel: int = 1,
                           max_size: Optional[int] = None) -> Bundles:
    """Bundles = k-means clusters of standardized cost signatures (scenarios that are expensive/cheap
    under the same designs end up together)."""
    X = standardize(cost_signatures(inst, designs, n_designs, seed, solver, cache, parallel))
    labels = balance_labels(X, _kmeans(X, m, seed), max_size)
    return _labels_to_bundles(labels, [s.id for s in inst.scenarios])


def stratified_bundles(inst: SSNDInstance, m: int, seed: int = 0, basis: str = "costsig",
                       n_strata: Optional[int] = None, solver=None, cache: Optional[Dict] = None,
                       parallel: int = 1) -> Bundles:
    """Diverse (anti-clustered) bundles of near-equal size.

    1. Cluster the scenarios into q = ceil(|Omega|/m) strata of similar scenarios, on cost signatures
       (basis='costsig') or demand/disruption features (basis='features').
    2. List the scenarios stratum by stratum (inside a stratum: closest to the stratum centre first)
       and deal them round-robin over the m bundles.
    Each bundle therefore receives at most one or two scenarios from every stratum: it spans the
    regimes, as a random partition does on average, but by construction rather than by chance.
    """
    if basis == "costsig":
        X = standardize(cost_signatures(inst, seed=seed, solver=solver, cache=cache, parallel=parallel))
    else:
        X = standardize(scenario_features(inst))
    S = len(X)
    q = n_strata or int(np.ceil(S / m))
    q = max(1, min(q, S))
    lab = _kmeans(X, q, seed)
    order = []
    for j in sorted(set(int(l) for l in lab)):
        idx = np.where(lab == j)[0]
        c = X[idx].mean(axis=0)
        order.extend(idx[np.argsort(((X[idx] - c) ** 2).sum(axis=1))].tolist())
    ids = [s.id for s in inst.scenarios]
    bundles: List[List[int]] = [[] for _ in range(m)]
    for pos, i in enumerate(order):
        bundles[pos % m].append(int(ids[i]))
    return [sorted(b) for b in bundles if b]


def bundle_probabilities(inst: SSNDInstance, bundles: Bundles) -> np.ndarray:
    return np.array([sum(inst.scenarios[w].prob for w in b) for b in bundles])


def describe_bundles(inst: SSNDInstance, bundles: Bundles) -> str:
    lines = []
    for j, b in enumerate(bundles):
        regimes = [inst.scenarios[w].regime for w in b]
        lines.append(f"  B{j}: {b}  P={sum(inst.scenarios[w].prob for w in b):.3f}  regimes={regimes}")
    return "\n".join(lines)
