"""Learning components (v5).

Two learned components plug into the bundled Lagrangian decomposition without touching its
validity: every learned output is either a *starting multiplier* (passed through the guard, so the
start is never worse than lambda = 0, and every L(lambda) stays a valid lower bound) or a *choice of
partition* among valid partitions (any partition gives a valid bound, Proposition 1).

1. Learned dual warm start  (DualStartPolicy; regression variant MultiplierPredictor)
   lambda(w) = F (.) Pi(sum_k w_k d_k), with directions d_k built from size-independent features
   (the bundle's availability of each arc, its demand level, its LP-relaxation design, its mass).
   The weights are fitted on training instances by derivative-free search that maximises the share
   of the bound gap closed by the start itself. The start is projected onto Lambda and offered to the
   guard, so it is never worse than lambda = 0. (Regressing directly onto lambda* of long runs,
   MultiplierPredictor, is kept for comparison: lambda* is not unique, and the regression did not
   generalise in our tests.)

2. Learned partition selection  (PartitionScorer)
   Scores candidate partitions from cheap structural features (within-bundle diversity, similarity
   of the bundles to each other, regime coverage, balance) and returns the candidate with the highest
   predicted bound at lambda = 0. Trained on (partition features, L(0)) pairs; the target is
   standardised within each instance, so the model learns a ranking, not instance-specific levels.
   Model: ridge regression on features standardised within each instance's candidate pool; the
   hybrid scorer also uses the LP proxy as a feature. Non-learning reference: lp_proxy_scores (sum of
   bundle LP relaxations at lambda = 0, m LPs per candidate).

3. Warm transfer (WarmTransfer), a non-learning baseline: lambda* of the nearest solved training
   instance of the same network, bundles matched by assignment, rescaled by fixed cost.

Features never use the deterministic equivalent or z*; everything is computable before the
decomposition starts.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .bundling import (Bundles, cost_signatures, kmeans_bundles, random_bundles, scenario_features,
                       standardize, stratified_bundles, cost_signature_bundles, bundle_probabilities)
from .instance import SSNDInstance
from .models import solve_bundle_lp
from .warmstart import project_balanced

MODES = ("road", "rail", "barge")


# ============================================================================ multiplier features
def bundle_lp_designs(inst: SSNDInstance, bundles: Bundles, solver=None) -> Tuple[np.ndarray, np.ndarray]:
    """LP relaxations of all bundle subproblems at lambda = 0: (values (m,), fractional designs (m x n))."""
    vals, ys = [], []
    for b in bundles:
        v, y = solve_bundle_lp(inst, b, None, solver)
        vals.append(v)
        ys.append(y)
    return np.array(vals), np.vstack(ys)


MULT_FEATURES = ["P_rel", "size_rel", "F_rel", "U_rel", "mode_road", "mode_rail", "mode_barge",
                 "xi_b", "xi_dev", "dem_dev", "ylp_b", "ylp_mean", "ylp_dev", "ylp_dev_abs"]


def multiplier_features(inst: SSNDInstance, bundles: Bundles, y_lp: np.ndarray) -> np.ndarray:
    """Feature matrix with one row per (bundle b, service arc a), rows ordered b-major.

    y_lp: fractional bundle designs at lambda = 0 (from bundle_lp_designs).
    """
    sids = inst.service_ids
    m, n, S = len(bundles), len(sids), len(inst.scenarios)
    F = np.array([inst.arcs[a].F for a in sids])
    U = np.array([inst.arcs[a].U for a in sids])
    modes = [inst.arcs[a].mode for a in sids]
    p = np.array([s.prob for s in inst.scenarios])
    Xi = np.array([[s.xi.get(a, 1.0) for a in sids] for s in inst.scenarios])          # S x n
    D = np.array([sum(s.demand.values()) for s in inst.scenarios])                      # S
    xi_glob = p @ Xi
    d_glob = float(p @ D)
    P = bundle_probabilities(inst, bundles)
    ylp_mean = y_lp.mean(axis=0)
    rows = []
    for b, bundle in enumerate(bundles):
        w = p[bundle] / max(P[b], 1e-12)
        xi_b = w @ Xi[bundle]
        d_b = float(w @ D[bundle])
        for j in range(n):
            dev = y_lp[b, j] - ylp_mean[j]
            rows.append([P[b] * m, len(bundle) * m / S, F[j] / F.mean(), U[j] / U.mean(),
                         float(modes[j] == "road"), float(modes[j] == "rail"), float(modes[j] == "barge"),
                         xi_b[j], xi_b[j] - xi_glob[j], d_b / max(d_glob, 1e-12) - 1.0,
                         y_lp[b, j], ylp_mean[j], dev, abs(dev)])
    return np.array(rows, dtype=float)


def multiplier_targets(inst: SSNDInstance, lam: np.ndarray) -> np.ndarray:
    """lambda_b,a / F_a, flattened b-major (same order as multiplier_features)."""
    F = np.array([inst.arcs[a].F for a in inst.service_ids])
    return (np.asarray(lam) / F[None, :]).ravel()


@dataclass
class MultiplierPredictor:
    model: object = None
    feature_names: List[str] = field(default_factory=lambda: list(MULT_FEATURES))
    info: Dict = field(default_factory=dict)

    @staticmethod
    def _new_model(seed: int = 0):
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                             l2_regularization=1e-3, random_state=seed)

    def fit(self, X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None,
            cv_folds: int = 5, seed: int = 0) -> Dict:
        """Fit on all rows; if groups (instance ids) are given, report grouped cross-validated R^2."""
        from sklearn.metrics import r2_score
        info = {"n_rows": int(len(y))}
        if groups is not None and len(np.unique(groups)) >= cv_folds:
            from sklearn.model_selection import GroupKFold
            pred = np.zeros_like(y)
            for tr, te in GroupKFold(n_splits=cv_folds).split(X, y, groups):
                mdl = self._new_model(seed).fit(X[tr], y[tr])
                pred[te] = mdl.predict(X[te])
            info["cv_r2"] = float(r2_score(y, pred))
            info["cv_sign_agreement"] = float(np.mean(np.sign(pred) == np.sign(y)))
        self.model = self._new_model(seed).fit(X, y)
        self.info = info
        return info

    def predict(self, inst: SSNDInstance, bundles: Bundles, y_lp: Optional[np.ndarray] = None,
                solver=None) -> np.ndarray:
        """Predicted multipliers (m x n), projected onto Lambda."""
        if y_lp is None:
            _, y_lp = bundle_lp_designs(inst, bundles, solver)
        X = multiplier_features(inst, bundles, y_lp)
        F = np.array([inst.arcs[a].F for a in inst.service_ids])
        lam = self.model.predict(X).reshape(len(bundles), len(F)) * F[None, :]
        return project_balanced(lam)


# ============================================================================ partition features
PART_FEATURES = ["sig_within", "sig_between", "phi_within", "phi_between", "regime_cov",
                 "regime_cov_min", "P_cv", "P_max_rel", "size_cv"]


def partition_features(inst: SSNDInstance, bundles: Bundles, X_sig: np.ndarray,
                       Phi: np.ndarray) -> np.ndarray:
    """Structural description of a partition.

    X_sig: standardised cost signatures (S x D); Phi: standardised scenario features (S x d).
    within  = mean over bundles of the mean per-dimension std inside the bundle (diversity of a bundle)
    between = mean per-dimension std of the bundle means (how different the bundles are from each other)
    """
    ids = [s.id for s in inst.scenarios]
    pos = {w: i for i, w in enumerate(ids)}
    regimes = sorted({s.regime for s in inst.scenarios})
    P = bundle_probabilities(inst, bundles)
    m = len(bundles)

    def spread(X):
        within, means = [], []
        for b in bundles:
            Z = X[[pos[w] for w in b]]
            within.append(Z.std(axis=0).mean() if len(b) > 1 else 0.0)
            means.append(Z.mean(axis=0))
        return float(np.mean(within)), float(np.vstack(means).std(axis=0).mean())

    sw, sb = spread(X_sig)
    pw, pb = spread(Phi)
    cov = [len({inst.scenarios[pos[w]].regime for w in b}) / len(regimes) for b in bundles]
    sizes = np.array([len(b) for b in bundles], dtype=float)
    return np.array([sw, sb, pw, pb, float(np.mean(cov)), float(np.min(cov)),
                     float(P.std() / max(P.mean(), 1e-12)), float(P.max() * m),
                     float(sizes.std() / max(sizes.mean(), 1e-12))])


def instance_signature_data(inst: SSNDInstance, solver=None, cache: Optional[Dict] = None,
                            parallel: int = 1, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """(standardised cost signatures, standardised scenario features), computed once per instance."""
    X_sig = standardize(cost_signatures(inst, seed=seed, solver=solver, cache=cache, parallel=parallel))
    Phi = standardize(scenario_features(inst))
    return X_sig, Phi


def swap_perturbation(bundles: Bundles, n_swaps: int, seed: int) -> Bundles:
    """Swap scenarios between bundles n_swaps times (keeps bundle sizes)."""
    rng = np.random.default_rng(seed)
    B = [list(b) for b in bundles]
    for _ in range(n_swaps):
        i, j = rng.choice(len(B), size=2, replace=False)
        a, b = rng.integers(len(B[i])), rng.integers(len(B[j]))
        B[i][a], B[j][b] = B[j][b], B[i][a]
    return [sorted(b) for b in B]


def candidate_partitions(inst: SSNDInstance, m: int, n_candidates: int = 20, seed: int = 0,
                         solver=None, cache: Optional[Dict] = None, parallel: int = 1,
                         max_size: Optional[int] = None) -> List[Tuple[str, Bundles]]:
    """A diverse pool of valid partitions: homogeneous (k-means, cost signature), diverse
    (stratified on both bases), random draws, and swap perturbations of stratified partitions."""
    pool: List[Tuple[str, Bundles]] = [
        ("kmeans", kmeans_bundles(inst, m, seed=seed, max_size=max_size)),
        ("costsig", cost_signature_bundles(inst, m, seed=seed, solver=solver, cache=cache,
                                           parallel=parallel, max_size=max_size)),
    ]
    k = 0
    while len(pool) < n_candidates:
        r = len(pool) % 4
        if r == 0:
            pool.append((f"stratified_c{k}", stratified_bundles(inst, m, seed=seed + 101 * k, basis="costsig",
                                                                solver=solver, cache=cache, parallel=parallel)))
        elif r == 1:
            pool.append((f"stratified_f{k}", stratified_bundles(inst, m, seed=seed + 101 * k, basis="features")))
        elif r == 2:
            pool.append((f"random{k}", random_bundles(inst, m, seed=seed + 1000 + k)))
        else:
            base = stratified_bundles(inst, m, seed=seed + 101 * k, basis="costsig", solver=solver,
                                      cache=cache, parallel=parallel)
            pool.append((f"swap{k}", swap_perturbation(base, n_swaps=max(1, len(inst.scenarios) // 4),
                                                       seed=seed + 7 * k)))
            k += 1
    # drop duplicates (same partition produced twice)
    seen, out = set(), []
    for name, B in pool:
        key = tuple(sorted(tuple(sorted(b)) for b in B))
        if key not in seen:
            seen.add(key)
            out.append((name, B))
    return out


def lp_proxy_scores(inst: SSNDInstance, partitions: Sequence[Bundles], solver=None) -> np.ndarray:
    """Non-learning reference: L_LP(0) = sum of bundle LP relaxations at lambda = 0."""
    return np.array([bundle_lp_designs(inst, B, solver)[0].sum() for B in partitions])


def within_instance_standardise(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    out = np.zeros_like(values, dtype=float)
    for g in np.unique(groups):
        idx = groups == g
        v = values[idx]
        sd = v.std()
        out[idx] = (v - v.mean()) / (sd if sd > 1e-12 else 1.0)
    return out


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(spearmanr(a, b).correlation)


@dataclass
class PartitionScorer:
    """Ranks candidate partitions of one instance by their predicted L(0).

    Features are standardised *within the candidate pool of each instance* (training: within each
    training instance), so the model compares partitions of the same instance and transfers across
    instance sizes. Model: ridge regression (interpretable coefficients; tree ensembles overfitted
    the few hundred training partitions). use_lp_proxy=True adds the LP proxy L_LP(0) as a feature
    (hybrid scorer: m LPs per candidate at selection time).
    """
    model: object = None
    use_lp_proxy: bool = True
    feature_names: List[str] = field(default_factory=lambda: list(PART_FEATURES))
    info: Dict = field(default_factory=dict)

    @staticmethod
    def _new_model():
        from sklearn.linear_model import RidgeCV
        return RidgeCV(alphas=np.logspace(-2, 3, 20))

    def _design(self, X: np.ndarray, groups: np.ndarray, lp_proxy: Optional[np.ndarray]) -> np.ndarray:
        cols = [within_instance_standardise(X[:, j], groups) for j in range(X.shape[1])]
        if self.use_lp_proxy:
            if lp_proxy is None:
                raise ValueError("this scorer uses the LP proxy as a feature: pass lp_proxy")
            cols.append(within_instance_standardise(np.asarray(lp_proxy, dtype=float), groups))
        return np.column_stack(cols)

    def fit(self, X: np.ndarray, L0: np.ndarray, groups: np.ndarray, cv_folds: int = 5,
            seed: int = 0, lp_proxy: Optional[np.ndarray] = None) -> Dict:
        """Target: L(0) standardised within each instance. Reports grouped CV of the ranking
        (mean within-instance Spearman rho, and the regret of picking the top-scored candidate,
        as % of the instance's best L(0)); the same metrics for the raw LP proxy if given."""
        groups = np.asarray(groups)
        y = within_instance_standardise(L0, groups)
        Z = self._design(X, groups, lp_proxy)
        info: Dict = {"n_rows": int(len(y)), "n_instances": int(len(np.unique(groups))),
                      "use_lp_proxy": self.use_lp_proxy}
        if len(np.unique(groups)) >= cv_folds:
            from sklearn.model_selection import GroupKFold
            pred = np.zeros_like(y)
            for tr, te in GroupKFold(n_splits=cv_folds).split(Z, y, groups):
                pred[te] = self._new_model().fit(Z[tr], y[tr]).predict(Z[te])
            info.update(self._ranking_metrics(pred, L0, groups, prefix="cv_"))
            if lp_proxy is not None:
                info.update(self._ranking_metrics(np.asarray(lp_proxy), L0, groups, prefix="lpproxy_"))
        self.model = self._new_model().fit(Z, y)
        names = self.feature_names + (["lp_proxy"] if self.use_lp_proxy else [])
        info["coefficients"] = {k: round(float(c), 4) for k, c in zip(names, self.model.coef_)}
        self.info = info
        return info

    @staticmethod
    def _ranking_metrics(score, L0, groups, prefix):
        rhos, regrets = [], []
        for g in np.unique(groups):
            idx = np.where(groups == g)[0]
            rhos.append(spearman(score[idx], L0[idx]))
            pick = idx[np.argmax(score[idx])]
            best = L0[idx].max()
            regrets.append(100 * (best - L0[pick]) / max(abs(best), 1e-12))
        return {prefix + "spearman": float(np.nanmean(rhos)), prefix + "top1_regret_pct": float(np.mean(regrets))}

    def select(self, inst: SSNDInstance, partitions: Sequence[Tuple[str, Bundles]], X_sig: np.ndarray,
               Phi: np.ndarray, lp_proxy: Optional[np.ndarray] = None, solver=None) -> Tuple[int, np.ndarray]:
        feats = np.vstack([partition_features(inst, B, X_sig, Phi) for _, B in partitions])
        if self.use_lp_proxy and lp_proxy is None:
            lp_proxy = lp_proxy_scores(inst, [B for _, B in partitions], solver)
        Z = self._design(feats, np.zeros(len(partitions), dtype=int), lp_proxy)
        s = self.model.predict(Z)
        return int(np.argmax(s)), s


# ============================================================================ dual start policy
POLICY_DIRECTIONS = ["xi_dev", "dem_dev", "ylp_dev", "P_dev"]


def policy_directions(inst: SSNDInstance, bundles: Bundles, y_lp: np.ndarray) -> np.ndarray:
    """K x m x n stack of balanced, unit-scaled directions built from the multiplier features."""
    m, n = len(bundles), len(inst.service_ids)
    X = multiplier_features(inst, bundles, y_lp)
    col = {k: MULT_FEATURES.index(k) for k in ("xi_dev", "dem_dev", "ylp_dev", "P_rel")}
    out = []
    for name in POLICY_DIRECTIONS:
        j = col["P_rel"] if name == "P_dev" else col[name]
        d = project_balanced(X[:, j].reshape(m, n))
        nrm = np.abs(d).max()
        out.append(d / nrm if nrm > 1e-12 else d)
    return np.stack(out)


@dataclass
class DualStartPolicy:
    """Learned dual warm start  lambda(w)_b = Pi( s_b F (.) sum_k w_k d_k ),  d_k = POLICY_DIRECTIONS.

    Scale s_b: with scale="bundle" (default for every policy fitted from v9 on) s_b = P_b, the
    probability of bundle b, which is the weight of the fixed cost F in the bundle subproblem
    (P_b F^T y^b + lambda_b^T y^b). The multipliers are then a fixed fraction of the fixed cost the
    bundle actually carries, and the same weights apply for any number of bundles m. With
    scale="none" (policies fitted up to v8) s_b = 1: the multipliers do not shrink when m grows, so a
    policy fitted at m = 4 overshoots by a factor m/4 at larger m (at m = 48 it drives L below
    L(0) by thousands). `rescale_policy` converts such a policy exactly at its training m.

    The weights w are fitted by derivative-free search that maximises the mean *closure* of the
    training instances, closure = (L(lambda(w)) - L(0)) / (LB* - L(0)): the share of the gap between
    the zero start and the best bound of a long run that the start alone closes. Training on the
    dual value itself avoids regressing onto lambda*, which is not unique (the dual optimum of a
    MILP-based Lagrangian is a face, and a subgradient run ends at an arbitrary point of it).
    """
    w: np.ndarray = field(default_factory=lambda: np.zeros(len(POLICY_DIRECTIONS)))
    info: Dict = field(default_factory=dict)
    scale: str = "none"     # class-level default "none" keeps pickles from v5-v8 unchanged

    @staticmethod
    def bundle_scale(inst: SSNDInstance, bundles: Bundles, scale: str) -> np.ndarray:
        if scale == "bundle":
            return bundle_probabilities(inst, bundles)[:, None]
        return np.ones((len(bundles), 1))

    def multipliers(self, inst: SSNDInstance, bundles: Bundles, y_lp: Optional[np.ndarray] = None,
                    solver=None, w: Optional[np.ndarray] = None) -> np.ndarray:
        if y_lp is None:
            _, y_lp = bundle_lp_designs(inst, bundles, solver)
        D = policy_directions(inst, bundles, y_lp)
        w = self.w if w is None else w
        F = np.array([inst.arcs[a].F for a in inst.service_ids])
        s = self.bundle_scale(inst, bundles, getattr(self, "scale", "none"))
        return project_balanced(s * F[None, :] * np.tensordot(w, D, axes=1))

    def fit(self, cases: Sequence[Dict], evaluate, step0: float = 0.02, step_min: float = 0.0025,
            max_evals: int = 120, verbose: bool = True, scale: str = "bundle",
            w0: Optional[np.ndarray] = None) -> Dict:
        """cases: dicts with keys inst, bundles, D (directions), L0, LBstar.
        evaluate(case, lam) -> L(lam) (m bundle MILPs). Coordinate search with step halving from w = 0."""
        cases = [c for c in cases if c["LBstar"] - c["L0"] > 1e-6 * max(1.0, abs(c["L0"]))]
        self.scale = scale
        F = {id(c): self.bundle_scale(c["inst"], c["bundles"], scale)
                    * np.array([c["inst"].arcs[a].F for a in c["inst"].service_ids])[None, :] for c in cases}

        def objective(w):
            vals = []
            for c in cases:
                lam = project_balanced(F[id(c)] * np.tensordot(w, c["D"], axes=1))
                vals.append((evaluate(c, lam) - c["L0"]) / (c["LBstar"] - c["L0"]))
            return float(np.mean(vals)), vals

        w = np.zeros(len(POLICY_DIRECTIONS)) if w0 is None else np.asarray(w0, dtype=float).copy()
        best = 0.0 if w0 is None else objective(w)[0]
        if verbose and w0 is not None:
            print(f"    policy search: initial w={np.round(w, 4)}  closure={best:.4f}", flush=True)
        step, evals = step0, 0
        while step >= step_min and evals < max_evals:
            improved = False
            for k in range(len(w)):
                for sgn in (+1, -1):
                    w2 = w.copy(); w2[k] += sgn * step
                    val, _ = objective(w2); evals += 1
                    if val > best + 1e-6:
                        w, best, improved = w2, val, True
                        if verbose:
                            print(f"    policy search: eval {evals:3d}  closure={best:.4f}  w={np.round(w, 4)}", flush=True)
                        break
                if evals >= max_evals:
                    break
            if not improved:
                step /= 2
        _, per_case = objective(w)
        self.w = w
        self.info = {"scale": scale, "train_closure_mean": best, "train_closure_min": float(np.min(per_case)),
                     "train_cases": len(cases), "evals": evals,
                     "weights": {k: round(float(v), 5) for k, v in zip(POLICY_DIRECTIONS, w)}}
        return self.info


# ============================================================================ persistence
def save_models(path: str, multiplier, scorer: Optional[PartitionScorer], meta: Optional[Dict] = None) -> None:
    """multiplier: a DualStartPolicy (default) or a MultiplierPredictor (regression variant)."""
    with open(path, "wb") as f:
        pickle.dump({"multiplier": multiplier, "scorer": scorer, "meta": meta or {}}, f)


def rescale_policy(policy: "DualStartPolicy", m_train: int) -> "DualStartPolicy":
    """Exact conversion of a scale="none" policy fitted with m_train equal bundles to scale="bundle":
    at m = m_train (P_b = 1/m_train) both give the same multipliers; at other m the converted policy
    scales them by m_train/m."""
    if getattr(policy, "scale", "none") == "bundle":
        return policy
    info = dict(policy.info); info["rescaled_from_m"] = m_train
    return DualStartPolicy(w=np.asarray(policy.w, dtype=float) * m_train, info=info, scale="bundle")


def load_models(path: str):
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d.get("multiplier"), d.get("scorer"), d.get("meta", {})


# ============================================================================ warm transfer (baseline)
REGIMES = ("nominal", "localized", "corridor", "severe")


def instance_features(inst: SSNDInstance) -> np.ndarray:
    """Instance-level description used to find the nearest solved training instance."""
    p = np.array([s.prob for s in inst.scenarios])
    D = np.array([sum(s.demand.values()) for s in inst.scenarios])
    mu = float(p @ D)
    cv = float(np.sqrt(p @ (D - mu) ** 2) / max(mu, 1e-12))
    reg = [float(sum(s.prob for s in inst.scenarios if s.regime == r)) for r in REGIMES]
    sids = inst.service_ids
    xi = float(np.mean([[s.xi.get(a, 1.0) for a in sids] for s in inst.scenarios]))
    F = np.array([inst.arcs[a].F for a in sids])
    U = np.array([inst.arcs[a].U for a in sids])
    return np.array([mu, cv, *reg, xi, F.mean(), U.mean()])


def bundle_features(inst: SSNDInstance, bundles: Bundles) -> np.ndarray:
    """One row per bundle (mass, demand level, regime mix, mean availability), for bundle matching."""
    sids = inst.service_ids
    p = np.array([s.prob for s in inst.scenarios])
    D = np.array([sum(s.demand.values()) for s in inst.scenarios])
    d_glob = float(p @ D)
    P = bundle_probabilities(inst, bundles)
    m = len(bundles)
    rows = []
    for b, B in enumerate(bundles):
        w = p[B] / max(P[b], 1e-12)
        reg = [float(sum(w[i] for i, sid in enumerate(B) if inst.scenarios[sid].regime == r)) for r in REGIMES]
        xi = float(np.mean([[inst.scenarios[sid].xi.get(a, 1.0) for a in sids] for sid in B]))
        rows.append([P[b] * m, float(w @ D[B]) / max(d_glob, 1e-12), *reg, xi])
    return np.array(rows)


@dataclass
class WarmTransfer:
    """Non-learning baseline: start from the best multiplier lambda* of the nearest solved training
    instance of the same network (same service arcs) and the same number of bundles.
    Bundles are matched by an assignment on bundle features; lambda*_b,a is transferred relative to
    the fixed cost (lambda*/F) and rescaled by the test instance's F, then projected onto Lambda."""
    feats: List[np.ndarray] = field(default_factory=list)       # instance features
    bfeats: List[np.ndarray] = field(default_factory=list)      # bundle features (m x d)
    lam_rel: List[np.ndarray] = field(default_factory=list)     # lambda*/F (m x n)
    sids: List[Tuple] = field(default_factory=list)
    names: List[str] = field(default_factory=list)

    def add(self, inst: SSNDInstance, bundles: Bundles, lam_rel: np.ndarray, name: str = "") -> None:
        self.feats.append(instance_features(inst))
        self.bfeats.append(bundle_features(inst, bundles))
        self.lam_rel.append(np.asarray(lam_rel).reshape(len(bundles), len(inst.service_ids)))
        self.sids.append(tuple(inst.service_ids))
        self.names.append(name)

    def __len__(self):
        return len(self.feats)

    def multipliers(self, inst: SSNDInstance, bundles: Bundles) -> Tuple[Optional[np.ndarray], str]:
        """(projected start, name of the source run); (None, reason) if no compatible run exists."""
        from scipy.optimize import linear_sum_assignment
        key, m = tuple(inst.service_ids), len(bundles)
        ok = [i for i in range(len(self)) if self.sids[i] == key and self.lam_rel[i].shape[0] == m]
        if not ok:
            return None, "no training run with the same network and number of bundles"
        Fm = np.vstack([self.feats[i] for i in ok])
        mu, sd = Fm.mean(axis=0), Fm.std(axis=0) + 1e-12
        q = (instance_features(inst) - mu) / sd
        j = ok[int(np.argmin((((Fm - mu) / sd - q) ** 2).sum(axis=1)))]
        Bt, Bs = bundle_features(inst, bundles), self.bfeats[j]
        both = np.vstack([Bt, Bs])
        s = both.std(axis=0) + 1e-12
        cost = (((Bt[:, None, :] - Bs[None, :, :]) / s) ** 2).sum(axis=2)
        rows, cols = linear_sum_assignment(cost)
        perm = np.empty(m, dtype=int)
        perm[rows] = cols
        F = np.array([inst.arcs[a].F for a in inst.service_ids])
        lam = self.lam_rel[j][perm] * F[None, :]
        return project_balanced(lam), self.names[j]
