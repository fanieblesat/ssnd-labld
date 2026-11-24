"""Bundled Lagrangian decomposition with projected subgradient ascent, guarded warm start,
and consensus-based primal recovery (Algorithm 1 of the draft, with the audit fixes).

Dual problem
    max_{lambda in Lambda}  L(lambda) = sum_b phi_b(lambda_b),
    Lambda = {sum_b lambda_b = 0},   Pi(lambda)_b = lambda_b - (1/m) sum_c lambda_c.

Projected subgradient (consistent with the unweighted projection; audit 4.7)
    g_b = y^b(lambda) - (1/m) sum_c y^c(lambda),     lambda <- Pi(lambda + alpha_t g).

Step rules
    'polyak'      alpha_t = theta_t (U_t - L(lambda_t)) / (||g||^2 + eps), 0 < theta_t < 2
                  U_t = best upper bound so far. theta_t is halved after `theta_patience` iterations
                  without improvement of the best bound, and the iterate is reset to the best
                  multiplier found (Held-Karp style); without this the step overshoots whenever
                  U_t is still far from z* and L(lambda) oscillates instead of climbing.
    'diminishing' alpha_t = alpha0 / sqrt(t + 1)

Every evaluated L(lambda) is a valid lower bound (Proposition 1) and every recovered design
is evaluated exactly with all scenario recourse LPs, so U is a valid upper bound.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .bundling import Bundles, bundle_probabilities
from .instance import SSNDInstance
from .models import (DesignEvaluation, design_key, evaluate_design, get_solver,
                     solve_bundle_subproblem)
from .warmstart import project_balanced


@dataclass
class LDResult:
    best_lb: float
    best_ub: float
    best_y: Optional[Dict[int, float]]
    gap: float
    iterations: int
    seconds: float
    n_bundle_solves: int
    n_recourse_solves: int
    start_source: str
    lam_final: np.ndarray
    history: List[Dict] = field(default_factory=list)
    best_eval: Optional[DesignEvaluation] = None
    n_unproved_bundle_solves: int = 0     # bundle MILPs stopped by the time limit (LP bound used)
    n_design_evals: int = 0               # distinct designs evaluated (cache misses)

    def summary(self) -> str:
        return (f"LB*={self.best_lb:.2f}  UB*={self.best_ub:.2f}  gap={100 * self.gap:.2f}%  "
                f"iters={self.iterations}  bundle solves={self.n_bundle_solves}  "
                f"recourse LPs={self.n_recourse_solves}  time={self.seconds:.1f}s  start={self.start_source}"
                + (f"  [unproved bundle MILPs: {self.n_unproved_bundle_solves}]" if self.n_unproved_bundle_solves else ""))


class BundledLagrangian:
    def __init__(self, inst: SSNDInstance, bundles: Bundles, solver=None,
                 step_rule: str = "polyak", theta: float = 1.0, alpha0: Optional[float] = None,
                 tau: float = 0.5, max_iter: int = 50, time_limit: Optional[float] = None,
                 gap_tol: float = 1e-3, eps: float = 1e-9,
                 primal_recovery: bool = True, recovery_every: int = 1,
                 local_search: bool = True, ls_max_evals: int = 15,
                 parallel: int = 1, verbose: bool = True,
                 recourse_solver=None, eval_cache: Optional[Dict] = None,
                 theta_patience: int = 4, theta_min: float = 1e-3, reset_to_best: bool = True,
                 theta_warm: Optional[float] = None, backend=None):
        inst.validate()
        self.inst, self.bundles = inst, [list(b) for b in bundles]
        covered = sorted(w for b in self.bundles for w in b)
        assert covered == [s.id for s in inst.scenarios], "bundles must partition the scenario set"
        self.solver = solver or get_solver()
        self.recourse_solver = recourse_solver or self.solver   # e.g. LP solves need no time limit
        self.eval_cache = {} if eval_cache is None else eval_cache  # design -> DesignEvaluation
        self.m, self.n = len(self.bundles), len(inst.service_ids)
        self.P = bundle_probabilities(inst, self.bundles)
        self.sids = inst.service_ids
        self.step_rule, self.theta, self.tau = step_rule, theta, tau
        self.theta_patience, self.theta_min, self.reset_to_best = theta_patience, theta_min, reset_to_best
        # initial Polyak factor after an accepted non-zero start (None: same as theta). A start that the guard
        # accepted is already closer to the dual optimum, so a full first step tends to overshoot.
        self.theta_warm = theta_warm
        # optional persistent-model backend (grb_backend.GurobiBackend) for large scenario sets
        self.backend = backend
        F = np.array([inst.arcs[a].F for a in self.sids])
        self.alpha0 = alpha0 if alpha0 is not None else 0.1 * float(F.mean())
        self.max_iter, self.time_limit, self.gap_tol, self.eps = max_iter, time_limit, gap_tol, eps
        self.primal_recovery, self.recovery_every = primal_recovery, recovery_every
        self.local_search, self.ls_max_evals = local_search, ls_max_evals
        self.parallel, self.verbose = max(1, parallel), verbose
        self.n_bundle_solves = 0
        self.n_recourse_solves = 0
        self.n_design_evals = 0
        self.n_unproved = 0
        self._deadline = None                   # absolute wall-clock deadline (backend runs only)
        self._last = None                       # (lam, bundle bounds, bundle designs, proved) of the last evaluation
        self.timed_out = False
        self.open_count = np.zeros(self.n)      # how often each service was opened in a bundle solution

    # ------------------------------------------------------------------ dual side
    def project(self, lam: np.ndarray) -> np.ndarray:
        return project_balanced(lam)

    def evaluate(self, lam: np.ndarray, partial_ok: bool = False):
        """L(lambda) and the bundle solutions y^b (m x n). Bundle subproblems run in parallel threads
        (CBC runs as a subprocess, so threads give real parallelism)."""
        lam = np.asarray(lam, dtype=float)
        assert lam.shape == (self.m, self.n)
        if self.backend is not None:
            vals, ys, proved, complete = self.backend.solve_bundles(lam, deadline=self._bundle_deadline(),
                                                                     partial_ok=partial_ok)
            if not complete:                    # time ran out inside the iteration: L(lam) is not valid
                return None, None
            self.n_bundle_solves += self.m
            self.n_unproved += int((~proved).sum())
            self.open_count += (ys > 0.5).sum(axis=0)
            self._last = (lam.copy(), vals.copy(), ys.copy(), proved.copy())
            return float(vals.sum()), ys
        if self.parallel > 1:
            with ThreadPoolExecutor(max_workers=self.parallel) as ex:
                res = list(ex.map(lambda b: solve_bundle_subproblem(self.inst, self.bundles[b], lam[b], self.solver),
                                  range(self.m)))
        else:
            res = [solve_bundle_subproblem(self.inst, self.bundles[b], lam[b], self.solver) for b in range(self.m)]
        self.n_bundle_solves += self.m
        self.n_unproved += sum(1 for r in res if not r.proved)
        L = float(sum(r.value for r in res))
        ys = np.vstack([r.y for r in res])
        self.open_count += (ys > 0.5).sum(axis=0)
        return L, ys

    def subgradient(self, ys: np.ndarray) -> np.ndarray:
        """Projected subgradient g_b = y^b - (1/m) sum_c y^c (already in Lambda)."""
        return ys - ys.mean(axis=0, keepdims=True)

    def guarded_start(self, candidates: Sequence[np.ndarray], names: Optional[Sequence[str]] = None):
        """eq. (guard), generalised to any number of candidates: evaluate 0 and every projected candidate,
        return the one with the largest dual value together with its bundle solutions."""
        cands = [np.zeros((self.m, self.n))] + [self.project(c) for c in candidates]
        labels = ["zero"] + list(names or [f"cand{i}" for i in range(len(candidates))])
        best = None
        for i, (lam, lab) in enumerate(zip(cands, labels)):
            # the zero start also respects the deadline; bundles it cannot reach count with the bound 0
            L, ys = self.evaluate(lam, partial_ok=(i == 0))
            if L is None:
                continue
            if self.verbose:
                print(f"  guard: L({lab}) = {L:.3f}")
            if best is None or L > best[0]:
                best = (L, ys, lam, lab)
        return best  # (L, ys, lam, label)

    # ---------------------------------------------------------------- primal side
    def _eval_design(self, y: Dict[int, float]) -> DesignEvaluation:
        if design_key(self.inst, y) not in self.eval_cache:      # count only real (uncached) work
            self.n_recourse_solves += len(self.inst.scenarios)
            self.n_design_evals += 1
        if self.backend is not None:
            return self.backend.evaluate_design(y, cache=self.eval_cache)
        return evaluate_design(self.inst, y, self.recourse_solver, cache=self.eval_cache,
                               parallel=self.parallel)

    def recover_primal(self, ys: np.ndarray, incumbent: Optional[float] = None):
        """Probability-weighted voting (eq. (voting)) + recourse repair + add/drop local search.
        Returns (y_hat, evaluation)."""
        s = (self.P[:, None] * ys).sum(axis=0)                 # vote strength s_a
        y = {a: (1.0 if s[j] >= self.tau else 0.0) for j, a in enumerate(self.sids)}
        if not self._eval_fits():
            return None, None
        ev = self._eval_design(y)
        evals = 1
        order_add = [self.sids[j] for j in np.argsort(-s) if y[self.sids[j]] == 0.0]   # strongest votes first
        order_drop = [self.sids[j] for j in np.argsort(s) if y[self.sids[j]] == 1.0]   # weakest votes first

        # recourse repair: add services while shortage or spot use remains and adding pays off
        for a in order_add:
            if evals >= self.ls_max_evals or not self._eval_fits():
                break
            needs_repair = any(r.shortage > 1e-6 or r.spot > 1e-6 for r in ev.per_scenario)
            if not needs_repair:
                break
            y2 = dict(y); y2[a] = 1.0
            ev2 = self._eval_design(y2); evals += 1
            if ev2.expected_total_cost < ev.expected_total_cost - 1e-9:
                y, ev = y2, ev2

        # local improvement: first-improvement drop / add moves
        if self.local_search:
            improved = True
            while improved and evals < self.ls_max_evals and self._eval_fits():
                improved = False
                for a in [x for x in order_drop if y[x] == 1.0] + [x for x in order_add if y[x] == 0.0]:
                    if evals >= self.ls_max_evals or not self._eval_fits():
                        break
                    y2 = dict(y); y2[a] = 1.0 - y[a]
                    ev2 = self._eval_design(y2); evals += 1
                    if ev2.expected_total_cost < ev.expected_total_cost - 1e-9:
                        y, ev, improved = y2, ev2, True
                        break
        return y, ev

    def _tighten_unproved(self):
        """Zero subgradient with bundle MILPs stopped at their cap: the bound is not final, because the
        unfinished bundles can still be solved further at the same multipliers. Give them more time and
        keep the larger bound per bundle. Returns (L, ys) or None if nothing improved."""
        if self.backend is None or getattr(self, "_last", None) is None:
            return None
        lam, vals, ys, proved = self._last
        idx = np.nonzero(~proved)[0]
        if len(idx) == 0 or self._out_of_time():
            return None
        nv, ny, npr = self.backend.resolve_bundles(idx, deadline=self._bundle_deadline())
        self.n_bundle_solves += len(idx)
        better = nv > vals[idx] + 1e-9
        if not better.any():
            return None
        vals, ys, proved = vals.copy(), ys.copy(), proved.copy()
        vals[idx] = np.maximum(vals[idx], nv)
        has = ny.any(axis=1) | npr
        ys[idx[has]] = ny[has]
        proved[idx] = npr
        self._last = (lam, vals, ys, proved)
        return float(vals.sum()), ys

    def _out_of_time(self) -> bool:
        return self._deadline is not None and time.time() >= self._deadline

    def _eval_fits(self) -> bool:
        """A design evaluation started now ends before the deadline (measured duration + 10 %)."""
        if self._deadline is None or self.backend is None:
            return not self._out_of_time()
        return time.time() + 1.1 * self.backend.eval_s <= self._deadline

    def _bundle_deadline(self):
        """Bundle rounds end early enough to leave time for one design evaluation afterwards."""
        if self._deadline is None or self.backend is None:
            return self._deadline
        return self._deadline - 1.2 * self.backend.eval_s

    # ------------------------------------------------------------------ main loop
    def run(self, candidates: Optional[Sequence[np.ndarray]] = None,
            candidate_names: Optional[Sequence[str]] = None) -> LDResult:
        t0 = time.time()
        if self.backend is not None and self.time_limit is not None:
            self._deadline = t0 + self.time_limit
        L, ys, lam, src = self.guarded_start(candidates or [], candidate_names)
        best_lb, best_ub, best_y, best_ev = L, np.inf, None, None
        best_lam, best_ys = lam.copy(), ys.copy()
        theta = self.theta_warm if (self.theta_warm and src != "zero") else self.theta
        no_improve = 0
        history: List[Dict] = []
        t = 0
        while True:
            # (bundle solutions at lam are already available: no redundant re-solve, audit 5.1)
            U_t = None
            if self.primal_recovery and (t % self.recovery_every == 0) and (t == 0 or not self._out_of_time()):
                y_hat, ev = self.recover_primal(ys, best_ub)
                if ev is not None:
                    U_t = ev.expected_total_cost
                    if U_t < best_ub:
                        best_ub, best_y, best_ev = U_t, y_hat, ev
            gap = (best_ub - best_lb) / max(abs(best_ub), 1e-9) if np.isfinite(best_ub) else np.inf
            g = self.subgradient(ys)
            gnorm2 = float((g ** 2).sum())
            history.append(dict(t=t, L=L, best_lb=best_lb, U=U_t, best_ub=best_ub, gap=gap,
                                gnorm=np.sqrt(gnorm2), theta=theta, time=time.time() - t0))
            if self.verbose:
                ub = f"{best_ub:.2f}" if np.isfinite(best_ub) else "  inf"
                print(f"  t={t:3d}  L={L:10.2f}  LB*={best_lb:10.2f}  UB*={ub:>10}  gap={100 * gap:6.2f}%  "
                      f"||g||={np.sqrt(gnorm2):.3f}  theta={theta:.3g}")
            # zero subgradient but bundles stopped at their cap: tighten them instead of stopping
            if gnorm2 <= 1e-12 and gap > self.gap_tol and t + 1 < self.max_iter:
                tight = self._tighten_unproved()
                if tight is not None:
                    L, ys = tight
                    t += 1
                    if L > best_lb + 1e-9:
                        best_lb, best_lam, best_ys = L, lam.copy(), ys.copy()
                    continue
            # stopping
            if gap <= self.gap_tol or gnorm2 <= 1e-12 or t + 1 >= self.max_iter or \
                    (self.time_limit is not None and time.time() - t0 > self.time_limit):
                break
            # step
            if self.step_rule == "polyak" and np.isfinite(best_ub):
                alpha = theta * (best_ub - L) / (gnorm2 + self.eps)
            else:
                alpha = self.alpha0 / np.sqrt(t + 1)
            lam_new = self.project(lam + alpha * g)
            L_new, ys_new = self.evaluate(lam_new)
            if L_new is None:                   # deadline hit inside the iteration: keep the last valid state
                self.timed_out = True
                break
            lam, L, ys = lam_new, L_new, ys_new
            t += 1
            if L > best_lb + 1e-9:
                best_lb, best_lam, best_ys, no_improve = L, lam.copy(), ys.copy(), 0
            else:
                no_improve += 1
                if self.step_rule == "polyak" and no_improve >= self.theta_patience:
                    theta, no_improve = max(self.theta_min, 0.5 * theta), 0
                    if self.reset_to_best:              # restart from the best multiplier (no re-solve needed)
                        lam, ys, L = best_lam.copy(), best_ys.copy(), best_lb
        gap = (best_ub - best_lb) / max(abs(best_ub), 1e-9) if np.isfinite(best_ub) else np.inf
        return LDResult(best_lb=best_lb, best_ub=best_ub, best_y=best_y, gap=gap, iterations=t + 1,
                        seconds=time.time() - t0, n_bundle_solves=self.n_bundle_solves,
                        n_recourse_solves=self.n_recourse_solves, start_source=src,
                        lam_final=best_lam, history=history, best_eval=best_ev,
                        n_unproved_bundle_solves=self.n_unproved, n_design_evals=self.n_design_evals)


# ---------------------------------------------------------------------- diagnostics
def time_to_target(history: List[Dict], target_lb: float, key: str = "time") -> Optional[float]:
    """T(lambda^0): first time (or iteration, key='t') at which best_lb >= target_lb; None if never."""
    for h in history:
        if h["best_lb"] >= target_lb - 1e-9:
            return h[key]
    return None


def oracle_headroom_capture(T_zero: float, T_ml: float, T_oracle: float, eps: float = 1e-9,
                            min_headroom: float = 0.10) -> Optional[float]:
    """OHC = (T(0) - T(ML)) / (T(0) - T(oracle) + eps), reported only if the relative headroom
    H = (T(0) - T(oracle)) / T(0) is at least `min_headroom` (audit 5.3)."""
    if T_zero is None or T_ml is None or T_oracle is None or T_zero <= 0:
        return None
    H = (T_zero - T_oracle) / T_zero
    if H < min_headroom:
        return None
    return (T_zero - T_ml) / (T_zero - T_oracle + eps)
