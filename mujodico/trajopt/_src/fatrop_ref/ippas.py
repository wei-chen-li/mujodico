"""IP-PAS: a fully jitted JAX port of the FATROP interior-point algorithm.

The complete nonlinear algorithm -- barrier update, inertia-corrected search
direction with iterative refinement, filter line search with second-order
corrections, watchdog, tiny-step detection, soft restoration and the
minimum-L1 restoration phase -- runs inside a single ``jax.lax.while_loop``.
A solve therefore performs no host synchronization until it returns, and the
compiled executable is reused across problem instances (initial state, goal,
weights ... are runtime ``params``).

The algorithm follows FATROP master (``meco-group/fatrop``, identical in its
algorithmic content to the 1.1.8 release bundled with CasADi) source file by
source file; the corresponding FATROP routine is named in each docstring.
The structured linear algebra is in :mod:`kkt` and offers the serial Riccati
recursion and the parallel associative scan (PAS) backends.

Problem class (one stage function set, ``k = 0..N-1``)::

    min   sum_k l(x_k, w_k; p) + l_N(x_N; p)
    s.t.  x_0 = x0(p),
          x_{k+1} = f(x_k, w_k),
          h(x_k, w_k) = 0,
          lb <= g(x_k, w_k) <= ub.

As in FATROP every inequality row gets a slack ``s`` with ``g - s = 0`` and
bounds on ``s``; the initial state is an equality-constrained variable.
"""
from __future__ import annotations

from dataclasses import dataclass
import functools
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from mujodico.trajopt._src.fatrop_ref import kkt

Array = jax.Array

# Termination status codes.
CONTINUE, SUCCESS, ACCEPTABLE, MAX_ITER, LOCAL_INFEASIBILITY, \
    LINESEARCH_FAILED, STEP_ERROR = range(7)
STATUS_NAMES = {
    SUCCESS: 'success', ACCEPTABLE: 'acceptable', MAX_ITER: 'max_iter',
    LOCAL_INFEASIBILITY: 'local_infeasibility',
    LINESEARCH_FAILED: 'restoration_failed', STEP_ERROR: 'step_error'}
# Iteration type codes (FATROP's alpha_primal_type character).
TYPE_CHARS = 'xfhFHwsSt?'
T_X, T_F, T_H, T_FS, T_HS, T_W, T_SOFT, T_SOFT_ORIG, T_TINY, T_UNKNOWN = range(
    10)
# Pending work at the start of a loop pass.
P_NONE, P_CHECK, P_SOFT, P_INIT = range(4)


@dataclass(frozen=True)
class IpPasOptions:
  """FATROP option names and defaults (see the FATROP headers)."""
  max_iter: int = 1000
  tol: float = 1e-8
  acceptable_tol: float = 1e-6
  acceptable_iter: int = 15
  constr_viol_tol: float = 1e-4
  # IpMonotoneMuUpdate
  mu_init: float = 1e-1
  barrier_tol_factor: float = 10.0
  mu_linear_decrease_factor: float = 0.2
  mu_superlinear_decrease_power: float = 1.5
  tau_min: float = 0.99
  compl_inf_tol: float = 1e-4
  mu_allow_fast_monotone_decrease: bool = True
  # Extensions (not in FATROP; the defaults keep FATROP's behavior):
  # ``mu_min`` floors the barrier parameter; ``mu_update_max_iter >= 0``
  # stops barrier updates after that iteration.  ``mu_update_max_iter=1``
  # reproduces the pre-fix ``FatropReferenceJAX`` used for the paper, whose
  # stale error estimate froze mu after the first iteration.
  mu_min: float = 0.0
  mu_update_max_iter: int = -1
  # IpIterate
  kappa_d: float = 1e-5
  smax: float = 100.0
  kappa_sigma: float = 1e10
  # IpNlpOrig / IpInitializer / IpEqMultInitializer
  bound_relax_factor: float = 1e-8
  bound_push: float = 1e-2
  bound_frac: float = 1e-2
  lam_max: float = 1e3
  # IpSearchDirImpl / AugSystemSolver / LinearSolver
  delta_w0: float = 1e-4
  delta_wmin: float = 1e-20
  kappa_wmin: float = 1.0 / 3.0
  kappa_wplus: float = 8.0
  kappa_wplusem: float = 100.0
  delta_w_max: float = 1e40
  pivot_tol: float = 1e-8
  max_it_ref: int = 5
  min_it_ref: int = 0
  iref_tol: float = 1e-8
  # IpLinesearch
  max_soc: int = 4
  kappa_soc: float = 0.99
  theta_min_fact: float = 1e-4
  theta_max_fact: float = 1e4
  s_phi: float = 2.3
  s_theta: float = 1.1
  delta: float = 1.0
  tiny_step_tol: float = 10 * float(np.finfo(np.float64).eps)
  tiny_step_y_tol: float = 1e-2
  gamma_theta: float = 1e-5
  gamma_phi: float = 1e-8
  eta_phi: float = 1e-8
  alpha_min_frac: float = 0.05
  max_filter_resets: int = -1
  filter_reset_trigger: int = 5
  obj_max_incr: float = 5.0
  watchdog_shortened_iter_trigger: int = 10
  watchdog_trial_iter_max: int = 3
  alpha_red_factor: float = 0.5
  soft_rest_pd_error_reduction_factor: float = 1.0 - 1e-4
  max_soft_resto_iters: int = 10
  # Restoration (IpNlpResto / IpInitializerResto / IpConvergenceCheckResto /
  # IpRestoPhaseMinCl1)
  resto_rho: float = 1000.0
  # Whether the initial-state equality receives elastic variables in the
  # restoration phase.  With CasADi's FATROP interface (the reference
  # implementation used for validation) the constraint ``X[0] == x0`` is
  # detected as the gap-closing row of a leading interval and therefore stays
  # hard, exactly like the dynamics rows.  Keep that behavior by default.
  resto_relax_initial: bool = False
  # FATROP 1.1.8/master calls the least-squares multiplier initializer of the
  # *restoration* data object at the end of a successful restoration phase,
  # so the original iterate keeps the equality multipliers of its last trial
  # point (the soft-restoration attempt).  ``False`` reproduces that exactly;
  # ``True`` re-estimates them (the evident intent, as in IPOPT).
  resto_handoff_ls_multipliers: bool = False
  required_infeasibility_reduction: float = 0.9
  bound_mult_reset_threshold: float = 1e3
  # Implementation
  max_ls: int = 64           # safety cap on line-search passes
  # Evaluate this many backtracking step lengths per model evaluation
  # (alpha, alpha*r, alpha*r^2, ...) and cache them.  The acceptance logic is
  # unchanged; 0 selects 4 on GPU (evaluations are latency bound) and 1 on CPU.
  ls_batch: int = 0
  max_inertia_tries: int = 60  # safety cap on inertia corrections
  record_trace: bool = True


# ---------------------------------------------------------------------------
# Pytrees
# ---------------------------------------------------------------------------
class Rows(NamedTuple):
  """Equality-constraint space: initial, dynamics, path eq., ineq. rows."""
  c0: Array  # (nx,)
  cd: Array  # (N, nx)
  ch: Array  # (N, ne)
  cg: Array  # (N, ni)


class ERows(NamedTuple):
  """Rows that receive elastic variables in restoration (not dynamics)."""
  c0: Array
  ch: Array
  cg: Array


class Iterate(NamedTuple):
  x: Array      # (N+1, nx)
  w: Array      # (N, nw)
  s: Array      # (N, ni)
  y: Rows       # equality multipliers
  zL: Array     # (N, ni)
  zU: Array     # (N, ni)
  p: ERows      # restoration elastics (unused in the original problem)
  n: ERows
  zp: ERows
  zn: ERows
  mu: Array


class Direction(NamedTuple):
  dx: Array
  dw: Array
  ds: Array
  dy: Rows
  dzL: Array
  dzU: Array
  dp: ERows
  dn: ERows
  dzp: ERows
  dzn: ERows


class Lin(NamedTuple):
  """Function values and derivatives at a primal point."""
  obj: Array
  gx: Array     # (N+1, nx) objective gradient
  gw: Array     # (N, nw)
  Hq: Array     # (N, nx, nx) objective Hessian blocks
  Hs: Array     # (N, nx, nw)
  Hr: Array     # (N, nw, nw)
  Hf: Array     # (nx, nx)
  Cq: Array     # constraint curvature (exact Lagrangian mode only)
  Cs: Array
  Cr: Array
  c: Rows       # original constraint violation (g row: g(x,w) - s)
  graw: Array   # (N, ni) raw inequality values g(x, w)
  A: Array
  B: Array
  E: Array
  F: Array
  Gx: Array
  Gw: Array


class Trial(NamedTuple):
  """Cheap function values at a trial point."""
  obj: Array
  c: Rows       # original constraint violation incl. -s
  graw: Array


class FullRhs(NamedTuple):
  """Right-hand side of FATROP's full primal-dual system."""
  fx: Array     # (N+1, nx)
  fw: Array     # (N, nw)
  fs: Array     # (N, ni)
  g: Rows
  cl: Array
  cu: Array
  fp: ERows
  fn: ERows
  cp: ERows
  cn: ERows


class LSState(NamedTuple):
  filt_phi: Array
  filt_theta: Array
  filt_n: Array
  theta_min: Array
  theta_max: Array
  filter_reset_count: Array
  filter_reject_count: Array
  last_rejection_due_to_filter: Array
  in_watchdog: Array
  watchdog_shortened_iter_count: Array
  watchdog_trial_iter_count: Array
  watchdog_theta: Array
  watchdog_barr: Array
  watchdog_gbd: Array
  watchdog_alpha_test: Array
  tiny_step_last_iteration: Array
  in_soft_resto: Array
  soft_resto_counter: Array
  last_mu: Array
  ref_theta: Array
  ref_barr: Array
  ref_gbd: Array


class AlgState(NamedTuple):
  """State owned by one ``IpAlgorithm`` instance (original or resto)."""
  ls: LSState
  delta_w_last: Array
  mu_initialized: Array
  tiny_step_flag: Array
  acceptable_counter: Array
  stored_it: Iterate
  stored_lin: Lin
  stored_dir: Direction


class StepInfo(NamedTuple):
  alpha_pr: Array
  alpha_du: Array
  ls: Array
  typ: Array
  d_norm: Array
  rg: Array


class Refs(NamedTuple):
  """Restoration reference point and the original iterate at entry."""
  xref: Array
  wref: Array
  sref: Array
  zeta: Array
  orig_it: Iterate
  orig_lin: Lin
  orig_dir: Direction
  orig_inf_pr: Array
  trial_y: Rows        # multipliers of the last original trial point


class State(NamedTuple):
  status: Array
  pending: Array
  need_lin: Array
  it: Array
  mode: Array          # 0 original problem, 1 restoration
  cur: Iterate
  lin: Lin
  dir: Direction
  prev: Iterate        # iterate before a pending soft-restoration step
  prev_lin: Lin
  prev_dir: Direction
  soft_kind: Array     # 0 in-phase, 1 after-failure
  info: StepInfo
  orig: AlgState
  resto: AlgState
  refs: Refs
  first_resto_iter: Array
  n_resto: Array
  trace: Array
  trace_n: Array
  max_iter: Array
  stats: Array         # [KKT solves, model evaluations, linearizations]


def _tmap(f, *trees):
  return jax.tree.map(f, *trees)


def _where(c, a, b):
  return jax.tree.map(lambda u, v: jnp.where(c, u, v), a, b)


def _sum(tree):
  return sum(jnp.sum(l) for l in jax.tree.leaves(tree))


def _l1(tree):
  return sum(jnp.sum(jnp.abs(l)) for l in jax.tree.leaves(tree))


def _linf(tree):
  leaves = [jnp.max(jnp.abs(l), initial=0.0) for l in jax.tree.leaves(tree)]
  return functools.reduce(jnp.maximum, leaves) if leaves else jnp.asarray(0.)


def _finite(tree):
  leaves = [jnp.all(jnp.isfinite(l)) for l in jax.tree.leaves(tree)]
  return functools.reduce(jnp.logical_and, leaves)


def _cast_like(out, ref):
  """Cast a pytree to the dtypes/shapes of ``ref`` (stable loop carries)."""
  return jax.tree.map(
      lambda o, r: jnp.broadcast_to(jnp.asarray(o, dtype=jnp.asarray(r).dtype),
                                    jnp.shape(r)), out, ref)


def _scond(pred, f, g, x):
  """``lax.cond`` for state -> state transforms with dtype-stable output."""
  return jax.lax.cond(pred, lambda v: _cast_like(f(v), v),
                      lambda v: _cast_like(g(v), v), x)


def _sswitch(i, fs, x):
  return jax.lax.switch(
      i, [(lambda v, f=f: _cast_like(f(v), v)) for f in fs], x)


def _swhile(cond, body, x):
  return jax.lax.while_loop(cond, lambda v: _cast_like(body(v), v), x)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
class IpPasSolver:
  """FATROP-faithful IPM for stagewise OCPs with a PAS/serial KKT backend.

  Args:
    horizon, nx, nw: problem dimensions.
    stage_cost: ``l(x, w, params)``.
    terminal_cost: ``l_N(x, params)``.
    dynamics: ``f(x, w)``.
    stage_equality: ``h(x, w)`` (may return an empty vector).
    stage_inequality: ``g(x, w)`` with row bounds ``ineq_lb``/``ineq_ub``.
    hessian_mode: ``'cost'`` (Gauss-Newton: objective Hessian only, the
      FATROP/MJX benchmark setting) or ``'exact_lagrangian'``.
    linear_backend: ``'pas'`` or ``'serial'``.
    options: :class:`IpPasOptions`.
  """

  def __init__(self, *, horizon: int, nx: int, nw: int,
               stage_cost: Callable, terminal_cost: Callable,
               dynamics: Callable, stage_equality: Callable,
               stage_inequality: Callable, ineq_lb, ineq_ub,
               hessian_mode: str = 'cost', linear_backend: str = 'pas',
               options: IpPasOptions = IpPasOptions(),
               eq_identity_cols=None, check_state=None,
               equality_takes_next: bool = False):
    if hessian_mode not in ('cost', 'exact_lagrangian'):
      raise ValueError(hessian_mode)
    if linear_backend not in ('pas', 'serial'):
      raise ValueError(linear_backend)
    if not jax.config.jax_enable_x64:
      raise RuntimeError('IpPasSolver requires jax_enable_x64=True.')
    self.N, self.nx, self.nw = int(horizon), int(nx), int(nw)
    self.stage_cost = stage_cost
    self.terminal_cost = terminal_cost
    self.dynamics = dynamics
    self.stage_equality = stage_equality
    self.stage_inequality = stage_inequality
    # stage_equality(x, w, x_next): x_next (the trajectory's next state) is
    # only a solver warm start; no derivatives are taken through it.
    self._eq_next = equality_takes_next
    self.hessian_mode = hessian_mode
    self.linear_backend = linear_backend
    self.opt = options
    z = jnp.zeros
    self.ne = int(stage_equality(z(nx), z(nw)).shape[0])
    self.ni = int(stage_inequality(z(nx), z(nw)).shape[0])
    lb = np.asarray(ineq_lb, dtype=np.float64).reshape(self.ni)
    ub = np.asarray(ineq_ub, dtype=np.float64).reshape(self.ni)
    if np.any(~np.isfinite(lb) & ~np.isfinite(ub)):
      raise ValueError('every inequality row needs a finite bound')
    # IpNlpOrig::get_bounds: relax the bounds.
    relax = lambda b: np.minimum(options.constr_viol_tol,
                                 options.bound_relax_factor *
                                 np.maximum(1.0, np.abs(b)))
    with np.errstate(invalid='ignore'):
      lb_r = np.where(np.isfinite(lb), lb - relax(lb), -np.inf)
      ub_r = np.where(np.isfinite(ub), ub + relax(ub), np.inf)
    self._lb = lb_r
    self._ub = ub_r
    self._flo = np.isfinite(lb_r)
    self._fhi = np.isfinite(ub_r)
    self._n_bounds_orig = self.N * int(self._flo.sum() + self._fhi.sum())
    self._n_eq = self.nx + self.N * (self.nx + self.ne + self.ni)
    self._n_elastic = (self.nx if options.resto_relax_initial else 0) + \
        self.N * (self.ne + self.ni)
    self._trace_cap = (options.max_iter + 2) * 2 if options.record_trace else 1
    m0 = 1.0 if options.resto_relax_initial else 0.0
    self._emask = ERows(np.full((self.nx,), m0), np.ones((self.N, self.ne)),
                        np.ones((self.N, self.ni)))
    self._filter_cap = options.max_iter + 8
    self._K = options.ls_batch or (4 if jax.default_backend() == 'gpu' else 1)
    # Optional exact condensation of path equalities with an identity block
    # in the columns ``eq_identity_cols`` of w (lifted contact forces).
    self._cond = None
    if eq_identity_cols is not None and linear_backend == 'pas' and self.ne:
      if_ = np.asarray(eq_identity_cols, dtype=np.int64)
      iu = np.setdiff1d(np.arange(self.nw), if_)
      xc = jnp.zeros(nx) if check_state is None else jnp.asarray(check_state)
      F0 = np.asarray(jax.jacfwd(stage_equality, argnums=1)(xc, jnp.zeros(nw)))
      if (if_.size != self.ne or
          not np.allclose(F0[:, if_], np.eye(self.ne), atol=1e-12)):
        raise ValueError('eq_identity_cols must index an identity block of '
                         'the stage equality Jacobian (one row per column)')
      self._cond = (iu, if_)
    self._solve = jax.jit(self._solve_impl)
    self._compiled = {}

  # -------------------------------------------------------------------------
  # Model evaluation
  # -------------------------------------------------------------------------
  def _cons(self, x, w, xn=None):
    eq = (self.stage_equality(x, w, jax.lax.stop_gradient(xn))
          if (self._eq_next and xn is not None) else self.stage_equality(x, w))
    return jnp.concatenate([self.dynamics(x, w), eq,
                            self.stage_inequality(x, w)])

  def _split(self, v):
    nx, ne = self.nx, self.ne
    return v[..., :nx], v[..., nx:nx + ne], v[..., nx + ne:]

  def _values(self, x, w, s, params) -> Trial:
    vals = jax.vmap(self._cons)(x[:-1], w, x[1:])
    fd, h, g = self._split(vals)
    costs = jax.vmap(self.stage_cost, in_axes=(0, 0, None))(x[:-1], w, params)
    obj = jnp.sum(costs) + self.terminal_cost(x[-1], params)
    c = Rows(x[0] - params['x0'], fd - x[1:], h, g - s)
    return Trial(obj, c, g)

  def _linearize(self, it: Iterate, params) -> Lin:
    x, w = it.x, it.w

    def stage(xk, wk, xnk):
      jac, val = jax.jacfwd(lambda a, b: (self._cons(a, b, xnk),) * 2,
                            argnums=(0, 1), has_aux=True)(xk, wk)
      return val, jac[0], jac[1]

    vals, Jx, Jw = jax.vmap(stage)(x[:-1], w, x[1:])
    fd, h, g = self._split(vals)
    nx, ne = self.nx, self.ne
    A, E, Gx = Jx[:, :nx], Jx[:, nx:nx + ne], Jx[:, nx + ne:]
    B, F, Gw = Jw[:, :nx], Jw[:, nx:nx + ne], Jw[:, nx + ne:]

    def cost(xk, wk):
      return self.stage_cost(xk, wk, params)

    gx, gw = jax.vmap(jax.grad(cost, argnums=(0, 1)))(x[:-1], w)
    (Hq, Hs), (_, Hr) = jax.vmap(jax.hessian(cost, argnums=(0, 1)))(x[:-1], w)
    costs = jax.vmap(cost)(x[:-1], w)
    term = lambda xN: self.terminal_cost(xN, params)
    obj = jnp.sum(costs) + term(x[-1])
    gf = jax.grad(term)(x[-1])
    Hf = jax.hessian(term)(x[-1])
    if self.hessian_mode == 'exact_lagrangian':
      yv = jnp.concatenate([it.y.cd, it.y.ch, it.y.cg], axis=1)

      def lag(xk, wk, yk, xnk):
        return jnp.dot(yk, self._cons(xk, wk, xnk))

      (Cq, Cs), (_, Cr) = jax.vmap(jax.hessian(lag, argnums=(0, 1)))(
          x[:-1], w, yv, x[1:])
    else:
      Cq = Cs = Cr = jnp.zeros((0,))
    c = Rows(x[0] - params['x0'], fd - x[1:], h, g - it.s)
    return Lin(obj, jnp.concatenate([gx, gf[None]], 0), gw, Hq, Hs, Hr, Hf,
               Cq, Cs, Cr, c, g, A, B, E, F, Gx, Gw)

  # -------------------------------------------------------------------------
  # IpIterate quantities
  # -------------------------------------------------------------------------
  def _bounds(self):
    lb = jnp.asarray(self._lb)[None]
    ub = jnp.asarray(self._ub)[None]
    flo = jnp.asarray(self._flo)[None]
    fhi = jnp.asarray(self._fhi)[None]
    return lb, ub, flo, fhi

  def _deltas(self, s):
    lb, ub, flo, fhi = self._bounds()
    Sl = jnp.where(flo, s - lb, 1.0)
    Su = jnp.where(fhi, ub - s, 1.0)
    return Sl, Su

  def _cviol(self, c: Rows, it: Iterate, mode) -> Rows:
    """``IpIterate::constr_viol`` (resto adds p - n on damped rows)."""
    r = jnp.asarray(mode, c.c0.dtype)
    m0 = self._emask.c0
    return Rows(c.c0 + r * m0 * (it.p.c0 - it.n.c0), c.cd,
                c.ch + r * (it.p.ch - it.n.ch), c.cg + r * (it.p.cg - it.n.cg))

  def _objective(self, it: Iterate, obj_orig, mode, refs: Refs):
    """Original objective or ``IpNlpResto::eval_objective``."""
    dx = (it.x - refs.xref) / jnp.maximum(1.0, jnp.abs(refs.xref))
    dw = (it.w - refs.wref) / jnp.maximum(1.0, jnp.abs(refs.wref))
    ds = (it.s - refs.sref) / jnp.maximum(1.0, jnp.abs(refs.sref))
    prox = 0.5 * refs.zeta * (jnp.sum(dx * dx) + jnp.sum(dw * dw) +
                              jnp.sum(ds * ds))
    resto = prox + self.opt.resto_rho * _sum(_tmap(
        lambda m, p, n: m * (p + n), self._emask, it.p, it.n))
    return jnp.where(mode == 1, resto, obj_orig)

  def _obj_grad(self, it: Iterate, lin: Lin, mode, refs: Refs):
    """Objective gradient w.r.t. (x, w, s) and the elastics (p, n)."""
    drx = 1.0 / jnp.maximum(1.0, jnp.abs(refs.xref))
    drw = 1.0 / jnp.maximum(1.0, jnp.abs(refs.wref))
    drs = 1.0 / jnp.maximum(1.0, jnp.abs(refs.sref))
    r = mode == 1
    gx = jnp.where(r, refs.zeta * drx * drx * (it.x - refs.xref), lin.gx)
    gw = jnp.where(r, refs.zeta * drw * drw * (it.w - refs.wref), lin.gw)
    gs = jnp.where(r, refs.zeta * drs * drs * (it.s - refs.sref), 0.0)
    rho = jnp.where(r, self.opt.resto_rho, 0.0)
    gp = _tmap(lambda m, a: m * jnp.ones_like(a) * rho, self._emask, it.p)
    return gx, gw, gs, gp

  def _barrier(self, it: Iterate, mode):
    """``IpIterate::barrier_value`` incl. the kappa_d damping term."""
    lb, ub, flo, fhi = self._bounds()
    Sl, Su = self._deltas(it.s)
    kd = self.opt.kappa_d
    lo1 = flo & ~fhi
    hi1 = fhi & ~flo
    val = (jnp.sum(jnp.where(flo, -jnp.log(Sl), 0.0)) +
           jnp.sum(jnp.where(fhi, -jnp.log(Su), 0.0)) +
           jnp.sum(jnp.where(lo1, kd * Sl, 0.0)) +
           jnp.sum(jnp.where(hi1, kd * Su, 0.0)))
    pn = sum(jnp.sum(m * (-jnp.log(q) + kd * q))
             for m, q in zip(jax.tree.leaves((self._emask, self._emask)),
                             jax.tree.leaves((it.p, it.n))))
    val = val + jnp.where(mode == 1, pn, 0.0)
    return it.mu * val

  def _lin_dec(self, it: Iterate, lin: Lin, d: Direction, mode, refs):
    """``linear_decrease_barrier + linear_decrease_objective``."""
    lb, ub, flo, fhi = self._bounds()
    Sl, Su = self._deltas(it.s)
    mu, kd = it.mu, self.opt.kappa_d
    lo1 = flo & ~fhi
    hi1 = fhi & ~flo
    bg = (jnp.where(flo, -mu / Sl, 0.0) + jnp.where(fhi, mu / Su, 0.0) +
          jnp.where(lo1, mu * kd, 0.0) + jnp.where(hi1, -mu * kd, 0.0))
    dec = jnp.sum(bg * d.ds)
    r = mode == 1
    pn = sum(jnp.sum(m * (-mu / q + mu * kd) * dq)
             for m, q, dq in zip(jax.tree.leaves((self._emask, self._emask)),
                                 jax.tree.leaves((it.p, it.n)),
                                 jax.tree.leaves((d.dp, d.dn))))
    dec = dec + jnp.where(r, pn, 0.0)
    gx, gw, gs, gp = self._obj_grad(it, lin, mode, refs)
    dec = dec + jnp.sum(gx * d.dx) + jnp.sum(gw * d.dw) + jnp.sum(gs * d.ds)
    dec = dec + jnp.where(r, _sum(_tmap(lambda a, b: a * b, gp, d.dp)) +
                          _sum(_tmap(lambda a, b: a * b, gp, d.dn)), 0.0)
    return dec

  def _dual_inf(self, it: Iterate, lin: Lin, mode, refs):
    """``dual_infeasibility_x`` and ``dual_infeasibility_s``."""
    gx, gw, gs, gp = self._obj_grad(it, lin, mode, refs)
    y = it.y
    mtv = lambda M, v: jnp.einsum('nji,nj->ni', M, v)
    rx = (gx[:-1] + mtv(lin.A, y.cd) + mtv(lin.E, y.ch) + mtv(lin.Gx, y.cg))
    rx = rx.at[0].add(y.c0)
    rx = rx.at[1:].add(-y.cd[:-1])
    rxf = gx[-1] - y.cd[-1]
    fx = jnp.concatenate([rx, rxf[None]], 0)
    fw = gw + mtv(lin.B, y.cd) + mtv(lin.F, y.ch) + mtv(lin.Gw, y.cg)
    fs = gs + it.zU - it.zL - y.cg
    r = jnp.asarray(mode == 1, fs.dtype)
    yr = ERows(y.c0, y.ch, y.cg)
    M = self._emask
    fp = _tmap(lambda m, g, z, l: m * r * (g - z + l), M, gp, it.zp, yr)
    fn = _tmap(lambda m, g, z, l: m * r * (g - z - l), M, gp, it.zn, yr)
    return fx, fw, fs, fp, fn

  def _compl(self, it: Iterate, mu, mode):
    """Relaxed complementarities ``z*delta - mu`` (zero where unbounded)."""
    lb, ub, flo, fhi = self._bounds()
    Sl, Su = self._deltas(it.s)
    cl = jnp.where(flo, Sl * it.zL - mu, 0.0)
    cu = jnp.where(fhi, Su * it.zU - mu, 0.0)
    r = jnp.asarray(mode == 1, cl.dtype)
    M = self._emask
    cp = _tmap(lambda m, q, z: m * r * (q * z - mu), M, it.p, it.zp)
    cn = _tmap(lambda m, q, z: m * r * (q * z - mu), M, it.n, it.zn)
    return cl, cu, cp, cn

  def _full_rhs(self, it, lin, mode, refs, mu=None, g=None) -> FullRhs:
    mu = it.mu if mu is None else mu
    fx, fw, fs, fp, fn = self._dual_inf(it, lin, mode, refs)
    cl, cu, cp, cn = self._compl(it, mu, mode)
    if g is None:
      g = self._cviol(lin.c, it, mode)
    return FullRhs(fx, fw, fs, g, cl, cu, fp, fn, cp, cn)

  def _e_mu(self, it: Iterate, lin: Lin, mu, mode, refs):
    """``IpIterate::e_mu``."""
    fx, fw, fs, fp, fn = self._dual_inf(it, lin, mode, refs)
    cl, cu, cp, cn = self._compl(it, mu, mode)
    cv = _linf(self._cviol(lin.c, it, mode))
    r = mode == 1
    du = jnp.maximum(_linf((fx, fw, fs)), jnp.where(r, _linf((fp, fn)), 0.0))
    co = jnp.maximum(_linf((cl, cu)), jnp.where(r, _linf((cp, cn)), 0.0))
    zl1 = (jnp.sum(jnp.abs(it.zL)) + jnp.sum(jnp.abs(it.zU)) +
           jnp.where(r, _l1(_tmap(lambda m, z: m * z, self._emask, it.zp)) +
                     _l1(_tmap(lambda m, z: m * z, self._emask, it.zn)), 0.0))
    nb = self._n_bounds_orig + jnp.where(r, 2 * self._n_elastic, 0)
    nb = jnp.maximum(nb, 1)
    lam_mean = (zl1 + _l1(it.y)) / (nb + self._n_eq)
    z_mean = zl1 / nb
    smax = self.opt.smax
    du = jnp.where(lam_mean > smax, du / (lam_mean / smax), du)
    co = jnp.where(z_mean > smax, co / (z_mean / smax), co)
    return jnp.maximum(cv, jnp.maximum(du, co)), cv, du, co

  def _pd_error(self, it, lin, mode, refs):
    """``primal_dual_error`` used by the soft restoration phase."""
    fx, fw, fs, fp, fn = self._dual_inf(it, lin, mode, refs)
    cl, cu, cp, cn = self._compl(it, it.mu, mode)
    c = self._cviol(lin.c, it, mode)
    r = mode == 1
    dx = (_l1((fx, fw)) + jnp.where(r, 0.0, 0.0)) / (fx.size + fw.size)
    ns = self.N * self.ni + jnp.where(r, 2 * self._n_eq, 0)
    ns = jnp.maximum(ns, 1)
    ds = (_l1(fs) + jnp.where(r, _l1((fp, fn)), 0.0)) / ns
    dc = _l1(c) / self._n_eq
    dl = (_l1(cl) + jnp.where(r, _l1(cp) + _l1(cn), 0.0)) / ns
    du = (_l1(cu) + jnp.where(r, 0.0, 0.0)) / ns
    return dx + ds + dc + dl + du

  def _max_step_primal(self, it: Iterate, ds, dp, dn, mode):
    tau = jnp.maximum(self.opt.tau_min, 1.0 - it.mu)
    lb, ub, flo, fhi = self._bounds()
    Sl, Su = self._deltas(it.s)
    a = jnp.minimum(jnp.min(jnp.where(flo & (ds < 0), -tau * Sl / ds, 1.0),
                            initial=1.0),
                    jnp.min(jnp.where(fhi & (ds > 0), tau * Su / ds, 1.0),
                            initial=1.0))
    pn = [jnp.min(jnp.where(dq < 0, -tau * q / dq, 1.0), initial=1.0)
          for q, dq in zip(jax.tree.leaves((it.p, it.n)),
                           jax.tree.leaves((dp, dn)))]
    apn = functools.reduce(jnp.minimum, pn, jnp.asarray(1.0))
    return jnp.minimum(1.0, jnp.where(mode == 1, jnp.minimum(a, apn), a))

  def _max_step_dual(self, it: Iterate, d: Direction, mode):
    tau = jnp.maximum(self.opt.tau_min, 1.0 - it.mu)
    lb, ub, flo, fhi = self._bounds()
    a = jnp.minimum(
        jnp.min(jnp.where(flo & (d.dzL < 0), -tau * it.zL / d.dzL, 1.0),
                initial=1.0),
        jnp.min(jnp.where(fhi & (d.dzU < 0), -tau * it.zU / d.dzU, 1.0),
                initial=1.0))
    pn = [jnp.min(jnp.where(dz < 0, -tau * z / dz, 1.0), initial=1.0)
          for z, dz in zip(jax.tree.leaves((it.zp, it.zn)),
                           jax.tree.leaves((d.dzp, d.dzn)))]
    apn = functools.reduce(jnp.minimum, pn, jnp.asarray(1.0))
    return jnp.minimum(1.0, jnp.where(mode == 1, jnp.minimum(a, apn), a))

  def _modify_dual_bounds(self, it: Iterate, mu, mode):
    """``IpIterate::modify_dual_bounds`` (kappa_sigma safeguard)."""
    ks = self.opt.kappa_sigma
    lb, ub, flo, fhi = self._bounds()
    Sl, Su = self._deltas(it.s)
    clamp = lambda z, d: jnp.maximum(jnp.minimum(z, ks * mu / d),
                                     mu / (ks * d))
    zL = jnp.where(flo, clamp(it.zL, Sl), 0.0)
    zU = jnp.where(fhi, clamp(it.zU, Su), 0.0)
    r = mode == 1
    zp = _tmap(lambda z, q: jnp.where(r, clamp(z, q), z), it.zp, it.p)
    zn = _tmap(lambda z, q: jnp.where(r, clamp(z, q), z), it.zn, it.n)
    return it._replace(zL=zL, zU=zU, zp=zp, zn=zn)

  def _set_primal(self, it: Iterate, d: Direction, alpha, mode):
    r = mode == 1
    return it._replace(
        x=it.x + alpha * d.dx, w=it.w + alpha * d.dw, s=it.s + alpha * d.ds,
        p=_tmap(lambda q, dq: jnp.where(r, q + alpha * dq, q), it.p, d.dp),
        n=_tmap(lambda q, dq: jnp.where(r, q + alpha * dq, q), it.n, d.dn))

  def _dual_step(self, cur: Iterate, d: Direction, trial: Iterate, a_pr, a_du,
                 mode):
    """``IpLinesearch::perform_dual_step`` (applied on the trial iterate)."""
    lb, ub, flo, fhi = self._bounds()
    r = mode == 1
    t = trial._replace(
        y=_tmap(lambda y, dy: y + a_pr * dy, cur.y, d.dy),
        zL=jnp.where(flo, cur.zL + a_du * d.dzL, 0.0),
        zU=jnp.where(fhi, cur.zU + a_du * d.dzU, 0.0),
        zp=_tmap(lambda z, dz: jnp.where(r, z + a_du * dz, z), cur.zp, d.dzp),
        zn=_tmap(lambda z, dz: jnp.where(r, z + a_du * dz, z), cur.zn, d.dzn))
    return self._modify_dual_bounds(t, cur.mu, mode)

  # -------------------------------------------------------------------------
  # Linear algebra (PdSolverOrig / PdSolverResto + LinearSolver)
  # -------------------------------------------------------------------------
  def _assemble(self, it: Iterate, lin: Lin, mode, refs, delta_w, dual_init):
    """Matrices of the reduced system and the data to (de)reduce RHSs."""
    N, nx, nw, ni = self.N, self.nx, self.nw, self.ni
    r = mode == 1
    dt = lin.Hq.dtype
    if dual_init:
      # IpEqMultInitializer: zero Hessian, Dx = 1, Sl = Su = 1, Z = 0.
      Q = jnp.zeros_like(lin.Hq)
      S = jnp.zeros_like(lin.Hs)
      R = jnp.zeros_like(lin.Hr)
      Qf = jnp.zeros_like(lin.Hf)
      Dxx = jnp.ones((N, nx), dt)
      Dxw = jnp.ones((N, nw), dt)
      Dxf = jnp.ones((nx,), dt)
      Dxs = jnp.ones((N, ni), dt)
      Sigma = Dxs
      Sl = Su = jnp.ones((N, ni), dt)
      zL = zU = jnp.zeros((N, ni), dt)
      Xp = Xn = _tmap(jnp.zeros_like, it.p)
      resto = jnp.asarray(False)
    else:
      scale = jnp.where(r, 0.0, 1.0)
      Q, S, R, Qf = (scale * lin.Hq, scale * lin.Hs, scale * lin.Hr,
                     scale * lin.Hf)
      if self.hessian_mode == 'exact_lagrangian':
        Q, S, R = Q + lin.Cq, S + lin.Cs, R + lin.Cr
      drx = 1.0 / jnp.maximum(1.0, jnp.abs(refs.xref))
      drw = 1.0 / jnp.maximum(1.0, jnp.abs(refs.wref))
      drs = 1.0 / jnp.maximum(1.0, jnp.abs(refs.sref))
      damp = jnp.where(r, refs.zeta, 0.0)
      Dx_all = delta_w + damp * drx * drx
      Dxx, Dxf = Dx_all[:-1], Dx_all[-1]
      Dxw = delta_w + damp * drw * drw
      Dxs = delta_w + damp * drs * drs
      Sl, Su = self._deltas(it.s)
      lb, ub, flo, fhi = self._bounds()
      zL, zU = it.zL, it.zU
      Sigma = (Dxs + jnp.where(flo, zL / Sl, 0.0) +
               jnp.where(fhi, zU / Su, 0.0))
      Xp = _tmap(lambda m, q, z: m * jnp.where(r, q / (q * delta_w + z), 0.0),
                 self._emask, it.p, it.zp)
      Xn = _tmap(lambda m, q, z: m * jnp.where(r, q / (q * delta_w + z), 0.0),
                 self._emask, it.n, it.zn)
      resto = r
    De = Xp.ch + Xn.ch
    De0 = Xp.c0 + Xn.c0
    Dii = 1.0 / Sigma + Xp.cg + Xn.cg
    sys = kkt.KKTSystem(Q, S, R, Qf, Dxx, Dxw, Dxf, lin.A, lin.B, lin.E,
                        lin.F, lin.Gx, lin.Gw, De, Dii, De0, resto)
    red = dict(Sigma=Sigma, Sl=Sl, Su=Su, zL=zL, zU=zU, Xp=Xp, Xn=Xn,
               Dxs=Dxs, p=it.p, n=it.n, zp=it.zp, zn=it.zn,
               dpn=jnp.asarray(delta_w, dt), resto=resto)
    return sys, red

  def _reduce(self, red, b: FullRhs) -> kkt.KKTRhs:
    lb, ub, flo, fhi = self._bounds()
    Sl, Su, Sigma = red['Sl'], red['Su'], red['Sigma']
    ss = b.fs + jnp.where(flo, b.cl / Sl, 0.0) - jnp.where(fhi, b.cu / Su, 0.0)
    fpp = _tmap(lambda f, c, q: f + c / q, b.fp, b.cp, red['p'])
    fnn = _tmap(lambda f, c, q: f + c / q, b.fn, b.cn, red['n'])
    gmod = lambda g, xp, fp_, xn, fn_: g - xp * fp_ + xn * fn_
    c0 = gmod(b.g.c0, red['Xp'].c0, fpp.c0, red['Xn'].c0, fnn.c0)
    ch = gmod(b.g.ch, red['Xp'].ch, fpp.ch, red['Xn'].ch, fnn.ch)
    cg = gmod(b.g.cg, red['Xp'].cg, fpp.cg, red['Xn'].cg, fnn.cg)
    gii = cg + ss / Sigma
    return kkt.KKTRhs(b.fx[:-1], b.fw, b.fx[-1], b.g.cd, ch, gii, c0)

  def _dereduce(self, red, b: FullRhs, sol: kkt.KKTSolution) -> Direction:
    lb, ub, flo, fhi = self._bounds()
    Sl, Su, Sigma = red['Sl'], red['Su'], red['Sigma']
    ss = b.fs + jnp.where(flo, b.cl / Sl, 0.0) - jnp.where(fhi, b.cu / Su, 0.0)
    ds = (sol.dgamma - ss) / Sigma
    dzL = jnp.where(flo, (-b.cl - red['zL'] * ds) / Sl, 0.0)
    dzU = jnp.where(fhi, (-b.cu + red['zU'] * ds) / Su, 0.0)
    dy = Rows(sol.dlam0, sol.dlam, sol.dbeta, sol.dgamma)
    dye = ERows(sol.dlam0, sol.dbeta, sol.dgamma)
    fpp = _tmap(lambda f, c, q: f + c / q, b.fp, b.cp, red['p'])
    fnn = _tmap(lambda f, c, q: f + c / q, b.fn, b.cn, red['n'])
    dp = _tmap(lambda X, f, l: X * (-f - l), red['Xp'], fpp, dye)
    dn = _tmap(lambda X, f, l: X * (-f + l), red['Xn'], fnn, dye)
    r = red['resto']
    dzp = _tmap(lambda c, z, d, q: jnp.where(r, (-c - z * d) / q, 0.0),
                b.cp, red['zp'], dp, red['p'])
    dzn = _tmap(lambda c, z, d, q: jnp.where(r, (-c - z * d) / q, 0.0),
                b.cn, red['zn'], dn, red['n'])
    return Direction(sol.dx, sol.dw, ds, dy, dzL, dzU, dp, dn, dzp, dzn)

  def _full_residual(self, sys, red, b: FullRhs, d: Direction) -> FullRhs:
    """Residual of the full primal-dual system (``apply_on_right`` + rhs)."""
    mv = lambda M, v: jnp.einsum('nij,nj->ni', M, v)
    mtv = lambda M, v: jnp.einsum('nji,nj->ni', M, v)
    y = d.dy
    rx = (mv(sys.Q, d.dx[:-1]) + sys.Dxx * d.dx[:-1] + mv(sys.S, d.dw) +
          mtv(sys.A, y.cd) + mtv(sys.E, y.ch) + mtv(sys.Gx, y.cg))
    rx = rx.at[0].add(y.c0).at[1:].add(-y.cd[:-1]) + b.fx[:-1]
    rxf = sys.Qf @ d.dx[-1] + sys.Dxf * d.dx[-1] - y.cd[-1] + b.fx[-1]
    rw = (mtv(sys.S, d.dx[:-1]) + mv(sys.R, d.dw) + sys.Dxw * d.dw +
          mtv(sys.B, y.cd) + mtv(sys.F, y.ch) + mtv(sys.Gw, y.cg) + b.fw)
    rs = red['Dxs'] * d.ds - y.cg - d.dzL + d.dzU + b.fs
    r = jnp.asarray(red['resto'], rs.dtype)
    dye = ERows(y.c0, y.ch, y.cg)
    dpn = red['dpn']
    M = self._emask
    rp = _tmap(lambda m, dp, l, dz, f: m * r * (dpn * dp + l - dz + f),
               M, d.dp, dye, d.dzp, b.fp)
    rn = _tmap(lambda m, dn, l, dz, f: m * r * (dpn * dn - l - dz + f),
               M, d.dn, dye, d.dzn, b.fn)
    pn = _tmap(lambda m, a, c: m * r * (a - c), M, d.dp, d.dn)
    g0 = d.dx[0] + pn.c0 + b.g.c0
    gd = mv(sys.A, d.dx[:-1]) + mv(sys.B, d.dw) - d.dx[1:] + b.g.cd
    gh = mv(sys.E, d.dx[:-1]) + mv(sys.F, d.dw) + pn.ch + b.g.ch
    gg = mv(sys.Gx, d.dx[:-1]) + mv(sys.Gw, d.dw) - d.ds + pn.cg + b.g.cg
    lb, ub, flo, fhi = self._bounds()
    rl = jnp.where(flo, red['zL'] * d.ds + red['Sl'] * d.dzL + b.cl, 0.0)
    ru = jnp.where(fhi, -red['zU'] * d.ds + red['Su'] * d.dzU + b.cu, 0.0)
    rcp = _tmap(lambda m, z, dq, q, dz, c: m * r * (z * dq + q * dz + c),
                M, red['zp'], d.dp, red['p'], d.dzp, b.cp)
    rcn = _tmap(lambda m, z, dq, q, dz, c: m * r * (z * dq + q * dz + c),
                M, red['zn'], d.dn, red['n'], d.dzn, b.cn)
    return FullRhs(jnp.concatenate([rx, rxf[None]], 0), rw, rs,
                   Rows(g0, gd, gh, gg), rl, ru, rp, rn, rcp, rcn)

  def _kkt_solve(self, sys, rhs):
    fn = kkt.solve_pas if self.linear_backend == 'pas' else kkt.solve_serial
    return fn(sys, rhs, self.opt.pivot_tol)

  def _solve_full(self, sys, red, b: FullRhs, factor: bool, fac=None,
                  return_fac=False):
    """``LinearSolver::solve_in_place`` with iterative refinement.

    The factorization/solve appears once (one compiled PAS instance): loop
    pass 0 solves with ``b``; later passes solve for the refinement
    correction with the residual as right-hand side.
    Returns ``(direction, flag)`` with flag 0 success, 1 indefinite, 2 NaN.
    """
    o = self.opt
    if self._cond is not None:
      # Hard path equalities with an identity block are condensed out;
      # restoration (regularized rows) uses the full saddle-point path.
      iu, if_ = self._cond

      def impl_cond(_):
        cd = kkt.condense(sys, iu, if_)
        f = kkt.factor_pas(cd.sys, o.pivot_tol) if fac is None else fac

        def kkt_c(bb):
          rb = self._reduce(red, bb)
          s_ = kkt.solve_factored(f, cd.sys, kkt.condense_rhs(cd, sys, rb, if_))
          return self._dereduce(red, bb, kkt.expand(cd, sys, rb, s_, if_)), f.ok
        return self._refine(sys, red, b, factor, kkt_c) + (f,)

      def impl_full(_):
        f_full = kkt.factor_pas(sys, o.pivot_tol)

        def kkt_f(bb):
          s_ = kkt.solve_factored(f_full, sys, self._reduce(red, bb))
          return self._dereduce(red, bb, s_), f_full.ok
        out = self._refine(sys, red, b, factor, kkt_f)
        shape = jax.eval_shape(lambda: kkt.factor_pas(
            kkt.condense(sys, iu, if_).sys, o.pivot_tol))
        return out + (_tmap(lambda l: jnp.zeros(l.shape, l.dtype), shape),)

      res = jax.lax.cond(sys.resto, impl_full, impl_cond, None)
      return res if return_fac else res[:3]
    if self.linear_backend == 'pas':
      # Factor once; every refinement solve reuses the factorization.
      if fac is None:
        fac = kkt.factor_pas(sys, o.pivot_tol)

      def kkt_(bb):
        sol_ = kkt.solve_factored(fac, sys, self._reduce(red, bb))
        return self._dereduce(red, bb, sol_), fac.ok
    else:
      def kkt_(bb):
        sol_, ok_ = self._kkt_solve(sys, self._reduce(red, bb))
        return self._dereduce(red, bb, sol_), ok_
    fac_out = fac if fac is not None else jnp.zeros((0,))
    out = self._refine(sys, red, b, factor, kkt_)
    return out + (fac_out,) if return_fac else out

  def _refine(self, sys, red, b, factor, kkt_):
    """FATROP ``LinearSolver::apply_iterative_refinement`` around ``kkt_``."""
    o = self.opt
    if o.max_it_ref == 0:
      d, ok = kkt_(b)
      return d, jnp.where(ok | (not factor), 0, 1), jnp.asarray(1)

    def cond(c):
      return ~c['done']

    def body(c):
      i = c['i']
      dc, ok_i = kkt_(c['rhs'])
      xc = _tmap(lambda a, b2: a + b2, c['xc'], dc)
      ok = jnp.where(i == 0, ok_i, c['ok'])
      indef = ((i == 0) & ~ok_i) if factor else jnp.asarray(False)
      finite = _finite(xc)
      rr = self._full_residual(sys, red, b, xc)
      res = _linf(rr)
      inc = res >= c['res_prev']
      xb = _where(inc, c['xb'], xc)
      stop = (indef | (~finite) | inc | ((res < o.iref_tol) &
                                         (i >= o.min_it_ref)) |
              (i == o.max_it_ref - 1))
      return dict(i=i + 1, rhs=rr, xb=xb, xc=xc, res_prev=res, done=stop,
                  nan=~finite & ~indef, ok=ok)

    zd = self._zero_dir(b)
    init = dict(i=jnp.asarray(0), rhs=b, xb=zd, xc=zd,
                res_prev=jnp.asarray(jnp.inf, b.fs.dtype),
                done=jnp.asarray(False), nan=jnp.asarray(False),
                ok=jnp.asarray(True))
    c = _swhile(cond, body, init)
    flag = jnp.where(c['nan'], 2, 0)
    if factor:
      flag = jnp.where(c['ok'], flag, 1)
    return c['xb'], flag, c['i']

  def _zero_dir(self, b: FullRhs) -> Direction:
    z = jnp.zeros_like
    return Direction(z(b.fx), z(b.fw), z(b.fs), _tmap(z, b.g), z(b.cl),
                     z(b.cu), _tmap(z, b.fp), _tmap(z, b.fn), _tmap(z, b.cp),
                     _tmap(z, b.cn))

  # -------------------------------------------------------------------------
  # Initialization (IpInitializer, IpEqMultInitializer)
  # -------------------------------------------------------------------------
  def _eq_mult_init(self, it: Iterate, lin: Lin, refs):
    """Least-squares equality multipliers; zero if too large or failed."""
    zero_y = _tmap(jnp.zeros_like, it.y)
    it0 = it._replace(y=zero_y)
    fx, fw, fs, _, _ = self._dual_inf(it0, lin, 0, refs)
    z = lambda a: jnp.zeros_like(a)
    b = FullRhs(fx, fw, fs, _tmap(z, it.y), z(it.zL), z(it.zU),
                _tmap(z, it.p), _tmap(z, it.n), _tmap(z, it.p),
                _tmap(z, it.n))
    sys, red = self._assemble(it0, lin, 0, refs, 0.0, dual_init=True)
    d, flag, _ = self._solve_full(sys, red, b, factor=True)
    ok = (flag == 0) & (_linf(d.dy) < self.opt.lam_max)
    return _where(ok, d.dy, zero_y)

  def _init_slacks(self, graw):
    o = self.opt
    lb, ub, flo, fhi = self._bounds()
    with np.errstate(invalid='ignore'):
      width = np.where(self._flo & self._fhi, self._ub - self._lb, np.inf)
    width = jnp.asarray(width)[None]
    pl = jnp.minimum(o.bound_push * jnp.maximum(1.0, jnp.abs(lb)),
                     o.bound_frac * width)
    pr = jnp.minimum(o.bound_push * jnp.maximum(1.0, jnp.abs(ub)),
                     o.bound_frac * width)
    dbl = jnp.minimum(jnp.maximum(graw, lb + pl), ub - pr)
    lo = jnp.maximum(graw, lb + o.bound_push * jnp.maximum(1.0, jnp.abs(lb)))
    hi = jnp.minimum(graw, ub - o.bound_push * jnp.maximum(1.0, jnp.abs(ub)))
    return jnp.where(flo & fhi, dbl, jnp.where(flo, lo, hi))

  # -------------------------------------------------------------------------
  # Filter line search (IpLinesearch)
  # -------------------------------------------------------------------------
  def _ls_reset(self, dt):
    cap = self._filter_cap
    f = lambda v: jnp.asarray(v)
    return LSState(
        filt_phi=jnp.zeros((cap,), dt), filt_theta=jnp.zeros((cap,), dt),
        filt_n=f(0), theta_min=f(-1.0).astype(dt), theta_max=f(-1.0).astype(dt),
        filter_reset_count=f(0), filter_reject_count=f(0),
        last_rejection_due_to_filter=f(False), in_watchdog=f(False),
        watchdog_shortened_iter_count=f(0), watchdog_trial_iter_count=f(0),
        watchdog_theta=f(0.0).astype(dt), watchdog_barr=f(0.0).astype(dt),
        watchdog_gbd=f(0.0).astype(dt), watchdog_alpha_test=f(0.0).astype(dt),
        tiny_step_last_iteration=f(False), in_soft_resto=f(False),
        soft_resto_counter=f(0), last_mu=f(-1.0).astype(dt),
        ref_theta=f(0.0).astype(dt), ref_barr=f(0.0).astype(dt),
        ref_gbd=f(0.0).astype(dt))

  @staticmethod
  def _reset_linesearch(ls: LSState):
    return ls._replace(filter_reject_count=jnp.asarray(0),
                       last_rejection_due_to_filter=jnp.asarray(False),
                       filt_n=jnp.asarray(0))

  def _augment(self, ls: LSState):
    o = self.opt
    i = jnp.minimum(ls.filt_n, self._filter_cap - 1)
    return ls._replace(
        filt_phi=ls.filt_phi.at[i].set(ls.ref_barr - o.gamma_phi * ls.ref_theta),
        filt_theta=ls.filt_theta.at[i].set((1.0 - o.gamma_theta) * ls.ref_theta),
        filt_n=jnp.minimum(ls.filt_n + 1, self._filter_cap))

  def _filter_ok(self, ls: LSState, barr, theta):
    act = jnp.arange(self._filter_cap) < ls.filt_n
    dom = act & (barr > ls.filt_phi) & (theta > ls.filt_theta)
    return ~jnp.any(dom)

  @staticmethod
  def _compare_le(a, b, base):
    eps = jnp.finfo(jnp.float64).eps
    return a - b <= 10.0 * eps * jnp.abs(base)

  def _is_f_type(self, ls: LSState, alpha):
    o = self.opt
    eps = jnp.finfo(jnp.float64).eps
    gbd = jnp.where((ls.ref_theta == 0.0) & (ls.ref_gbd > 0.0) &
                    (ls.ref_gbd < 100.0 * eps), -eps, ls.ref_gbd)
    ls = ls._replace(ref_gbd=gbd)
    f = (gbd < 0.0) & (alpha * jnp.power(jnp.maximum(-gbd, 0.0), o.s_phi) >
                       o.delta * jnp.power(ls.ref_theta, o.s_theta))
    return f, ls

  def _armijo(self, ls, alpha, trial_barr):
    return self._compare_le(trial_barr - ls.ref_barr,
                            self.opt.eta_phi * alpha * ls.ref_gbd, ls.ref_barr)

  def _acceptable_to_current(self, ls, barr, theta, from_resto):
    o = self.opt
    base = jnp.where(jnp.abs(ls.ref_barr) > 10.0,
                     jnp.log10(jnp.abs(ls.ref_barr)), 1.0)
    big = jnp.log10(jnp.maximum(barr - ls.ref_barr, 1e-300)) > o.obj_max_incr + base
    reject = (~jnp.asarray(from_resto)) & (barr > ls.ref_barr) & big
    ok = (self._compare_le(theta, (1.0 - o.gamma_theta) * ls.ref_theta,
                           ls.ref_theta) |
          self._compare_le(barr - ls.ref_barr, -o.gamma_phi * ls.ref_theta,
                           ls.ref_barr))
    return ok & ~reject

  def _check_acceptability(self, ls: LSState, theta, barr, alpha):
    """``check_acceptability_of_trial_point``; returns (accept, ls)."""
    o = self.opt
    ls = ls._replace(
        theta_max=jnp.where(ls.theta_max < 0.0,
                            o.theta_max_fact * jnp.maximum(1.0, ls.ref_theta),
                            ls.theta_max),
        theta_min=jnp.where(ls.theta_min < 0.0,
                            o.theta_min_fact * jnp.maximum(1.0, ls.ref_theta),
                            ls.theta_min))
    too_big = (ls.theta_max > 0) & (theta > ls.theta_max)
    fin = jnp.isfinite(barr)
    ftype, ls2 = self._is_f_type(ls, alpha)
    use_armijo = (alpha > 0.0) & ftype & (ls.ref_theta < ls.theta_min)
    ls = _where(alpha > 0.0, ls2, ls)
    acc_cur = jnp.where(use_armijo, self._armijo(ls, alpha, barr),
                        self._acceptable_to_current(ls, barr, theta, False))
    stage1 = (~too_big) & fin & acc_cur
    acc_filt = self._filter_ok(ls, barr, theta)
    # Filter rejection bookkeeping happens only if stage 1 passed.
    rej_filter = stage1 & ~acc_filt
    accept = stage1 & acc_filt
    allow = (o.max_filter_resets > 0) | (o.max_filter_resets == -1)
    can = allow & ((ls.filter_reset_count < o.max_filter_resets) |
                   (o.max_filter_resets == -1))
    cnt = jnp.where(ls.last_rejection_due_to_filter,
                    ls.filter_reject_count + 1, 0)
    do_reset = can & ls.last_rejection_due_to_filter & (
        cnt >= o.filter_reset_trigger)
    ls_acc = ls._replace(
        filter_reject_count=jnp.where(can, cnt, ls.filter_reject_count),
        filt_n=jnp.where(do_reset, 0, ls.filt_n),
        filter_reset_count=ls.filter_reset_count + do_reset.astype(jnp.int32),
        last_rejection_due_to_filter=jnp.asarray(False))
    ls_rej = ls._replace(last_rejection_due_to_filter=jnp.where(
        rej_filter, True, ls.last_rejection_due_to_filter))
    return accept, _where(accept, ls_acc, ls_rej)

  def _trial_merit(self, t: Iterate, vals: Trial, mode, refs):
    c = self._cviol(vals.c, t, mode)
    theta = _l1(c)
    obj = self._objective(t, vals.obj, mode, refs)
    barr = self._barrier(t, mode) + obj
    ok = _finite(c) & jnp.isfinite(obj) & jnp.isfinite(barr)
    return theta, barr, ok, c

  def _alpha_min(self, ls, it, lin, d, mode, refs):
    o = self.opt
    gbd = self._lin_dec(it, lin, d, mode, refs)
    th = _l1(self._cviol(lin.c, it, mode))
    am = jnp.asarray(o.gamma_theta, gbd.dtype)
    a1 = jnp.minimum(o.gamma_theta, o.gamma_phi * th / jnp.where(
        gbd < 0, -gbd, 1.0))
    a2 = jnp.minimum(a1, o.delta * jnp.power(th, o.s_theta) / jnp.power(
        jnp.where(gbd < 0, -gbd, 1.0), o.s_phi))
    a = jnp.where(th <= ls.theta_min, a2, a1)
    am = jnp.where((gbd < 0) & (th > 0), a, am)
    return o.alpha_min_frac * am

  def _line_search(self, S: State, A: AlgState, params, delta_w):
    """``IpLinesearch::find_acceptable_trial_point``.

    Returns (outcome, trial, cur, lin, dir, A, info) with outcome
    0 accepted, 1 soft-restoration step pending the pd-error test,
    2 failure (enter restoration / fail).
    """
    o = self.opt
    mode, refs = S.mode, S.refs
    cur, lin, d = S.cur, S.lin, S.dir
    ls = A.ls
    mu_changed = cur.mu != ls.last_mu
    ls = ls._replace(
        in_watchdog=ls.in_watchdog & ~mu_changed,
        watchdog_shortened_iter_count=jnp.where(
            mu_changed, 0, ls.watchdog_shortened_iter_count),
        last_mu=cur.mu)

    def merit_cur(it_, lin_, d_):
      th = _l1(self._cviol(lin_.c, it_, mode))
      barr = self._barrier(it_, mode) + self._objective(it_, lin_.obj, mode,
                                                        refs)
      return th, barr, self._lin_dec(it_, lin_, d_, mode, refs)

    th_c, barr_c, gbd_c = merit_cur(cur, lin, d)
    ls = ls._replace(
        ref_theta=jnp.where(ls.in_watchdog, ls.watchdog_theta, th_c),
        ref_barr=jnp.where(ls.in_watchdog, ls.watchdog_barr, barr_c),
        ref_gbd=jnp.where(ls.in_watchdog, ls.watchdog_gbd, gbd_c))
    # detect_tiny_step
    if o.tiny_step_tol == 0.0:
      tiny = jnp.asarray(False)
    else:
      mx = _linf(jnp.concatenate([
          (jnp.abs(d.dx) / (jnp.abs(cur.x) + 1.0)).ravel(),
          (jnp.abs(d.dw) / (jnp.abs(cur.w) + 1.0)).ravel(),
          (jnp.abs(d.ds) / (jnp.abs(cur.s) + 1.0)).ravel()]))
      mpn = sum(_linf(jnp.abs(dq) / (jnp.abs(q) + 1.0)) for q, dq in zip(
          jax.tree.leaves((cur.p, cur.n)), jax.tree.leaves((d.dp, d.dn))))
      mx = jnp.where(mode == 1, jnp.maximum(mx, mpn), mx)
      tiny = (mx <= o.tiny_step_tol) & (th_c <= 1e-4)
    # stop watchdog if a tiny step is detected while in watchdog.
    stopw = ls.in_watchdog & tiny

    def restore(args):
      cur_, lin_, d_, ls_ = args
      return (A.stored_it, A.stored_lin, A.stored_dir, ls_._replace(
          in_watchdog=jnp.asarray(False), ref_theta=ls_.watchdog_theta,
          ref_barr=ls_.watchdog_barr, ref_gbd=ls_.watchdog_gbd))

    cur, lin, d, ls = _scond(stopw, restore, lambda a: a,
                                   (cur, lin, d, ls))
    tiny = tiny & ~stopw
    ls = ls._replace(tiny_step_last_iteration=jnp.where(
        tiny, ls.tiny_step_last_iteration, False))
    start_w = ((o.watchdog_shortened_iter_trigger > 0) & ~ls.in_watchdog &
               ~tiny & ~ls.in_soft_resto &
               (ls.watchdog_shortened_iter_count >=
                o.watchdog_shortened_iter_trigger))
    apmax_c = self._max_step_primal(cur, d.ds, d.dp, d.dn, mode)
    ls = ls._replace(
        in_watchdog=ls.in_watchdog | start_w,
        watchdog_trial_iter_count=jnp.where(start_w, 0,
                                            ls.watchdog_trial_iter_count),
        watchdog_theta=jnp.where(start_w, ls.ref_theta, ls.watchdog_theta),
        watchdog_barr=jnp.where(start_w, ls.ref_barr, ls.watchdog_barr),
        watchdog_gbd=jnp.where(start_w, ls.ref_gbd, ls.watchdog_gbd),
        watchdog_alpha_test=jnp.where(start_w, apmax_c,
                                      ls.watchdog_alpha_test))
    A = A._replace(stored_it=_where(start_w, cur, A.stored_it),
                   stored_lin=_where(start_w, lin, A.stored_lin),
                   stored_dir=_where(start_w, d, A.stored_dir))
    soft_cnt = ls.soft_resto_counter + (ls.in_soft_resto & ~tiny).astype(
        jnp.int32)
    ls = ls._replace(soft_resto_counter=soft_cnt)
    soft_exceeded = ls.in_soft_resto & ~tiny & (soft_cnt > o.max_soft_resto_iters)
    # Phases: 0 tiny, 1 soft (in-phase), 2 backtracking, 3 soc,
    # 4 soft (after failure), 5 done.
    phase0 = jnp.where(tiny, 0, jnp.where(ls.in_soft_resto,
                                         jnp.where(soft_exceeded, 5, 1), 2))
    dt = cur.x.dtype
    alpha_min0 = jnp.where(ls.in_watchdog, apmax_c,
                           self._alpha_min(ls, cur, lin, d, mode, refs))
    atest0 = jnp.where(ls.in_watchdog, ls.watchdog_alpha_test, apmax_c)
    th_cur_l1 = th_c
    z = lambda: jnp.asarray(0.0, dt)
    # Loop-invariant data lives outside the while-loop carry (a trace-time
    # context): copying it through every conditional dominated GPU time.
    self._ctx = dict(lin=lin, A=A, refs=refs, mode=mode, cur=cur, d=d,
                     fac=getattr(self, '_search_fac', None))
    # cur/d/trial are not carried: they change only on a watchdog restart
    # (flag ``restored``) or an accepted SOC (``soc_taken``), and the trial
    # point is recomputed from its step length.
    carry = dict(
        phase=phase0, ls=ls, restored=jnp.asarray(False),
        tiny_flag=A.tiny_step_flag,
        alpha=apmax_c, apmax=apmax_c, alpha_min=alpha_min0, atest=atest0,
        n_steps=jnp.asarray(0), accept=jnp.asarray(False),
        eval_error=jnp.asarray(False), soc_taken=jnp.asarray(False),
        soc_count=jnp.asarray(0), theta_soc_old=z(), theta_trial=z(),
        alpha_soc=z(), soc_acc=self._cviol(lin.c, cur, mode), dsoc=d,
        trial_alpha=z(), trial_barr=z(), alpha_outer=z(),
        typ=jnp.asarray(T_UNKNOWN),
        tiny=tiny, outcome=jnp.asarray(2), satisfies=jnp.asarray(False),
        soft_kind=jnp.asarray(0), passes=jnp.asarray(0), th_cur=th_cur_l1,
        nkkt=jnp.asarray(0), nevals=jnp.asarray(0),
        cache_a=jnp.zeros((self._K,), dt), cache_th=jnp.zeros((self._K,), dt),
        cache_barr=jnp.zeros((self._K,), dt),
        cache_ok=jnp.zeros((self._K,), bool), cache_n=jnp.asarray(0))

    def cond(c):
      return (c['phase'] != 5) & (c['passes'] < o.max_ls)

    def candidate(c):
      """The primal point evaluated in this pass (one model evaluation)."""
      ph = c['phase']
      cur_, d_ = self._eff_cur(c), self._eff_d(c)
      admax = self._max_step_dual(cur_, d_, mode)
      a_soft = jnp.minimum(c['apmax'], admax)
      a = jnp.where(ph == 0, c['apmax'],
                    jnp.where((ph == 1) | (ph == 4), a_soft, c['alpha']))
      dsel = _where(ph == 3, c['dsoc'], d_)
      a = jnp.where(ph == 3, c['alpha_soc'], a)
      return self._set_primal(cur_, dsel, a, mode), a, dsel

    def soc_solve(c):
      """Second-order correction direction (``try_second_order_correction``)."""
      cur_, lin_ = self._ctx['cur'], self._ctx['lin']
      acc = c['soc_acc']
      sys, red = self._assemble(cur_, lin_, mode, refs, delta_w, False)
      b = self._full_rhs(cur_, lin_, mode, refs, g=acc)
      dsoc, flag, nsoc = self._solve_full(sys, red, b, factor=False,
                                          fac=self._ctx.get('fac'))
      asoc = self._max_step_primal(cur_, dsoc.ds, dsoc.dp, dsoc.dn, mode)
      ok = flag == 0
      return dict(c, dsoc=dsoc, alpha_soc=asoc, nkkt=c['nkkt'] + nsoc,
                  phase=jnp.where(ok, 3, 2),
                  # A failed SOC solve breaks the SOC loop: continue
                  # backtracking.
                  soc_count=jnp.where(ok, c['soc_count'], o.max_soc))

    def body(c):
      c = dict(c, passes=c['passes'] + 1)
      need_soc = c['phase'] == 3
      c = _scond(need_soc, soc_solve, lambda cc: cc, c)
      # A failed SOC solve sets phase back to 2 (backtracking) and is
      # handled below without evaluating a point.
      tpt, a, dsel = candidate(c)
      ph = c['phase']
      skip_eval = need_soc & (ph == 2)
      K = self._K
      is_bt = ph == 2
      match = (c['cache_a'] == a) & (jnp.arange(K) < c['cache_n'])
      hit = is_bt & jnp.any(match)

      def evaluate(_):
        # One (batched) model evaluation.  Backtracking trials evaluate the
        # next K step lengths of the sequence; other phases one point.
        def nxt(prev, _):
          return prev * o.alpha_red_factor, prev
        _, alphas = jax.lax.scan(nxt, a, None, length=K)
        alphas = jnp.where(is_bt, alphas, a)
        pts = jax.vmap(lambda al: self._set_primal(self._eff_cur(c), dsel, al,
                                                   mode))(alphas)
        vals = jax.vmap(lambda t: self._values(t.x, t.w, t.s, params))(pts)
        th, br, okv, cv = jax.vmap(
            lambda t, v: self._trial_merit(t, v, mode, refs))(pts, vals)
        return th, br, okv, _tmap(lambda l: l[0], cv), alphas

      def lookup(_):
        j = jnp.argmax(match)
        zero_c = _tmap(jnp.zeros_like, self._ctx['lin'].c)
        sel = lambda v: jnp.full((K,), v[j])
        return (sel(c['cache_th']), sel(c['cache_barr']), sel(c['cache_ok']),
                zero_c, jnp.full((K,), a))

      th_v, br_v, ok_v, cvec, alphas = jax.lax.cond(
          hit | skip_eval, lookup, evaluate, None)
      theta, barr, ok = th_v[0], br_v[0], ok_v[0]
      fill = is_bt & ~hit & ~skip_eval
      c = dict(c, nevals=c['nevals'] + (~hit & ~skip_eval).astype(jnp.int64),
               cache_a=jnp.where(fill, alphas, c['cache_a']),
               cache_th=jnp.where(fill, th_v, c['cache_th']),
               cache_barr=jnp.where(fill, br_v, c['cache_barr']),
               cache_ok=jnp.where(fill, ok_v, c['cache_ok']),
               cache_n=jnp.where(fill, K, c['cache_n']))
      branches = [functools.partial(f, a=a, tpt=tpt, theta=theta, barr=barr,
                                    ok=ok, cvec=cvec)
                  for f in (self._ph_tiny, self._ph_soft, self._ph_bt,
                            self._ph_soc, self._ph_soft)]
      idx = jnp.clip(ph, 0, 4)
      c_new = _sswitch(idx, branches, c)
      # Backtracking after an SOC-solve failure: undo the evaluation.
      return _where(skip_eval, self._bt_advance(c), c_new)

    c = _swhile(cond, body, carry)
    return c

  # --- line-search phases ----------------------------------------------------
  def _eff_cur(self, c):
    return _where(c['restored'], self._ctx['A'].stored_it, self._ctx['cur'])

  def _eff_d(self, c):
    d = _where(c['restored'], self._ctx['A'].stored_dir, self._ctx['d'])
    return _where(c['soc_taken'], c['dsoc'], d)
  def _bt_advance(self, c):
    """``alpha_primal *= alpha_red_factor; n_steps++`` and loop test."""
    o = self.opt
    alpha = c['alpha'] * o.alpha_red_factor
    n = c['n_steps'] + 1
    more = alpha > c['alpha_min']
    c = dict(c, alpha=alpha, n_steps=n)
    return _scond(more, lambda cc: dict(cc, phase=jnp.asarray(2)),
                        self._bt_pass_end, c)

  def _bt_pass_end(self, c):
    """End of ``do_backtracking_line_search`` and the watchdog loop."""
    o = self.opt
    ls = c['ls']
    accept = c['accept']
    in_w = ls.in_watchdog
    # char / filter update for an accepted trial point
    ftype, ls2 = self._is_f_type(ls, c['apmax'])
    arm = self._armijo(ls2, c['apmax'], c['trial_barr'])
    is_f = ftype & arm
    ls_upd = _scond(accept & ~is_f, self._augment, lambda l: l, ls2)
    ls = _where(accept, ls_upd, ls)
    typ = jnp.where(accept, jnp.where(is_f, T_F, T_H),
                    jnp.where(in_w, T_W, T_UNKNOWN))
    typ = jnp.where(c['soc_taken'] & (typ == T_F), T_FS, typ)
    typ = jnp.where(c['soc_taken'] & (typ == T_H), T_HS, typ)
    c = dict(c, ls=ls, typ=typ, alpha_outer=c['alpha'])
    # watchdog handling in find_acceptable_trial_point
    w_acc = in_w & accept
    w_rej = in_w & ~accept
    cnt = ls.watchdog_trial_iter_count + w_rej.astype(jnp.int32)
    stop = w_rej & (c['eval_error'] | (cnt >= o.watchdog_trial_iter_max))
    prov = w_rej & ~stop
    ls = ls._replace(watchdog_trial_iter_count=cnt,
                     in_watchdog=in_w & ~w_acc & ~stop)
    c = dict(c, ls=ls, accept=accept | prov)

    def restart(cc):
      ctx = self._ctx
      A = ctx['A']
      l = cc['ls']._replace(ref_theta=cc['ls'].watchdog_theta,
                            ref_barr=cc['ls'].watchdog_barr,
                            ref_gbd=cc['ls'].watchdog_gbd)
      cur_, lin_, d_ = A.stored_it, A.stored_lin, A.stored_dir
      apmax = self._max_step_primal(cur_, d_.ds, d_.dp, d_.dn, ctx['mode'])
      amin = self._alpha_min(l, cur_, lin_, d_, ctx['mode'], ctx['refs'])
      return dict(cc, ls=l, apmax=apmax,
                  restored=jnp.asarray(True), cache_n=jnp.asarray(0),
                  alpha=apmax * o.alpha_red_factor, alpha_min=amin,
                  atest=apmax, soc_taken=jnp.asarray(False),
                  accept=jnp.asarray(False), eval_error=jnp.asarray(False),
                  phase=jnp.asarray(2))

    def finish(cc):
      return self._ls_after(cc)

    # FATROP sets skip_first_trial_point after stop_watchdog(), but the
    # ``if (!in_watchdog_) done = true`` that follows ends the loop, so no
    # backtracking retry happens: the restored iterate goes straight to the
    # rejected-step branch (filter augmentation + soft restoration).
    return _scond(stop, lambda cc: self._ls_after(restart(cc)), finish, c)

  def _ls_after(self, c):
    """Post backtracking: soft restoration attempt or done."""
    accept = c['accept']
    ls = c['ls']
    # !accept & !in_soft_resto -> prepare_resto_phase_start + soft step.
    go_soft = ~accept & ~ls.in_soft_resto
    ls = _scond(go_soft, self._augment, lambda l: l, ls)
    return dict(c, ls=ls, phase=jnp.where(go_soft, 4, 5),
                soft_kind=jnp.where(go_soft, 1, c['soft_kind']),
                outcome=jnp.where(accept, 0, 2))

  def _ph_tiny(self, c, *, a, tpt, theta, barr, ok, cvec):
    o = self.opt
    ls = c['ls']
    flag = ok & ls.tiny_step_last_iteration
    tflag = c['tiny_flag'] | flag
    dy = _linf(self._eff_d(c).dy)
    ls = ls._replace(tiny_step_last_iteration=dy >= o.tiny_step_y_tol)
    # A non-finite tiny trial falls back to the ordinary line search.
    cnt = ls.soft_resto_counter + (ls.in_soft_resto & ~ok).astype(jnp.int32)
    ls = ls._replace(soft_resto_counter=cnt)
    nxt = jnp.where(ls.in_soft_resto,
                    jnp.where(cnt > o.max_soft_resto_iters, 5, 1), 2)
    return dict(c, ls=ls, tiny_flag=tflag, trial_alpha=a, trial_barr=barr,
                tiny=ok,
                accept=ok, outcome=jnp.where(ok, 0, 2),
                alpha_outer=jnp.where(ok, 0.0, c['alpha_outer']),
                typ=jnp.where(ok, T_TINY, c['typ']),
                phase=jnp.where(ok, 5, nxt))

  def _ph_soft(self, c, *, a, tpt, theta, barr, ok, cvec):
    """``try_soft_resto_step``: the pd-error test needs the trial Jacobian
    and is completed by the outer loop (outcome 1)."""
    acc, ls = self._check_acceptability(c['ls'], theta, barr, 0.0)
    kind = jnp.where(c['phase'] == 1, 0, 1)
    return dict(c, ls=ls, trial_alpha=a, trial_barr=barr, satisfies=acc,
                soft_kind=kind, outcome=jnp.where(acc, 0, 1), alpha_outer=a,
                phase=jnp.asarray(5), typ=jnp.where(acc, T_SOFT_ORIG, T_SOFT))

  def _ph_bt(self, c, *, a, tpt, theta, barr, ok, cvec):
    o = self.opt
    acc, ls = self._check_acceptability(c['ls'], theta, barr, c['atest'])
    acc = acc & ok
    ls = _where(ok, ls, c['ls'])
    c = dict(c, ls=ls, accept=acc, eval_error=c['eval_error'] | ~ok,
             trial_alpha=a, trial_barr=barr)
    in_w = ls.in_watchdog
    first = c['alpha'] == c['apmax']
    try_soc = (~acc & ~in_w & ok & first & (c['th_cur'] <= theta) &
               (o.max_soc > 0))

    def to_soc(cc):
      acc0 = _tmap(lambda g0, gt: g0 + cc['alpha'] * gt,
                   self._cviol(self._ctx['lin'].c, self._ctx['cur'],
                               self._ctx['mode']), cvec)
      return dict(cc, phase=jnp.asarray(3), soc_count=jnp.asarray(0),
                  theta_soc_old=theta, soc_acc=acc0)

    def done_or_next(cc):
      stop = cc['accept'] | in_w
      return _scond(stop, self._bt_pass_end, self._bt_advance, cc)

    return _scond(try_soc, to_soc, done_or_next, c)

  def _ph_soc(self, c, *, a, tpt, theta, barr, ok, cvec):
    o = self.opt
    acc, ls = self._check_acceptability(c['ls'], theta, barr, c['atest'])
    c = dict(c, ls=ls)

    def accepted(cc):
      return self._bt_pass_end(dict(
          cc, accept=jnp.asarray(True), soc_taken=jnp.asarray(True),
          alpha=cc['alpha_soc'], trial_alpha=cc['alpha_soc'],
          trial_barr=barr))

    def rejected(cc):
      cnt = cc['soc_count'] + 1
      more = (cnt < o.max_soc) & (theta < o.kappa_soc * cc['theta_soc_old'])
      acc2 = _tmap(lambda g0, gt: g0 + cc['alpha_soc'] * gt, cc['soc_acc'],
                   cvec)
      nxt = dict(cc, soc_count=cnt, theta_soc_old=theta, soc_acc=acc2)
      return _scond(more, lambda q: dict(q, phase=jnp.asarray(3)),
                          self._bt_advance, nxt)

    return _scond(acc, accepted, rejected, c)

  # -------------------------------------------------------------------------
  # One IpAlgorithm iteration
  # -------------------------------------------------------------------------
  def _record(self, S: State, params):
    if not self.opt.record_trace:
      return S
    it, lin, mode = S.cur, S.lin, S.mode
    fx, fw, fs, fp, fn = self._dual_inf(it, lin, mode, S.refs)
    inf_du = _linf((fx, fw, fs))
    inf_pr = _linf(lin.c)  # original constraint violation (resto prints it)
    row = jnp.stack([
        S.it.astype(jnp.float64), S.mode.astype(jnp.float64), lin.obj, inf_pr,
        inf_du, jnp.log10(it.mu), S.info.d_norm, S.info.rg, S.info.alpha_du,
        S.info.alpha_pr, S.info.typ.astype(jnp.float64),
        S.info.ls.astype(jnp.float64)])
    i = jnp.minimum(S.trace_n, self._trace_cap - 1)
    return S._replace(trace=S.trace.at[i].set(row),
                      trace_n=jnp.minimum(S.trace_n + 1, self._trace_cap))

  def _active(self, S: State) -> AlgState:
    return _where(S.mode == 1, S.resto, S.orig)

  def _store_active(self, S: State, A: AlgState) -> State:
    r = S.mode == 1
    return S._replace(orig=_where(r, S.orig, A), resto=_where(r, A, S.resto))

  def _mu_update(self, S: State, A: AlgState):
    """``IpMonotoneMuUpdate::update_barrier_parameter``."""
    o = self.opt
    cur = S.cur
    floor = max(min(o.tol, o.compl_inf_tol) / (o.barrier_tol_factor + 1.0),
                o.mu_min)
    allowed = (jnp.asarray(o.mu_update_max_iter < 0) |
               (S.it < o.mu_update_max_iter) | (S.mode == 1))

    def err(mu):
      return self._e_mu(cur, S.lin, mu, S.mode, S.refs)[0]

    def cond(c):
      mu, e, done, tiny, changed_any, last_changed = c
      return ((e < o.barrier_tol_factor * mu) | tiny) & ~done & allowed

    def body(c):
      mu, e, done, tiny, changed_any, last_changed = c
      new = jnp.minimum(o.mu_linear_decrease_factor * mu,
                        jnp.power(mu, o.mu_superlinear_decrease_power))
      new = jnp.maximum(new, floor)
      changed = new != mu
      stop_fast = A.mu_initialized & (not o.mu_allow_fast_monotone_decrease)
      e_new = err(new)
      done_new = jnp.where(stop_fast | ~changed, True,
                           e_new > o.barrier_tol_factor * new)
      e_out = jnp.where(stop_fast | ~changed, e, e_new)
      return (new, e_out, done_new, jnp.asarray(False),
              changed_any | (done_new & changed), changed)

    mu0 = cur.mu
    init = (mu0, err(mu0), jnp.asarray(False), A.tiny_step_flag,
            jnp.asarray(False), jnp.asarray(False))
    mu, _, _, _, reset, _ = _swhile(cond, body, init)
    ls = _scond(reset, self._reset_linesearch, lambda l: l, A.ls)
    A = A._replace(ls=ls, mu_initialized=jnp.asarray(True),
                   tiny_step_flag=jnp.asarray(False))
    return S._replace(cur=cur._replace(mu=mu)), A

  def _search_dir(self, S: State, A: AlgState):
    """``IpSearchDirImpl::compute_search_dir`` (inertia correction)."""
    o = self.opt
    cur, lin, mode, refs = S.cur, S.lin, S.mode, S.refs
    b = self._full_rhs(cur, lin, mode, refs)

    def solve(dw):
      sys, red = self._assemble(cur, lin, mode, refs, dw, False)
      return self._solve_full(sys, red, b, factor=True, return_fac=True)

    def cond(c):
      dw, d, flag, tries, fail, ns, fac = c
      return (tries == 0) | ((flag == 1) & ~fail &
                             (tries < o.max_inertia_tries))

    def body(c):
      dw, d, flag, tries, fail, ns, fac = c
      last = A.delta_w_last
      first = tries == 1
      dw_new = jnp.where(
          tries == 0, 0.0,
          jnp.where(first, jnp.where(last == 0.0, o.delta_w0,
                                     jnp.maximum(last * o.kappa_wmin,
                                                 o.delta_wmin)),
                    jnp.where(last == 0.0, o.kappa_wplusem * dw,
                              o.kappa_wplus * dw)))
      fail_new = ~(dw_new <= o.delta_w_max)
      d_new, flag_new, n_new, fac_new = solve(jnp.where(fail_new, dw, dw_new))
      return (jnp.where(fail_new, dw, dw_new), d_new,
              jnp.where(fail_new, flag, flag_new), tries + 1, fail_new,
              ns + n_new, fac_new)

    fac0 = jax.eval_shape(lambda: solve(jnp.asarray(0.0, cur.x.dtype))[3])
    fac0 = _tmap(lambda l: jnp.zeros(l.shape, l.dtype), fac0)
    dw, d, flag, _, fail, ns, fac = _swhile(
        cond, body, (jnp.asarray(0.0, cur.x.dtype), self._zero_dir(b),
                     jnp.asarray(1), jnp.asarray(0), jnp.asarray(False),
                     jnp.asarray(0), fac0))
    ok = flag == 0
    A = A._replace(delta_w_last=jnp.where(ok & (dw > 0), dw, A.delta_w_last))
    self._search_fac = fac if self.linear_backend == 'pas' else None
    return S._replace(dir=d, stats=S.stats.at[0].add(ns)), A, dw, ok

  def _iteration(self, S: State, params):
    """Body of the ``while (conv_status == Continue)`` loop of
    ``IpAlgorithm::optimize`` up to (and excluding) ``accept_trial_iterate``."""
    S = self._record(S, params)
    A = self._active(S)
    S, A = self._mu_update(S, A)
    S, A, delta_w, sd_ok = self._search_dir(S, A)

    def do_ls(args):
      S_, A_ = args
      c = self._line_search(S_, A_, params, delta_w)
      return self._ls_finish(S_, c, delta_w)

    def sd_fail(args):
      S_, A_ = args
      S_ = self._store_active(S_, A_)
      # Error in the step computation: the original problem terminates;
      # inside restoration this makes the restoration (and thus the outer
      # line search) fail.
      return S_._replace(status=jnp.where(S_.mode == 1, LINESEARCH_FAILED,
                                          STEP_ERROR))

    S = self._store_active(S, A)
    return _scond(sd_ok, lambda S_: do_ls((S_, self._active(S_))),
                  lambda S_: sd_fail((S_, self._active(S_))), S)

  def _ls_finish(self, S: State, c, delta_w):
    o = self.opt
    mode = S.mode
    ctx = self._ctx
    A = ctx['A']._replace(ls=c['ls'], tiny_step_flag=c['tiny_flag'])
    cur, d = self._eff_cur(c), self._eff_d(c)
    mode_ = ctx['mode']
    tprim = self._set_primal(cur, d, c['trial_alpha'], mode_)
    is_soft = (c['typ'] == T_SOFT) | (c['typ'] == T_SOFT_ORIG)
    c_trial = _where(is_soft, self._dual_step(cur, d, tprim, c['trial_alpha'],
                                              c['trial_alpha'], mode_), tprim)
    lin = _where(c['restored'], A.stored_lin, ctx['lin'])
    outcome = c['outcome']
    accept = outcome == 0
    tiny = c['tiny']
    ls = A.ls
    # perform_dual_step for ordinary/tiny/watchdog acceptance (not for a
    # soft restoration step taken while staying in the soft phase).
    ordinary = accept & (~ls.in_soft_resto | tiny) & ~(
        (c['soft_kind'] == 1) & c['satisfies'] & (c['phase'] == 5) &
        (c['typ'] == T_SOFT_ORIG))
    soft_inphase_sat = accept & c['satisfies'] & (c['soft_kind'] == 0) & (
        c['typ'] == T_SOFT_ORIG)
    # In-phase soft step satisfying the original criterion leaves the soft
    # phase and then also receives the (alpha_primal = 0) dual step.
    ls = ls._replace(
        in_soft_resto=jnp.where(soft_inphase_sat, False, ls.in_soft_resto),
        soft_resto_counter=jnp.where(soft_inphase_sat, 0,
                                     ls.soft_resto_counter))
    ordinary = ordinary | soft_inphase_sat
    admax = self._max_step_dual(cur, d, mode)
    a_pr = jnp.where(soft_inphase_sat | tiny, 0.0, c['alpha_outer'])
    trial_ord = self._dual_step(cur, d, c_trial, a_pr, admax, mode)
    trial = _where(ordinary, trial_ord, c_trial)
    ls = ls._replace(watchdog_shortened_iter_count=jnp.where(
        ordinary, jnp.where(c['n_steps'] == 0, 0,
                            ls.watchdog_shortened_iter_count + 1),
        ls.watchdog_shortened_iter_count))
    A = A._replace(ls=ls)
    d_norm = jnp.maximum(_linf((d.dx, d.dw)), _linf(d.ds))
    info = StepInfo(alpha_pr=c['alpha_outer'], alpha_du=admax,
                    ls=c['n_steps'] + 1, typ=c['typ'], d_norm=d_norm,
                    rg=jnp.asarray(delta_w, d_norm.dtype))
    S = S._replace(cur=cur, lin=lin, dir=d, stats=S.stats.at[0].add(
        c['nkkt']).at[1].add(c['nevals']))
    S = self._store_active(S, A)

    def accepted(S_):
      return S_._replace(prev=S_.cur, prev_lin=S_.lin, prev_dir=S_.dir,
                         cur=trial, info=info, pending=jnp.asarray(P_CHECK),
                         need_lin=jnp.asarray(True))

    def soft_pending(S_):
      return S_._replace(prev=S_.cur, prev_lin=S_.lin, prev_dir=S_.dir,
                         cur=trial, info=info, pending=jnp.asarray(P_SOFT),
                         need_lin=jnp.asarray(True),
                         soft_kind=c['soft_kind'])

    def failed(S_):
      return self._enter_resto_or_fail(S_)

    return _sswitch(jnp.clip(outcome, 0, 2),
                          [accepted, soft_pending, failed], S)

  # -------------------------------------------------------------------------
  # Soft restoration completion, restoration entry / exit
  # -------------------------------------------------------------------------
  def _finish_soft(self, S: State):
    o = self.opt
    mode, refs = S.mode, S.refs
    e_trial = self._pd_error(S.cur, S.lin, mode, refs)
    e_prev = self._pd_error(S.prev, S.prev_lin, mode, refs)
    ok = e_trial <= o.soft_rest_pd_error_reduction_factor * e_prev
    A = self._active(S)

    def acc(S_):
      ls = A.ls._replace(in_soft_resto=jnp.where(
          S_.soft_kind == 1, True, A.ls.in_soft_resto))
      S_ = self._store_active(S_, A._replace(ls=ls))
      return S_._replace(pending=jnp.asarray(P_CHECK))

    def rej(S_):
      trial_y = S_.cur.y
      S_ = S_._replace(cur=S_.prev, lin=S_.prev_lin, dir=S_.prev_dir,
                       pending=jnp.asarray(P_NONE))
      return self._enter_resto_or_fail(S_, trial_y)

    return _scond(ok, acc, rej, S)

  @staticmethod
  def _resto_fail(S: State):
    # The restoration phase failed and so does the original line search.
    # FATROP copies the restoration primal point into the original iterate.
    o_it = S.refs.orig_it._replace(x=S.cur.x, w=S.cur.w, s=S.cur.s)
    return S._replace(status=jnp.asarray(LINESEARCH_FAILED),
                      pending=jnp.asarray(P_NONE), cur=o_it,
                      mode=jnp.asarray(0))

  def _enter_resto_or_fail(self, S: State, trial_y=None):
    trial_y = S.cur.y if trial_y is None else trial_y
    return _scond(S.mode == 1, self._resto_fail,
                  lambda S_: self._enter_resto(S_, trial_y), S)

  def _enter_resto(self, S: State, trial_y=None):
    """``IpRestoPhaseMinCl1::perform_restoration`` set-up and
    ``IpInitializerResto::initialize``."""
    o = self.opt
    cur, lin = S.cur, S.lin
    rho = o.resto_rho
    c = lin.c
    mu_r = jnp.maximum(cur.mu, _linf(c))

    def pn(cv):
      p = (mu_r - rho * cv) / (2 * rho)
      p = p + jnp.sqrt(p * p + mu_r * cv / (2 * rho))
      return p, cv + p

    pe = ERows(*(pn(v)[0] for v in (c.c0, c.ch, c.cg)))
    ne = ERows(*(pn(v)[1] for v in (c.c0, c.ch, c.cg)))
    zp = _tmap(lambda q: mu_r / q, pe)
    zn = _tmap(lambda q: mu_r / q, ne)
    lb, ub, flo, fhi = self._bounds()
    r_it = cur._replace(
        y=_tmap(jnp.zeros_like, cur.y),
        zL=jnp.where(flo, jnp.minimum(cur.zL, rho), 0.0),
        zU=jnp.where(fhi, jnp.minimum(cur.zU, rho), 0.0),
        p=pe, n=ne, zp=zp, zn=zn, mu=mu_r)
    refs = Refs(xref=cur.x, wref=cur.w, sref=cur.s, zeta=jnp.sqrt(cur.mu),
                orig_it=cur, orig_lin=lin, orig_dir=S.dir,
                orig_inf_pr=_linf(c),
                trial_y=cur.y if trial_y is None else trial_y)
    # Original line search bookkeeping after perform_restoration().
    ols = S.orig.ls._replace(in_soft_resto=jnp.asarray(False),
                             soft_resto_counter=jnp.asarray(0),
                             watchdog_shortened_iter_count=jnp.asarray(0))
    orig = S.orig._replace(ls=ols)
    resto = AlgState(ls=self._ls_reset(cur.x.dtype),
                     delta_w_last=jnp.asarray(0.0, cur.x.dtype),
                     mu_initialized=jnp.asarray(False),
                     tiny_step_flag=jnp.asarray(False),
                     acceptable_counter=jnp.asarray(0),
                     stored_it=r_it, stored_lin=lin, stored_dir=S.dir)
    if self.hessian_mode == 'exact_lagrangian':
      lin = lin._replace(Cq=jnp.zeros_like(lin.Cq), Cs=jnp.zeros_like(lin.Cs),
                         Cr=jnp.zeros_like(lin.Cr))
    S = S._replace(mode=jnp.asarray(1), cur=r_it, lin=lin, refs=refs,
                   orig=orig, resto=resto, it=S.it + 1,
                   first_resto_iter=jnp.asarray(True),
                   n_resto=S.n_resto + 1, pending=jnp.asarray(P_NONE),
                   info=S.info._replace(typ=jnp.asarray(T_X)))
    # Initial convergence check of the restoration algorithm.
    return self._check_resto(S, initial=True)

  def _init_phase(self, S: State):
    """``IpInitializer::initialize`` + the initial convergence check."""
    lb, ub, flo, fhi = self._bounds()
    it, lin = S.cur, S.lin
    s = self._init_slacks(lin.graw)
    one = jnp.ones_like(s)
    it = it._replace(s=s, zL=jnp.where(flo, one, 0.0),
                     zU=jnp.where(fhi, one, 0.0))
    lin = lin._replace(c=lin.c._replace(cg=lin.graw - s))
    refs = S.refs._replace(sref=s, orig_it=it, orig_lin=lin)
    it = it._replace(y=self._eq_mult_init(it, lin, refs))
    S = S._replace(cur=it, lin=lin, refs=refs, prev=it, prev_lin=lin,
                   orig=S.orig._replace(stored_it=it, stored_lin=lin),
                   resto=S.resto._replace(stored_it=it, stored_lin=lin),
                   need_lin=jnp.asarray(self.hessian_mode ==
                                        'exact_lagrangian'))
    S = self._check_orig(S)
    # FATROP checks before the loop and counts the iteration after a step.
    return S._replace(it=S.it - 1)

  def _check_orig(self, S: State):
    """``IpConvergenceCheck::check_converged`` (+ iteration increment)."""
    o = self.opt
    A = S.orig
    e0, cv, _, _ = self._e_mu(S.cur, S.lin, 0.0, 0, S.refs)
    conv = (e0 <= o.tol) & (cv <= o.constr_viol_tol)
    accp = (e0 <= o.acceptable_tol) & (cv <= o.constr_viol_tol)
    cnt = jnp.where(accp, A.acceptable_counter + 1, 0)
    status = jnp.where(conv, SUCCESS,
                       jnp.where(accp & (cnt >= o.acceptable_iter), ACCEPTABLE,
                                 jnp.where(S.it >= S.max_iter, MAX_ITER,
                                           CONTINUE)))
    return S._replace(orig=A._replace(acceptable_counter=jnp.where(
        conv, A.acceptable_counter, cnt)), status=status,
                      it=S.it + 1, pending=jnp.asarray(P_NONE))

  def _check_resto(self, S: State, initial=False):
    """``IpConvergenceCheckResto::check_converged``."""
    o = self.opt
    refs = S.refs
    r_it = S.cur
    # original problem at the restoration point
    oc = S.lin.c
    o_inf = _linf(oc)
    o_theta = _l1(oc)
    o_trial = refs.orig_it._replace(x=r_it.x, w=r_it.w, s=r_it.s)
    o_barr = S.lin.obj + self._barrier(o_trial, 0)
    max_inf = jnp.maximum(o.required_infeasibility_reduction * refs.orig_inf_pr,
                          min(o.tol, o.constr_viol_tol))
    ols = S.orig.ls
    ok_filter = self._filter_ok(ols, o_barr, o_theta)
    ok_cur = self._acceptable_to_current(ols, o_barr, o_theta, True)
    converged = (~S.first_resto_iter) & (o_inf <= max_inf) & ok_filter & ok_cur
    # Base check on the restoration problem.
    A = S.resto
    e0, cv, _, _ = self._e_mu(r_it, S.lin, 0.0, 1, refs)
    b_conv = (e0 <= o.tol) & (cv <= o.constr_viol_tol)
    b_acc = (e0 <= o.acceptable_tol) & (cv <= o.constr_viol_tol)
    cnt = jnp.where(b_acc, A.acceptable_counter + 1, 0)
    b_stop = b_conv | (b_acc & (cnt >= o.acceptable_iter)) | (S.it >= S.max_iter)
    A = A._replace(acceptable_counter=jnp.where(~converged & ~b_conv, cnt,
                                                A.acceptable_counter))
    S = S._replace(resto=A, first_resto_iter=jnp.asarray(False))
    resto_fail = ~converged & b_stop
    if not initial:
      S = S._replace(it=S.it + 1)

    def fail(S_):
      return self._resto_fail(self._record(S_, None))

    def cont(S_):
      return S_._replace(pending=jnp.asarray(P_NONE))

    return _scond(
        converged, lambda S_: self._resto_handoff(self._record(S_, None)),
        lambda S_: _scond(resto_fail, fail, cont, S_), S)

  def _resto_handoff(self, S: State):
    """Successful ``perform_restoration``: build the original trial iterate."""
    o = self.opt
    refs = S.refs
    oi = refs.orig_it
    od = refs.orig_dir
    r_it = S.cur
    lb, ub, flo, fhi = self._bounds()
    Sl, Su = self._deltas(oi.s)
    ds = r_it.s - oi.s
    cl, cu, _, _ = self._compl(oi, oi.mu, 0)
    # FATROP uses the *direction* of the bound multipliers of the original
    # iterate here (``curr_it_orig.delta_dual_bounds_l()``).
    dzL = jnp.where(flo, (-cl - od.dzL * ds) / Sl, 0.0)
    dzU = jnp.where(fhi, (-cu + od.dzU * ds) / Su, 0.0)
    # maximum_step_size_dual is called with the (positive) trial multipliers
    # as "directions" and is therefore identically one.
    zL = jnp.where(flo, oi.zL + dzL, 0.0)
    zU = jnp.where(fhi, oi.zU + dzU, 0.0)
    big = jnp.maximum(_linf(zL), _linf(zU)) > o.bound_mult_reset_threshold
    zL = jnp.where(big, jnp.where(flo, 1.0, 0.0), zL)
    zU = jnp.where(big, jnp.where(fhi, 1.0, 0.0), zU)
    t = oi._replace(x=r_it.x, w=r_it.w, s=r_it.s, zL=zL, zU=zU,
                    p=_tmap(jnp.ones_like, oi.p), n=_tmap(jnp.ones_like, oi.n),
                    zp=_tmap(jnp.zeros_like, oi.zp),
                    zn=_tmap(jnp.zeros_like, oi.zn))
    t = self._modify_dual_bounds(t, oi.mu, 0)
    if o.resto_handoff_ls_multipliers:
      y = self._eq_mult_init(t, S.lin, refs)
    else:
      y = refs.trial_y
    t = t._replace(y=y)
    it = S.it - 1
    S = S._replace(mode=jnp.asarray(0), cur=t, it=it,
                   need_lin=jnp.asarray(self.hessian_mode == 'exact_lagrangian'),
                   info=S.info._replace(typ=jnp.asarray(T_X)))
    return self._check_orig(S)

  # -------------------------------------------------------------------------
  # Main loop
  # -------------------------------------------------------------------------
  def _solve_impl(self, params, x_init, w_init, max_iter):
    o = self.opt
    N, nx, nw, ne, ni = self.N, self.nx, self.nw, self.ne, self.ni
    dt = jnp.float64
    x = jnp.asarray(x_init, dt)
    w = jnp.asarray(w_init, dt)
    zero_rows = Rows(jnp.zeros((nx,), dt), jnp.zeros((N, nx), dt),
                     jnp.zeros((N, ne), dt), jnp.zeros((N, ni), dt))
    ones_e = ERows(jnp.ones((nx,), dt), jnp.ones((N, ne), dt),
                   jnp.ones((N, ni), dt))
    zeros_e = _tmap(jnp.zeros_like, ones_e)
    lb, ub, flo, fhi = self._bounds()
    it = Iterate(x, w, jnp.zeros((N, ni), dt), zero_rows,
                 jnp.zeros((N, ni), dt), jnp.zeros((N, ni), dt),
                 ones_e, ones_e, zeros_e, zeros_e, jnp.asarray(o.mu_init, dt))
    zdir = Direction(jnp.zeros_like(x), jnp.zeros_like(w),
                     jnp.zeros((N, ni), dt), zero_rows,
                     jnp.zeros((N, ni), dt), jnp.zeros((N, ni), dt), zeros_e,
                     zeros_e, zeros_e, zeros_e)
    lin = jax.eval_shape(lambda: self._linearize(it, params))
    lin = _tmap(lambda l: jnp.zeros(l.shape, l.dtype), lin)
    refs = Refs(x, w, it.s, jnp.asarray(0.0, dt), it, lin, zdir,
                jnp.asarray(0.0, dt), zero_rows)
    alg = AlgState(ls=self._ls_reset(dt), delta_w_last=jnp.asarray(0.0, dt),
                   mu_initialized=jnp.asarray(False),
                   tiny_step_flag=jnp.asarray(False),
                   acceptable_counter=jnp.asarray(0), stored_it=it,
                   stored_lin=lin, stored_dir=zdir)
    info = StepInfo(jnp.asarray(0.0, dt), jnp.asarray(0.0, dt), jnp.asarray(0),
                    jnp.asarray(T_X), jnp.asarray(0.0, dt), jnp.asarray(0.0, dt))
    S = State(status=jnp.asarray(CONTINUE), pending=jnp.asarray(P_INIT),
              need_lin=jnp.asarray(True), it=jnp.asarray(0),
              mode=jnp.asarray(0), cur=it, lin=lin, dir=zdir, prev=it,
              prev_lin=lin, prev_dir=zdir, soft_kind=jnp.asarray(0),
              info=info, orig=alg, resto=alg, refs=refs,
              first_resto_iter=jnp.asarray(False), n_resto=jnp.asarray(0),
              trace=jnp.full((self._trace_cap, 12), jnp.nan, dt),
              trace_n=jnp.asarray(0), max_iter=jnp.asarray(max_iter),
              stats=jnp.zeros((3,), jnp.int64))

    def cond(S_):
      return ((S_.status == CONTINUE) | (S_.pending != P_NONE)) & (
          S_.it < 3 * max_iter + 10)

    def body(S_):
      lin = jax.lax.cond(S_.need_lin,
                         lambda: self._linearize(S_.cur, params),
                         lambda: S_.lin)
      S_ = S_._replace(lin=lin, need_lin=jnp.asarray(False),
                       stats=S_.stats.at[2].add(S_.need_lin.astype(jnp.int64)))
      # One switch per pass (no frequently taken identity branches: XLA
      # materializes every pass-through leaf of an untaken-work branch).
      def check(q):
        return _scond(q.mode == 1, self._check_resto, self._check_orig, q)

      def soft_then_check(q):
        q = self._finish_soft(q)
        return _scond(q.pending == P_CHECK, check, lambda r: r, q)

      phase = jnp.where(S_.pending == P_INIT, 0,
                        jnp.where(S_.pending == P_SOFT, 1,
                                  jnp.where(S_.pending == P_CHECK, 2, 3)))
      S_ = _sswitch(phase, [self._init_phase, soft_then_check, check,
                            lambda q: q], S_)
      S_ = _scond((S_.status == CONTINUE) & (S_.pending == P_NONE) &
                  ~S_.need_lin,
                  lambda q: self._iteration(q, params), lambda q: q, S_)
      return S_

    S = _swhile(cond, body, S)
    S = self._record(S, params)
    return S

  # -------------------------------------------------------------------------
  # Public API
  # -------------------------------------------------------------------------
  def solve(self, params: dict, x_init, w_init, max_iter: int | None = None):
    """Solve from ``(x_init, w_init)``; ``params['x0']`` is the initial
    state, the remaining entries are passed to the cost functions."""
    mi = self.opt.max_iter if max_iter is None else int(max_iter)
    if mi > self.opt.max_iter:
      raise ValueError('max_iter exceeds the compiled IpPasOptions.max_iter')
    args = (params, jnp.asarray(x_init, jnp.float64),
            jnp.asarray(w_init, jnp.float64), jnp.asarray(mi))
    key = jax.tree.structure(args), tuple(
        (jnp.shape(l), jnp.result_type(l)) for l in jax.tree.leaves(args))
    fn = self._compiled.get(key)
    if fn is None:
      lowered = self._solve.lower(*args)
      opts = {}
      if jax.default_backend() == 'gpu':
        # Work around an XLA GPU Triton-GEMM fusion crash on this program.
        opts['xla_gpu_enable_triton_gemm'] = False
      fn = lowered.compile(compiler_options=opts) if opts else lowered.compile()
      self._compiled[key] = fn
    return fn(*args)

  def compile(self, params: dict, x_init, w_init):
    """Ahead-of-time compile for the given argument shapes."""
    self.solve(params, x_init, w_init, max_iter=0)

  @staticmethod
  def summarize(S: State) -> dict:
    status = int(S.status)
    st = np.asarray(S.stats)
    out = dict(status=STATUS_NAMES.get(status, str(status)),
               iterations=int(S.it), n_restorations=int(S.n_resto),
               kkt_solves=int(st[0]), model_evaluations=int(st[1]),
               linearizations=int(st[2]), x=S.cur.x, w=S.cur.w)
    n = int(S.trace_n)
    if n:
      out['trace'] = np.asarray(S.trace[:n])
    return out

  @staticmethod
  def format_trace(trace) -> str:
    lines = ['iter    objective   inf_pr   inf_du lg(mu)    ||d||  lg(rg)'
             '   alpha_du   alpha_pr ls']
    for row in np.asarray(trace):
      it, mode, obj, ipr, idu, lmu, dn, rg, adu, apr, typ, ls = row
      rgs = '     -' if rg == 0 else f'{np.log10(rg):6.1f}'
      lines.append(
          f"{int(it):4d}{'r' if mode else ' '} {obj:14.8e} {ipr:8.2e} "
          f"{idu:8.2e} {lmu:6.1f} {dn:8.2e} {rgs} {adu:10.2e} {apr:10.2e}"
          f"{TYPE_CHARS[int(typ)]} {int(ls):2d}")
    return '\n'.join(lines)
