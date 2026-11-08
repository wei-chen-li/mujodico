"""Contact-implicit augmented Lagrangian splitting with an IPOPT primal block."""

from dataclasses import dataclass
from enum import Enum, IntEnum
from functools import partial
from numbers import Integral
from threading import Lock
from typing import NamedTuple

import jax
from jax.experimental import io_callback
import jax.numpy as jp
import numpy as np

from mujodico.trajopt._src.ipopt_trajopt import IPOptTrajOpt
from mujodico.trajopt._src.types import (
    Bounds,
    InitialEqualityConstraint,
    LocalComplementarityConstraint,
    LocalCost,
    LocalEqualityConstraint,
    LocalInequalityConstraint,
    Params,
    TerminalCost,
    TerminalEqualityConstraint,
    TerminalInequalityConstraint,
)


def project_complementarity(a: jax.Array, b: jax.Array):
  """Euclidean projection onto ``{(a, b): a >= 0, b >= 0, a*b = 0}``.

  The first branch wins positive ties, as in contact_implicit_al_admm.tex.
  """
  a, b = jp.maximum(a, 0), jp.maximum(b, 0)
  first = a >= b
  return jp.where(first, a, 0), jp.where(first, 0, b)


class _ALParams(NamedTuple):
  original: Params
  slacks: jax.Array
  multipliers: jax.Array
  rho: jax.Array


class _State(NamedTuple):
  xs: jax.Array
  us: jax.Array
  slacks: jax.Array
  multipliers: jax.Array
  info: dict


def _max_abs(value):
  return jp.max(jp.abs(value), initial=0)


class CIALSTrajOpt:
  """Partial AL splitting for a stagewise contact-implicit NLP.

  Only consistency with complementarity slacks is penalized. Dynamics and
  bounds remain hard; local inequalities referenced by complementarity pairs
  are lifted, while unreferenced inequalities remain hard. Costs and constraints
  have the same signatures as IPOptTrajOpt. ``params_trace`` supplies a pytree
  with the shapes/dtypes of runtime parameters (or None for parameter-free NLPs).

  Slacks and unscaled equality multipliers have shape ``(N, num_pairs, 2)``,
  with columns in the order of LocalComplementarityConstraint.pairs.

  Example::

    options = CIALSTrajOpt.Options(variant=CIALSTrajOpt.Variant.ADMM)
    solve = jax.jit(partial(solver.solve, options=options))
    xs, us, info = solve(params, xs_init, us_init)

  Construct the solver outside jit. Options are static; trajectory guesses,
  parameters, slacks and multipliers are dynamic. The outer loops execute in
  JAX, with an ordered host callback to IPOptTrajOpt for each trajectory block.
  Differentiation through solve and vmap of solve are not supported. Float64
  is recommended for tight IPOPT tolerances; this module does not change JAX's
  global precision configuration.
  """

  class Variant(Enum):
    AL_BCD = 'al_bcd'
    ADMM = 'admm'

  class Status(IntEnum):
    CONVERGED = 0
    MAX_ITERATIONS = 1
    INNER_FAILED = 2
    NONFINITE = 3

  @dataclass(frozen=True)
  class Options:
    variant: 'CIALSTrajOpt.Variant' = None
    rho: float = 1.0
    max_iterations: int = 100
    max_primal_sweeps: int = 10
    tolerance: float = 1e-5
    block_tolerance: float = 1e-5
    ipopt_max_iter: int = 200
    ipopt_tol: float = 1e-7
    print_level: int = 0

    def __post_init__(self):
      if self.variant is None:
        object.__setattr__(self, 'variant', CIALSTrajOpt.Variant.AL_BCD)
      if not isinstance(self.variant, CIALSTrajOpt.Variant):
        raise ValueError('variant must be a CIALSTrajOpt.Variant')
      for name in ('rho', 'tolerance', 'block_tolerance', 'ipopt_tol'):
        value = getattr(self, name)
        if not np.isfinite(value) or value <= 0:
          raise ValueError(f'{name} must be finite and positive')
      for name, minimum in (
          ('max_iterations', 0),
          ('max_primal_sweeps', 1),
          ('ipopt_max_iter', 1),
          ('print_level', 0),
      ):
        value = getattr(self, name)
        if (
            not isinstance(value, Integral)
            or isinstance(value, bool)
            or value < minimum
        ):
          raise ValueError(f'{name} must be an integer >= {minimum}')
      if self.print_level > 12:
        raise ValueError('print_level must be <= 12')

  def __init__(
      self,
      N: int,
      nx: int,
      nu: int,
      *,
      local_cost: LocalCost,
      terminal_cost: TerminalCost,
      initial_equality_constraint: InitialEqualityConstraint | None = None,
      local_equality_constraint: LocalEqualityConstraint | None = None,
      local_inequality_constraint: LocalInequalityConstraint | None = None,
      local_complementarity_constraint: (
          LocalComplementarityConstraint | None
      ) = None,
      terminal_equality_constraint: TerminalEqualityConstraint | None = None,
      terminal_inequality_constraint: (
          TerminalInequalityConstraint | None
      ) = None,
      x_bounds: Bounds | None = None,
      u_bounds: Bounds | None = None,
      gauss_newton: bool = True,
      params_trace: Params | None = None,
  ):
    if N < 1 or nx < 1 or nu < 0:
      raise ValueError('N and nx must be positive, and nu must be nonnegative')
    self.N, self.nx, self.nu = N, nx, nu
    self._local_cost = local_cost
    self._terminal_cost = terminal_cost
    self._local_inequality = local_inequality_constraint or (
        lambda x, u, xn, k, p: jp.zeros((0,))
    )
    x, u = jp.zeros(nx), jp.zeros(nu)
    ni = jax.eval_shape(
        self._local_inequality, x, u, x, jp.array(0), params_trace
    ).size
    side = LocalComplementarityConstraint.Side
    offsets = {
        side.X: 0,
        side.U: nx,
        side.XNEXT: nx + nu,
        side.INEQ: 2 * nx + nu,
    }
    sizes = {side.X: nx, side.U: nu, side.XNEXT: nx, side.INEQ: ni}
    pairs = (
        local_complementarity_constraint.pairs
        if local_complementarity_constraint is not None
        else []
    )
    indices, lifted_ineq = [], set()
    for pair in pairs:
      if len(pair) != 2:
        raise ValueError('each complementarity pair must contain two terms')
      row = []
      for term_side, index in pair:
        if (
            term_side not in sizes
            or not isinstance(index, Integral)
            or isinstance(index, bool)
            or not 0 <= index < sizes[term_side]
        ):
          raise ValueError(
              f'invalid complementarity term: {(term_side, index)}'
          )
        row.append(offsets[term_side] + index)
        if term_side == side.INEQ:
          lifted_ineq.add(index)
      indices.append(row)
    self._pair_indices = np.asarray(indices, dtype=np.int32).reshape((-1, 2))
    self.num_pairs = len(pairs)
    self._hard_ineq_indices = np.asarray(
        [i for i in range(ni) if i not in lifted_ineq], dtype=np.int32
    )

    def augmented_cost(x, u, xn, k, p):
      residual = self._local_pairs(x, u, xn, k, p.original) - p.slacks[k]
      shifted = residual + p.multipliers[k] / p.rho
      return local_cost(x, u, xn, k, p.original) + p.rho / 2 * jp.sum(
          shifted**2
      )

    def wrap_local(fn):
      if fn is None:
        return None
      return lambda x, u, xn, k, p: fn(x, u, xn, k, p.original)

    def wrap_state(fn):
      if fn is None:
        return None
      return lambda x, p: fn(x, p.original)

    al_trace = _ALParams(
        params_trace,
        jp.zeros((N, self.num_pairs, 2)),
        jp.zeros((N, self.num_pairs, 2)),
        jp.array(1.0),
    )
    self._trajectory_device = al_trace.slacks.device
    al_trace = jax.device_put(al_trace, self._trajectory_device)
    self._trajectory_solver = IPOptTrajOpt(
        N,
        nx,
        nu,
        local_cost=augmented_cost,
        terminal_cost=wrap_state(terminal_cost),
        initial_equality_constraint=wrap_state(initial_equality_constraint),
        local_equality_constraint=wrap_local(local_equality_constraint),
        local_inequality_constraint=lambda x, u, xn, k, p: (
            self._local_inequality(x, u, xn, k, p.original)[
                self._hard_ineq_indices
            ]
        ),
        terminal_equality_constraint=wrap_state(terminal_equality_constraint),
        terminal_inequality_constraint=wrap_state(
            terminal_inequality_constraint
        ),
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        gauss_newton=gauss_newton,
        params_trace=al_trace,
    )
    # IPOptTrajOpt owns mutable callback parameters/statistics. Serialize host
    # calls even when the same solver is used by concurrent compiled invocations.
    self._host_lock = Lock()
    self._nlp = self._trajectory_solver._nlp
    self._con_lb, self._con_ub = self._nlp.constraint_bounds
    self._var_lb, self._var_ub = self._nlp.var_bounds

  def _local_pairs(self, x, u, xn, k, params):
    ineq = self._local_inequality(x, u, xn, k, params)
    return jp.concatenate((x, u, xn, ineq))[self._pair_indices]

  def _pairs(self, xs, us, params):
    return jax.vmap(self._local_pairs, in_axes=(0, 0, 0, 0, None))(
        xs[:-1], us, xs[1:], jp.arange(self.N), params
    )

  @staticmethod
  def _project(values):
    return jp.stack(
        project_complementarity(values[..., 0], values[..., 1]), axis=-1
    )

  def _objective(self, xs, us, params):
    costs = jax.vmap(self._local_cost, in_axes=(0, 0, 0, 0, None))(
        xs[:-1], us, xs[1:], jp.arange(self.N), params
    )
    return jp.sum(costs) + self._terminal_cost(xs[-1], params)

  def _constraints(self, q, p):
    nlp = self._nlp
    xs, us = nlp._unpack(q)
    local = []
    for fn in (nlp.local_equality_constraint, nlp.local_inequality_constraint):
      local.append(
          jax.vmap(fn, in_axes=(0, 0, 0, 0, None))(
              xs[:-1], us, xs[1:], jp.arange(self.N), p
          )
      )
    return jp.concatenate((
        nlp.initial_equality_constraint(xs[0], p).ravel(),
        jp.concatenate(local, axis=1).ravel(),
        nlp.terminal_equality_constraint(xs[-1], p).ravel(),
        nlp.terminal_inequality_constraint(xs[-1], p).ravel(),
    ))

  def _host_solve(self, p, xs, us, *, options):
    # io_callback delivers CPU arrays, including committed parameter leaves.
    # Restore the device used to compile the inner NLP before invoking its AOT
    # functions; CPU parameters cannot be passed to a CUDA-compiled executable.
    with self._host_lock, jax.default_device(self._trajectory_device):
      p = jax.device_put(p, self._trajectory_device)
      xs_opt, us_opt, info = self._trajectory_solver.solve(
          p,
          xs,
          us,
          max_iter=options.ipopt_max_iter,
          tol=options.ipopt_tol,
          print_level=options.print_level,
          sb='yes',
          # Keep finite variable bounds hard to the requested tolerance.
          bound_relax_factor=0.0,
      )
    dtype = xs.dtype
    return (
        np.asarray(xs_opt, dtype=dtype),
        np.asarray(us_opt, dtype=dtype),
        np.asarray(info['mult_g'], dtype=dtype),
        np.asarray(info['mult_x_L'], dtype=dtype),
        np.asarray(info['mult_x_U'], dtype=dtype),
        np.asarray(info['status'], dtype=np.int32),
        np.asarray(info['total_time'], dtype=dtype),
    )

  def _sweep(self, state, params, options):
    p = _ALParams(
        params,
        state.slacks,
        state.multipliers,
        jp.asarray(options.rho, dtype=state.xs.dtype),
    )
    dtype = state.xs.dtype
    shape = lambda s: jax.ShapeDtypeStruct(s, dtype)
    xs, us, dual, zl, zu, ipopt_status, elapsed = io_callback(
        partial(self._host_solve, options=options),
        (
            shape(state.xs.shape),
            shape(state.us.shape),
            shape(self._con_lb.shape),
            shape(self._var_lb.shape),
            shape(self._var_lb.shape),
            jax.ShapeDtypeStruct((), jp.int32),
            shape(()),
        ),
        p,
        state.xs,
        state.us,
        ordered=True,
    )
    q = self._nlp._pack(xs, us)

    def pairs(q):
      return self._pairs(*self._nlp._unpack(q), params)

    contact, contact_vjp = jax.vjp(pairs, q)
    slacks = self._project(contact + state.multipliers / p.rho)
    residual = contact - slacks
    delta_slacks = slacks - state.slacks

    def hard_lagrangian(q):
      return self._objective(*self._nlp._unpack(q), params) + jp.vdot(
          dual, self._constraints(q, p)
      )

    # Evaluate stationarity AFTER projection, using the IPOPT hard-constraint
    # multipliers and the current (still fixed) AL equality multipliers.
    stationarity = jax.grad(hard_lagrangian)(q) - zl + zu
    stationarity += contact_vjp(state.multipliers + p.rho * residual)[0]
    dual_residual = p.rho * contact_vjp(delta_slacks)[0]
    constraints = self._constraints(q, p)
    feasibility = jp.maximum(
        _max_abs(jp.maximum(self._con_lb - constraints, 0)),
        _max_abs(jp.maximum(constraints - self._con_ub, 0)),
    )
    feasibility = jp.maximum(
        feasibility,
        jp.maximum(
            _max_abs(jp.maximum(self._var_lb - q, 0)),
            _max_abs(jp.maximum(q - self._var_ub, 0)),
        ),
    )
    # IPOPT uses negative multipliers for lower-bounded inequalities.
    inequality = np.isinf(self._con_ub)
    lower_distance = q - jp.where(jp.isfinite(self._var_lb), self._var_lb, q)
    upper_distance = jp.where(jp.isfinite(self._var_ub), self._var_ub, q) - q
    hard_complementarity = jp.max(
        jp.stack((
            _max_abs(jp.where(inequality, dual * constraints, 0)),
            _max_abs(zl * lower_distance),
            _max_abs(zu * upper_distance),
            _max_abs(jp.maximum(jp.where(inequality, dual, 0), 0)),
            _max_abs(jp.minimum(zl, 0)),
            _max_abs(jp.minimum(zu, 0)),
            _max_abs(jp.where(jp.isfinite(self._var_lb), 0, zl)),
            _max_abs(jp.where(jp.isfinite(self._var_ub), 0, zu)),
        ))
    )
    objective = self._objective(xs, us, params)
    metrics = {
        'objective': objective,
        'primal_residual': jp.linalg.norm(residual),
        'dual_residual': jp.linalg.norm(dual_residual),
        'stationarity': _max_abs(stationarity),
        'hard_feasibility': feasibility,
        'hard_complementarity': hard_complementarity,
        'slack_change': _max_abs(delta_slacks),
    }
    finite = jp.all(
        jp.stack([
            jp.all(jp.isfinite(v))
            for v in (
                xs,
                us,
                slacks,
                dual,
                zl,
                zu,
                stationarity,
                constraints,
                *metrics.values(),
            )
        ])
    )
    status = jp.where(
        finite,
        jp.where(
            (ipopt_status == 0) | (ipopt_status == 1),
            self.Status.MAX_ITERATIONS,
            self.Status.INNER_FAILED,
        ),
        self.Status.NONFINITE,
    ).astype(jp.int32)
    info = dict(
        state.info,
        **metrics,
        status=status,
        ipopt_status=ipopt_status,
        sweeps=state.info['sweeps'] + 1,
        ipopt_time=state.info['ipopt_time'] + elapsed,
    )
    return _State(xs, us, slacks, state.multipliers, info)

  def solve(
      self,
      params: Params,
      xs_init: jax.Array | np.ndarray,
      us_init: jax.Array | np.ndarray,
      *,
      slacks_init: jax.Array | None = None,
      multipliers_init: jax.Array | None = None,
      options: Options | None = None,
  ) -> tuple[jax.Array, jax.Array, dict]:
    """Solve with static ``options``; all returned info leaves are JAX arrays.

    ``Status.MAX_ITERATIONS`` includes a zero outer-iteration budget. An IPOPT
    failure stops both loops and suppresses that sweep's multiplier update.
    """
    options = options or self.Options()
    xs, us = jp.asarray(xs_init), jp.asarray(us_init)
    # StagewiseNLPModel compiles its trajectory functions at JAX's default
    # floating precision. Match it even when guesses have a narrower dtype;
    # otherwise bounds and residuals can change the while-loop carry dtype.
    dtype = jp.result_type(xs, us, jp.asarray(0.0, dtype=float))
    xs, us = xs.astype(dtype), us.astype(dtype)
    if xs.shape != (self.N + 1, self.nx) or us.shape != (self.N, self.nu):
      raise ValueError('initial trajectories have incorrect shapes')
    shape = (self.N, self.num_pairs, 2)
    slacks = (
        self._pairs(xs, us, params)
        if slacks_init is None
        else jp.asarray(slacks_init, dtype=dtype)
    )
    multipliers = (
        jp.zeros(shape, dtype=dtype)
        if multipliers_init is None
        else jp.asarray(multipliers_init, dtype=dtype)
    )
    if slacks.shape != shape or multipliers.shape != shape:
      raise ValueError(f'slacks and multipliers must have shape {shape}')
    slacks = self._project(slacks)
    inf = jp.asarray(jp.inf, dtype=dtype)
    info = {
        name: inf
        for name in (
            'primal_residual',
            'dual_residual',
            'stationarity',
            'hard_feasibility',
            'hard_complementarity',
            'slack_change',
        )
    }
    info.update(
        status=jp.int32(self.Status.MAX_ITERATIONS),
        ipopt_status=jp.int32(-999),
        iterations=jp.int32(0),
        sweeps=jp.int32(0),
        ipopt_time=jp.zeros((), dtype),
        objective=self._objective(xs, us, params),
        inner_converged=jp.array(False),
    )
    finite = jp.all(
        jp.stack([
            jp.all(jp.isfinite(v))
            for v in (xs, us, slacks, multipliers, info['objective'])
        ])
    )
    info['status'] = jp.where(finite, info['status'], self.Status.NONFINITE)
    state = _State(xs, us, slacks, multipliers, info)

    def inner_converged(s):
      i = s.info
      return (
          (i['stationarity'] <= options.block_tolerance)
          & (i['slack_change'] <= options.block_tolerance)
          & (i['hard_feasibility'] <= options.block_tolerance)
          & (i['hard_complementarity'] <= options.block_tolerance)
      )

    def outer_condition(s):
      return (s.info['status'] == self.Status.MAX_ITERATIONS) & (
          s.info['iterations'] < options.max_iterations
      )

    def outer_body(s):
      sweep_limit = (
          1
          if options.variant == self.Variant.ADMM or not self.num_pairs
          else options.max_primal_sweeps
      )

      def condition(carry):
        count, current = carry
        return (
            (count < sweep_limit)
            & (current.info['status'] == self.Status.MAX_ITERATIONS)
            & ((count == 0) | ~inner_converged(current))
        )

      _, new = jax.lax.while_loop(
          condition,
          lambda c: (c[0] + 1, self._sweep(c[1], params, options)),
          (jp.int32(0), s),
      )
      successful = new.info['status'] == self.Status.MAX_ITERATIONS
      residual = self._pairs(new.xs, new.us, params) - new.slacks
      multipliers = jp.where(
          successful, new.multipliers + options.rho * residual, new.multipliers
      )
      converged = (
          successful
          & (new.info['primal_residual'] <= options.tolerance)
          & (new.info['stationarity'] <= options.tolerance)
          & (new.info['hard_feasibility'] <= options.tolerance)
          & (new.info['hard_complementarity'] <= options.tolerance)
      )
      info = dict(
          new.info,
          iterations=new.info['iterations'] + successful.astype(jp.int32),
          inner_converged=inner_converged(new),
          status=jp.where(converged, self.Status.CONVERGED, new.info['status']),
      )
      return _State(new.xs, new.us, new.slacks, multipliers, info)

    state = jax.lax.while_loop(outer_condition, outer_body, state)
    info = dict(state.info, slacks=state.slacks, multipliers=state.multipliers)
    return state.xs, state.us, info
