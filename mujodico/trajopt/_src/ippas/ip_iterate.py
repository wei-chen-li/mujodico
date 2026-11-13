# Policies adapted from FATROP's Ipopt-derived interior-point algorithm.
# Copyright (c) Lander Vanroye, KU Leuven.
# Ipopt: Copyright (C) 2004, 2010 International Business Machines and others.
# SPDX-License-Identifier: EPL-2.0
"""Primal-dual state, residual metrics, and trial-iterate operations.

Corresponds to FATROP's ip_iterate component. All state is immutable JAX data.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import NamedTuple

import jax
import jax.numpy as jp


class Status(IntEnum):
  RUNNING = 0
  SUCCESS = 1
  ACCEPTABLE = 2
  MAX_ITER = 3
  NUMERICAL_ERROR = 4
  LINE_SEARCH_FAILED = 5
  UNSUPPORTED_EQUALITIES = 6
  INVALID_INITIAL_GUESS = 7


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Iterate:
  xs: jax.Array
  us: jax.Array
  s: jax.Array
  sf: jax.Array
  lmbda: jax.Array
  beta: jax.Array
  y: jax.Array
  yf: jax.Array
  z_l: jax.Array
  z_u: jax.Array
  zf_l: jax.Array
  zf_u: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class InitialGuess:
  xs: jax.Array
  us: jax.Array
  # A warm start is explicit; ordinary solves always initialize fresh duals.
  primal_dual: Iterate | None = None


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SolveResult:
  iterate: Iterate
  status: jax.Array
  iterations: jax.Array
  objective: jax.Array
  primal_error: jax.Array
  dual_error: jax.Array
  complementarity: jax.Array
  error: jax.Array
  mu: jax.Array

  @property
  def xs(self):
    return self.iterate.xs

  @property
  def us(self):
    return self.iterate.us

  @property
  def success(self):
    return (self.status == Status.SUCCESS) | (self.status == Status.ACCEPTABLE)


def max_abs(*values):
  return jp.max(jp.stack([jp.max(jp.abs(v), initial=0.0) for v in values]))


def all_finite(tree):
  return jp.all(
      jp.stack([jp.all(jp.isfinite(v)) for v in jax.tree.leaves(tree)])
  )


def distances(s, lower, upper):
  """Omitted sides have unit distance, keeping inactive arithmetic finite."""
  return (
      jp.where(jp.isfinite(lower), s - lower, 1.0),
      jp.where(jp.isfinite(upper), upper - s, 1.0),
  )


def push_interior(value, lower, upper, options):
  ml, mu = jp.isfinite(lower), jp.isfinite(upper)
  # Form finite intervals before taking their difference to avoid inf-inf.
  width = jp.where(ml & mu, upper, 1.0) - jp.where(ml & mu, lower, 0.0)
  pl = jp.minimum(
      options.bound_push * jp.maximum(1.0, jp.abs(lower)),
      options.bound_frac * width,
  )
  pu = jp.minimum(
      options.bound_push * jp.maximum(1.0, jp.abs(upper)),
      options.bound_frac * width,
  )
  pl = jp.where(
      ml & mu, pl, options.bound_push * jp.maximum(1.0, jp.abs(lower))
  )
  pu = jp.where(
      ml & mu, pu, options.bound_push * jp.maximum(1.0, jp.abs(upper))
  )
  lo = jp.where(
      ml, jp.maximum(lower + pl, jp.nextafter(lower, jp.inf)), -jp.inf
  )
  hi = jp.where(
      mu, jp.minimum(upper - pu, jp.nextafter(upper, -jp.inf)), jp.inf
  )
  return jp.clip(value, lo, hi)


def complementarity(s, zl, zu, lower, upper, target):
  dl, du = distances(s, lower, upper)
  return (
      jp.where(jp.isfinite(lower), dl * zl - target, 0.0),
      jp.where(jp.isfinite(upper), du * zu - target, 0.0),
  )


def maximum_step(value, step, lower, upper, tau):
  dl, du = distances(value, lower, upper)
  lower_limit = jp.where(
      jp.isfinite(lower) & (step < 0),
      -tau * dl / jp.where(step < 0, step, -1.0),
      1.0,
  )
  upper_limit = jp.where(
      jp.isfinite(upper) & (step > 0),
      tau * du / jp.where(step > 0, step, 1.0),
      1.0,
  )
  return jp.clip(
      jp.minimum(
          jp.min(lower_limit, initial=1.0), jp.min(upper_limit, initial=1.0)
      ),
      0.0,
      1.0,
  )


class Evaluation(NamedTuple):
  """Objective and residuals; inequality and terminal contain g-s, not raw g."""

  objective: jax.Array
  initial: jax.Array
  dynamics: jax.Array
  equality: jax.Array
  inequality: jax.Array
  terminal: jax.Array


class Derivatives(NamedTuple):
  """Tangent-coordinate derivatives and Lagrangian stationarity residuals."""

  gx: jax.Array
  gu: jax.Array
  A: jax.Array
  B: jax.Array
  Ex: jax.Array
  Eu: jax.Array
  Gx: jax.Array
  Gu: jax.Array
  Gf: jax.Array
  E0: jax.Array
  rx: jax.Array
  ru: jax.Array


class Metrics(NamedTuple):
  """Original-bound feasibility, raw norms, and scaled relaxed-NLP errors."""

  primal: jax.Array
  dual: jax.Array
  complementarity: jax.Array
  error: jax.Array
  barrier_error: jax.Array


def _mv(a, v):
  return jp.einsum('...ij,...j->...i', a, v)


def damping(lower, upper, mu, options):
  return (
      options.kappa_d
      * mu
      * (
          jp.isfinite(lower).astype(jp.float64)
          - jp.isfinite(upper).astype(jp.float64)
      )
  )


def valid(model, it, bounds):
  m = model
  valid = all_finite(it)
  for s, zl, zu, l, u in (
      (it.s, it.z_l, it.z_u, *bounds[:2]),
      (it.sf, it.zf_l, it.zf_u, *bounds[2:]),
  ):
    dl, du = distances(s, l, u)
    valid &= jp.all((dl > 0) & (du > 0))
    valid &= jp.all(jp.where(jp.isfinite(l), zl > 0, zl == 0))
    valid &= jp.all(jp.where(jp.isfinite(u), zu > 0, zu == 0))
  for i in m.quaternion_positions:
    valid &= jp.all(
        jp.abs(jp.sum(it.xs[:, i : i + 4] ** 2, axis=1) - 1) <= 1e-6
    )
  return valid


def metrics(it, ev, d, mu, bounds, original_bounds, options):
  """Measure original feasibility and both original and barrier KKT errors.

  Each bounds tuple contains stage lower/upper, then terminal lower/upper.
  """
  o = options
  lower, upper, lf, uf = bounds
  cl, cu = complementarity(it.s, it.z_l, it.z_u, lower, upper, 0.0)
  cfl, cfu = complementarity(it.sf, it.zf_l, it.zf_u, lf, uf, 0.0)
  cml, cmu = complementarity(it.s, it.z_l, it.z_u, lower, upper, mu)
  cmfl, cmfu = complementarity(it.sf, it.zf_l, it.zf_u, lf, uf, mu)
  raw, rawf = ev.inequality + it.s, ev.terminal + it.sf
  ol, ou, olf, ouf = original_bounds
  violation = max_abs(
      jp.maximum(ol - raw, 0),
      jp.maximum(raw - ou, 0),
      jp.maximum(olf - rawf, 0),
      jp.maximum(rawf - ouf, 0),
  )
  constraint_error = max_abs(*ev[1:])
  primal = jp.maximum(violation, constraint_error)
  rs = -it.y - it.z_l + it.z_u
  rsf = -it.yf - it.zf_l + it.zf_u
  dual, comp = max_abs(d.rx, d.ru, rs, rsf), max_abs(cl, cu, cfl, cfu)
  zsum = jp.sum(it.z_l) + jp.sum(it.z_u) + jp.sum(it.zf_l) + jp.sum(it.zf_u)
  zcount = sum(jp.sum(jp.isfinite(v)) for v in bounds)
  eqsum = sum(jp.sum(jp.abs(v)) for v in (it.lmbda, it.beta, it.y, it.yf))
  eqcount = it.lmbda.size + it.beta.size + it.y.size + it.yf.size
  sd = jp.maximum(
      1.0, (zsum + eqsum) / jp.maximum(1, zcount + eqcount) / o.s_max
  )
  sc = jp.maximum(1.0, zsum / jp.maximum(1, zcount) / o.s_max)
  damp, dampf = damping(lower, upper, mu, options), damping(lf, uf, mu, options)
  return Metrics(
      primal,
      dual,
      comp,
      # KKT errors target the relaxed NLP; original bounds are checked
      # separately against constr_viol_tol and retained in primal_error.
      jp.maximum(constraint_error, jp.maximum(dual / sd, comp / sc)),
      jp.maximum(
          constraint_error,
          jp.maximum(
              max_abs(d.rx, d.ru, rs + damp, rsf + dampf) / sd,
              max_abs(cml, cmu, cmfl, cmfu) / sc,
          ),
      ),
  )


def merit(it, ev, mu, bounds, options):
  """Return the barrier objective phi and equality-residual L1 norm theta."""
  barrier = jp.asarray(0.0, jp.float64)
  for s, l, u in ((it.s, *bounds[:2]), (it.sf, *bounds[2:])):
    dl, du = distances(s, l, u)
    barrier -= mu * jp.sum(
        jp.where(jp.isfinite(l), jp.log(dl), 0)
        + jp.where(jp.isfinite(u), jp.log(du), 0)
    )
    barrier += jp.sum(damping(l, u, mu, options) * s)
  theta = sum(jp.sum(jp.abs(v)) for v in ev[1:])
  return ev.objective + barrier, theta


def step_lengths(it, direction, mu, bounds, options):
  """Return independent primal-slack and bound-dual fraction-to-boundary limits."""
  tau = jp.maximum(options.tau_min, 1 - mu)
  ap = jp.minimum(
      maximum_step(it.s, direction.ds, *bounds[:2], tau),
      maximum_step(it.sf, direction.dsf, *bounds[2:], tau),
  )
  ad = jp.asarray(1.0, jp.float64)
  for z, dz, bound in (
      (it.z_l, direction.dz_l, bounds[0]),
      (it.z_u, direction.dz_u, bounds[1]),
      (it.zf_l, direction.dzf_l, bounds[2]),
      (it.zf_u, direction.dzf_u, bounds[3]),
  ):
    ad = jp.minimum(
        ad,
        maximum_step(
            z,
            dz,
            jp.where(jp.isfinite(bound), 0.0, -jp.inf),
            jp.full_like(z, jp.inf),
            tau,
        ),
    )
  return ap, ad


def candidate(
    model, it, direction, alpha, alpha_dual, mu, params, bounds, options
):
  """Retract primals and update equality/bound duals with their own step lengths."""
  s, sf = it.s + alpha * direction.ds, it.sf + alpha * direction.dsf

  def dual(z, dz, distance, bound):
    central = mu / distance
    return jp.where(
        jp.isfinite(bound),
        jp.clip(
            z + alpha_dual * dz,
            central / options.kappa_sigma,
            central * options.kappa_sigma,
        ),
        0.0,
    )

  dl, du = distances(s, *bounds[:2])
  dfl, dfu = distances(sf, *bounds[2:])
  return Iterate(
      model.retract_state(it.xs, direction.dx, alpha, params),
      model.retract_control(it.us, direction.du, alpha, params),
      s,
      sf,
      it.lmbda + alpha * direction.dlmbda,
      it.beta + alpha * direction.dbeta,
      it.y
      + alpha
      * (
          direction.dz_u
          - direction.dz_l
          - it.y
          - it.z_l
          + it.z_u
          + damping(bounds[0], bounds[1], mu, options)
      ),
      it.yf
      + alpha
      * (
          direction.dzf_u
          - direction.dzf_l
          - it.yf
          - it.zf_l
          + it.zf_u
          + damping(bounds[2], bounds[3], mu, options)
      ),
      dual(it.z_l, direction.dz_l, dl, bounds[0]),
      dual(it.z_u, direction.dz_u, du, bounds[1]),
      dual(it.zf_l, direction.dzf_l, dfl, bounds[2]),
      dual(it.zf_u, direction.dzf_u, dfu, bounds[3]),
  )
