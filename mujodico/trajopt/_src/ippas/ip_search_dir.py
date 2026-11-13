# Policies adapted from FATROP's Ipopt-derived interior-point algorithm.
# Copyright (c) Lander Vanroye, KU Leuven.
# Ipopt: Copyright (C) 2004, 2010 International Business Machines and others.
# SPDX-License-Identifier: EPL-2.0
"""Newton directions, curvature correction, residual checks, and refinement."""

from dataclasses import replace

import jax
import jax.numpy as jp

from mujodico.trajopt._src.ippas import ip_primal_dual_system as pd
from mujodico.trajopt._src.ippas.ip_iterate import all_finite
from mujodico.trajopt._src.ippas.ip_iterate import distances
from mujodico.trajopt._src.ippas.ip_iterate import max_abs

_STEPS = (
    'dx',
    'du',
    'dlmbda',
    'dbeta',
    'ds',
    'dsf',
    'dz_l',
    'dz_u',
    'dzf_l',
    'dzf_u',
)


def _mv(matrix, vector):
  return jp.einsum('...ij,...j->...i', matrix, vector)


def residuals(p):
  """Return the residual of every equation in the original Newton system."""
  rx = p.r_x + _mv(p.Hxx, p.dx)
  rx = rx.at[:-1].add(
      _mv(p.Hxu, p.du)
      + _mv(p.A.swapaxes(-1, -2), p.dlmbda[1:])
      + _mv(p.Ex.swapaxes(-1, -2), p.dbeta)
      - _mv(p.Gx.swapaxes(-1, -2), p.dz_l - p.dz_u)
  )
  rx = rx.at[0].add(p.E0.T @ p.dlmbda[0])
  rx = rx.at[1:].add(-p.dlmbda[1:])
  rx = rx.at[-1].add(-p.Gf.T @ (p.dzf_l - p.dzf_u))
  ru = p.r_u + _mv(p.Hxu.swapaxes(-1, -2), p.dx[:-1]) + _mv(p.Huu, p.du)
  ru += _mv(p.B.swapaxes(-1, -2), p.dlmbda[1:]) + _mv(
      p.Eu.swapaxes(-1, -2), p.dbeta
  )
  ru -= _mv(p.Gu.swapaxes(-1, -2), p.dz_l - p.dz_u)
  dl, du = distances(p.s, p.g_l, p.g_u)
  dfl, dfu = distances(p.sf, p.gf_l, p.gf_u)
  return dict(
      r_x=rx,
      r_u=ru,
      r_f=p.r_f + _mv(p.A, p.dx[:-1]) + _mv(p.B, p.du) - p.dx[1:],
      r_e=p.r_e + _mv(p.Ex, p.dx[:-1]) + _mv(p.Eu, p.du),
      r_g=p.r_g + _mv(p.Gx, p.dx[:-1]) + _mv(p.Gu, p.du) - p.ds,
      r_gf=p.r_gf + p.Gf @ p.dx[-1] - p.dsf,
      r_c_l=jp.where(
          jp.isfinite(p.g_l), p.r_c_l + p.z_l * p.ds + dl * p.dz_l, 0.0
      ),
      r_c_u=jp.where(
          jp.isfinite(p.g_u), p.r_c_u - p.z_u * p.ds + du * p.dz_u, 0.0
      ),
      r_cf_l=jp.where(
          jp.isfinite(p.gf_l), p.r_cf_l + p.zf_l * p.dsf + dfl * p.dzf_l, 0.0
      ),
      r_cf_u=jp.where(
          jp.isfinite(p.gf_u), p.r_cf_u - p.zf_u * p.dsf + dfu * p.dzf_u, 0.0
      ),
      r0=p.r0 + p.E0 @ p.dx[0],
  )


def refine(solution, options):
  threshold = options.linear_tol * (
      1 + max_abs(*(getattr(solution, name) for name in residuals(solution)))
  )

  def condition(c):
    return (c[2] < options.max_refinement) & (
        max_abs(*c[1].values()) > threshold
    )

  def body(c):
    original, residual, count = c
    correction = pd.solve(original.replace_rhs(**residual))
    improved = replace(
        original,
        **{
            name: getattr(original, name) + getattr(correction, name)
            for name in _STEPS
        }
    )
    return improved, residuals(improved), count + 1

  improved, residual, _ = jax.lax.while_loop(
      condition, body, (solution, residuals(solution), jp.asarray(0, jp.int32))
  )
  return improved, max_abs(*residual.values()) <= threshold


def compute(system, mu, last_delta, options):
  """Conservative primal curvature test; no change to the linear backend.

  Positive semidefinite stage Hessians are sufficient, but not necessary,
  for reduced positive curvature. This can damp more than FATROP's Riccati
  inertia checks. The unregularized system is always attempted first.
  """
  o = options
  joint = jp.concatenate(
      (
          jp.concatenate((system.Hxx[:-1], system.Hxu), axis=-1),
          jp.concatenate((system.Hxu.swapaxes(-1, -2), system.Huu), axis=-1),
      ),
      axis=-2,
  )
  mineig = jp.minimum(
      jp.min(jp.linalg.eigvalsh(joint)),
      jp.min(jp.linalg.eigvalsh(system.Hxx[-1])),
  )
  tolerance = (
      100
      * jp.finfo(joint.dtype).eps
      * jp.maximum(1, max_abs(joint, system.Hxx[-1]))
  )

  def attempt(delta):
    shifted = replace(
        system,
        Hxx=system.Hxx + delta * jp.eye(system.Hxx.shape[-1]),
        Huu=system.Huu + delta * jp.eye(system.Huu.shape[-1]),
    )
    solved = pd.solve(shifted)
    solved, accurate = refine(solved, o)
    finite = all_finite((
        solved.dx,
        solved.du,
        solved.dlmbda,
        solved.dbeta,
        solved.ds,
        solved.dsf,
        solved.dz_l,
        solved.dz_u,
        solved.dzf_l,
        solved.dzf_u,
    ))
    if o.print_level > 1:
      jax.debug.print(
          'Newton residual={r:.2e} reg={d:.2e}',
          r=max_abs(*residuals(solved).values()),
          d=delta,
      )
    return solved, finite & accurate & (mineig + delta >= -tolerance)

  zero = jp.asarray(0.0, jp.float64)
  solved, good = attempt(zero)
  initial = (solved, good, zero, jp.asarray(1, jp.int32))

  def cond(c):
    return ~c[1] & (c[3] < o.max_regularization) & (c[2] < o.delta_w_max)

  def body(c):
    first = jp.where(
        last_delta > 0,
        jp.maximum(o.delta_wmin, o.kappa_wmin * last_delta),
        o.delta_w0,
    )
    growth = jp.where(last_delta > 0, o.kappa_wplus, o.kappa_wplusem)
    delta = jp.minimum(o.delta_w_max, jp.where(c[2] == 0, first, c[2] * growth))
    solved, good = attempt(delta)
    return solved, good, delta, c[3] + 1

  solved, good, delta, _ = jax.lax.while_loop(cond, body, initial)
  return solved, good, delta


def rank_valid(d, options):
  if not d.Eu.shape[1]:
    return jp.asarray(True)
  sv = jp.linalg.svd(d.Eu, compute_uv=False)
  return jp.all(sv[:, -1] > options.rank_tol * jp.maximum(1, sv[:, 0]))
