# Policies adapted from FATROP's Ipopt-derived interior-point algorithm.
# Copyright (c) Lander Vanroye, KU Leuven.
# Ipopt: Copyright (C) 2004, 2010 International Business Machines and others.
# SPDX-License-Identifier: EPL-2.0
"""Interior-slack and equality-multiplier initialization, including warm starts."""

from dataclasses import replace

import jax
import jax.numpy as jp

from mujodico.trajopt._src.ippas import ip_primal_dual_system as pd
from mujodico.trajopt._src.ippas.ip_iterate import all_finite
from mujodico.trajopt._src.ippas.ip_iterate import Iterate
from mujodico.trajopt._src.ippas.ip_iterate import max_abs
from mujodico.trajopt._src.ippas.ip_iterate import push_interior


def initialize(model, params, guess, bounds, options):
  """Push cold-start slacks, or preserve a validated-shape primal-dual warm start."""
  m, o = model, options
  lower, upper, lf, uf = bounds
  xs, us = jp.asarray(guess.xs, jp.float64), jp.asarray(guess.us, jp.float64)
  if xs.shape != (m.N + 1, m.nx) or us.shape != (m.N, m.nu):
    raise ValueError(
        f'Initial guesses must have shapes {(m.N + 1, m.nx)} and {(m.N, m.nu)}.'
    )
  # Like FATROP, push slacks rather than the trajectory. Primal starts may
  # be infeasible; bound feasibility is recovered through g(x,u)-s = 0.
  g = m.stage_inequality_residual(xs[:-1], us, params)
  gf = m.terminal_inequality_residual(xs[-1], params)
  it = Iterate(
      xs,
      us,
      push_interior(g, lower, upper, o),
      push_interior(gf, lf, uf, o),
      jp.zeros((m.N + 1, m.nx_tangent)),
      jp.zeros((m.N, m.constraint_sizes[0])),
      jp.isfinite(upper).astype(jp.float64)
      - jp.isfinite(lower).astype(jp.float64),
      jp.isfinite(uf).astype(jp.float64) - jp.isfinite(lf).astype(jp.float64),
      jp.isfinite(lower).astype(jp.float64),
      jp.isfinite(upper).astype(jp.float64),
      jp.isfinite(lf).astype(jp.float64),
      jp.isfinite(uf).astype(jp.float64),
  )
  if guess.primal_dual is not None:
    warm = jax.tree.map(lambda v: jp.asarray(v, jp.float64), guess.primal_dual)
    for a, b in zip(jax.tree.leaves(it), jax.tree.leaves(warm)):
      if a.shape != b.shape:
        raise ValueError('Warm-start shapes must match this model.')
    # The supplied primal arrays take precedence over the saved trajectories.
    it = replace(
        warm,
        xs=jp.asarray(guess.xs, jp.float64),
        us=jp.asarray(guess.us, jp.float64),
    )
  return it


def initialize_multipliers(nlp, it, ev, d, params, bounds, options):
  """Project stationarity onto the equality Jacobian using the same scans."""
  system = nlp.system(it, ev, d, jp.asarray(0.0), params, bounds)
  zeros = {
      name: jp.zeros_like(getattr(system, name))
      for name in (
          'z_l',
          'z_u',
          'zf_l',
          'zf_u',
          'Hxu',
          'r_f',
          'r_e',
          'r0',
          'r_g',
          'r_gf',
          'r_c_l',
          'r_c_u',
          'r_cf_l',
          'r_cf_u',
      )
  }
  system = replace(
      system,
      **zeros,
      Hxx=jp.broadcast_to(jp.eye(nlp.model.nx_tangent), system.Hxx.shape),
      Huu=jp.broadcast_to(jp.eye(nlp.model.nu), system.Huu.shape),
  )
  solution = pd.solve(system)
  valid = all_finite((solution.dlmbda, solution.dbeta)) & (
      max_abs(solution.dlmbda, solution.dbeta) <= options.lam_max
  )
  return replace(
      it,
      lmbda=jp.where(valid, solution.dlmbda, it.lmbda),
      beta=jp.where(valid, solution.dbeta, it.beta),
  )
