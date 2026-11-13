# Policies adapted from FATROP's Ipopt-derived interior-point algorithm.
# Copyright (c) Lander Vanroye, KU Leuven.
# Ipopt: Copyright (C) 2004, 2010 International Business Machines and others.
# SPDX-License-Identifier: EPL-2.0
"""Monotone barrier policy; recompute the barrier error after every reduction."""

import jax
import jax.numpy as jp


def barrier_candidate(mu, options):
  floor = min(options.tol, options.compl_inf_tol) / (
      options.barrier_tol_factor + 1
  )
  return jp.minimum(
      mu,
      jp.maximum(
          floor,
          jp.minimum(
              options.mu_linear_decrease_factor * mu,
              mu**options.mu_superlinear_decrease_power,
          ),
      ),
  )


def update(mu, error_at_mu, options):
  """Reduce mu while its current barrier subproblem is sufficiently solved."""

  def reduce_cond(current):
    return (error_at_mu(current) < options.barrier_tol_factor * current) & (
        barrier_candidate(current, options) < current
    )

  return jax.lax.while_loop(
      reduce_cond, lambda current: barrier_candidate(current, options), mu
  )
