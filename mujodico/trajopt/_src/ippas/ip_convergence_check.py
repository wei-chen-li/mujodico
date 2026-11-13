# Policies adapted from FATROP's Ipopt-derived interior-point algorithm.
# Copyright (c) Lander Vanroye, KU Leuven.
# Ipopt: Copyright (C) 2004, 2010 International Business Machines and others.
# SPDX-License-Identifier: EPL-2.0
"""KKT convergence, acceptable iterates, and numerical failure classification."""

import jax.numpy as jp

from mujodico.trajopt._src.ippas import ip_iterate
from mujodico.trajopt._src.ippas import ip_search_dir
from mujodico.trajopt._src.ippas.ip_iterate import all_finite
from mujodico.trajopt._src.ippas.ip_iterate import Status


def check(model, it, ev, d, metrics, count, iteration, bounds, options):
  o = options
  status = jp.where(iteration >= o.max_iter, Status.MAX_ITER, Status.RUNNING)
  feasible = metrics.primal <= o.constr_viol_tol
  status = jp.where(
      (o.acceptable_iter > 0) & (count >= o.acceptable_iter) & feasible,
      Status.ACCEPTABLE,
      status,
  )
  status = jp.where((metrics.error <= o.tol) & feasible, Status.SUCCESS, status)
  status = jp.where(
      ip_search_dir.rank_valid(d, options),
      status,
      Status.UNSUPPORTED_EQUALITIES,
  )
  status = jp.where(
      ip_iterate.valid(model, it, bounds) & all_finite((ev, d, metrics)),
      status,
      Status.NUMERICAL_ERROR,
  )
  return status.astype(jp.int32)
