# Policies adapted from FATROP's Ipopt-derived interior-point algorithm.
# Copyright (c) Lander Vanroye, KU Leuven.
# Ipopt: Copyright (C) 2004, 2010 International Business Machines and others.
# SPDX-License-Identifier: EPL-2.0
"""Fixed-capacity filter storage and strict-dominance tests."""

from typing import NamedTuple

import jax
import jax.numpy as jp


class Filter(NamedTuple):
  theta: jax.Array
  phi: jax.Array
  count: jax.Array
  theta_min: jax.Array
  theta_max: jax.Array


def empty_filter(capacity, theta, options):
  return Filter(
      jp.zeros(capacity, theta.dtype),
      jp.zeros(capacity, theta.dtype),
      jp.asarray(0, jp.int32),
      options.theta_min_fact * jp.maximum(1.0, theta),
      options.theta_max_fact * jp.maximum(1.0, theta),
  )


def augment_filter(state, theta, phi, options):
  """Insert a reference iterate with the line search's acceptance margins."""
  return augment(
      state, (1 - options.gamma_theta) * theta, phi - options.gamma_phi * theta
  )


def filter_acceptable(state, theta, phi):
  active = jp.arange(state.theta.size) < state.count
  return jp.all(~active | (theta <= state.theta) | (phi <= state.phi))


def reset(state):
  """Clear active corners while preserving capacity and acceptance thresholds."""
  return state._replace(count=jp.asarray(0, jp.int32))


def augment(state, theta, phi):
  """Append an unshifted corner, as in FATROP's IpFilter::augment.

  Solves reserve max_iter+1 corners; standalone callers must reserve sufficient
  capacity before appending. Acceptance margins belong to augment_filter.
  """
  return state._replace(
      theta=state.theta.at[state.count].set(theta),
      phi=state.phi.at[state.count].set(phi),
      count=state.count + 1,
  )
