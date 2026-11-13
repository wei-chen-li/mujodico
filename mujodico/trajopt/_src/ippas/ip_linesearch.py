# Policies adapted from FATROP's Ipopt-derived interior-point algorithm.
# Copyright (c) Lander Vanroye, KU Leuven.
# Ipopt: Copyright (C) 2004, 2010 International Business Machines and others.
# SPDX-License-Identifier: EPL-2.0
"""Filter/Armijo backtracking, second-order corrections, and filter resets."""

from typing import NamedTuple

import jax
import jax.numpy as jp

from mujodico.trajopt._src.ippas import ip_filter
from mujodico.trajopt._src.ippas import ip_iterate
from mujodico.trajopt._src.ippas import ip_primal_dual_system as pd
from mujodico.trajopt._src.ippas import ip_search_dir
from mujodico.trajopt._src.ippas.ip_iterate import damping
from mujodico.trajopt._src.ippas.ip_iterate import distances
from mujodico.trajopt._src.ippas.ip_iterate import Evaluation
from mujodico.trajopt._src.ippas.ip_iterate import Iterate


class Search(NamedTuple):
  iterate: Iterate
  evaluation: Evaluation
  accepted: jax.Array
  alpha: jax.Array
  count: jax.Array
  filter_rejected: jax.Array


def _le(left, right, reference):
  rounding = (
      10
      * jp.finfo(jp.result_type(left, right)).eps
      * jp.maximum(1.0, jp.abs(reference))
  )
  return left <= right + rounding


def f_type(alpha_test, theta, dphi, options):
  # FATROP handles a tiny positive derivative at an exactly feasible point.
  eps = jp.finfo(dphi.dtype).eps
  dphi = jp.where((theta == 0) & (dphi > 0) & (dphi < 100 * eps), -eps, dphi)
  return (dphi < 0) & (
      alpha_test * jp.maximum(-dphi, 0) ** options.s_phi
      > options.delta * theta**options.s_theta
  )


def armijo(phi, phi0, alpha_test, dphi, options):
  return _le(phi - phi0, options.eta_phi * alpha_test * dphi, phi0)


def acceptable(
    phi, theta, finite, phi0, theta0, dphi, alpha_test, state, options
):
  switching = f_type(alpha_test, theta0, dphi, options)
  sufficient = _le(theta, (1 - options.gamma_theta) * theta0, theta0) | _le(
      phi - phi0, -options.gamma_phi * theta0, phi0
  )
  scale = jp.where(
      jp.abs(phi0) > 10, jp.log10(jp.maximum(jp.abs(phi0), 1.0)), 1.0
  )
  rise_ok = (phi <= phi0) | (
      jp.log10(jp.maximum(phi - phi0, jp.finfo(phi.dtype).tiny))
      <= options.obj_max_incr + scale
  )
  current_ok = jp.where(
      switching & (theta0 < state.theta_min),
      armijo(phi, phi0, alpha_test, dphi, options),
      sufficient & rise_ok,
  )
  filter_ok = ip_filter.filter_acceptable(state, theta, phi)
  return (
      finite
      & jp.isfinite(phi)
      & jp.isfinite(theta)
      & (theta <= state.theta_max)
      & current_ok
      & filter_ok
  )


def minimum_step(theta, dphi, state, options):
  descent = jp.maximum(-dphi, jp.finfo(dphi.dtype).tiny)
  alpha = jp.minimum(options.gamma_theta, options.gamma_phi * theta / descent)
  alpha = jp.where(
      theta <= state.theta_min,
      jp.minimum(
          alpha,
          options.delta * theta**options.s_theta / descent**options.s_phi,
      ),
      alpha,
  )
  return options.alpha_min_frac * jp.where(
      (theta > 0) & (dphi < 0), alpha, options.gamma_theta
  )


def find_acceptable_trial_point(
    nlp, it, ev, d, direction, mu, params, bounds, filter_state, options
):
  o = options
  alpha_max, ad = ip_iterate.step_lengths(it, direction, mu, bounds, options)
  phi0, theta0 = ip_iterate.merit(it, ev, mu, bounds, options)
  dphi = jp.sum(d.gx * direction.dx) + jp.sum(d.gu * direction.du)
  for s, ds, l, u in (
      (it.s, direction.ds, *bounds[:2]),
      (it.sf, direction.dsf, *bounds[2:]),
  ):
    dl, du = distances(s, l, u)
    dphi += jp.sum(
        ds
        * (
            -mu * jp.where(jp.isfinite(l), 1 / dl, 0.0)
            + mu * jp.where(jp.isfinite(u), 1 / du, 0.0)
            + damping(l, u, mu, options)
        )
    )

  def trial(direction, alpha, ad):
    candidate = ip_iterate.candidate(
        nlp.model, it, direction, alpha, ad, mu, params, bounds, options
    )
    evaluation = nlp.evaluate(candidate, params)
    phi, theta = ip_iterate.merit(candidate, evaluation, mu, bounds, options)
    accepted = acceptable(
        phi,
        theta,
        ip_iterate.valid(nlp.model, candidate, bounds),
        phi0,
        theta0,
        dphi,
        alpha_max,
        filter_state,
        o,
    )
    current_ok = acceptable(
        phi,
        theta,
        ip_iterate.valid(nlp.model, candidate, bounds),
        phi0,
        theta0,
        dphi,
        alpha_max,
        filter_state._replace(count=jp.asarray(0, jp.int32)),
        o,
    )
    rejected_by_filter = current_ok & ~ip_filter.filter_acceptable(
        filter_state, theta, phi
    )
    return (
        Search(
            candidate,
            evaluation,
            accepted,
            alpha,
            jp.asarray(0, jp.int32),
            rejected_by_filter,
        ),
        theta,
    )

  first, theta_first = trial(direction, alpha_max, ad)

  # SOC modifies only constraint RHSs and reuses the Newton factors. A failed
  # correction never replaces the original Newton direction for backtracking.
  initial = (
      first,
      direction,
      ev,
      theta_first,
      jp.asarray(jp.inf),
      jp.asarray(0, jp.int32),
  )

  def soc_cond(c):
    candidate, _, _, theta, old_theta, count = c
    return (
        (~candidate.accepted)
        & (theta_first >= theta0)
        & (count < o.max_soc)
        & (theta < o.kappa_soc * old_theta)
    )

  def soc_body(c):
    previous, solved, accumulated, theta, _, count = c
    acc = Evaluation(
        ev.objective,
        *(
            a + previous.alpha * b
            for a, b in zip(accumulated[1:], previous.evaluation[1:])
        ),
    )
    corrected = pd.solve(
        direction.replace_rhs(
            r0=acc.initial,
            r_f=acc.dynamics,
            r_e=acc.equality,
            r_g=acc.inequality,
            r_gf=acc.terminal,
        )
    )
    corrected, accurate = ip_search_dir.refine(corrected, o)
    ap, ad_soc = ip_iterate.step_lengths(it, corrected, mu, bounds, options)
    candidate, new_theta = trial(corrected, ap, ad_soc)
    candidate = candidate._replace(
        accepted=candidate.accepted & accurate,
        filter_rejected=previous.filter_rejected | candidate.filter_rejected,
    )
    return candidate, corrected, acc, new_theta, theta, count + 1

  corrected, _, _, _, _, _ = jax.lax.while_loop(soc_cond, soc_body, initial)
  alpha_min = minimum_step(theta0, dphi, filter_state, o)

  def backtrack_cond(c):
    return (
        ~c.accepted
        & (c.count < o.max_ls - 1)
        & (c.alpha * o.alpha_red_factor > alpha_min)
    )

  def backtrack_body(c):
    candidate, _ = trial(direction, c.alpha * o.alpha_red_factor, ad)
    return candidate._replace(
        count=c.count + 1,
        filter_rejected=c.filter_rejected | candidate.filter_rejected,
    )

  found = jax.lax.cond(
      corrected.accepted,
      lambda _: corrected,
      lambda _: jax.lax.while_loop(backtrack_cond, backtrack_body, first),
      None,
  )
  found = found._replace(
      filter_rejected=found.filter_rejected | corrected.filter_rejected
  )
  phi, _ = ip_iterate.merit(
      found.iterate, found.evaluation, mu, bounds, options
  )
  augment = found.accepted & (
      ~f_type(alpha_max, theta0, dphi, o)
      | ~armijo(phi, phi0, alpha_max, dphi, o)
  )
  updated_filter = jax.lax.cond(
      augment,
      lambda f: ip_filter.augment_filter(f, theta0, phi0, o),
      lambda f: f,
      filter_state,
  )
  return found, updated_filter


def update_filter_rejections(state, found, rejections, resets, options):
  """Reset filter memory after repeated filter-induced shortened steps."""
  rejections = jp.where(found.filter_rejected, rejections + 1, 0)
  reset = (
      found.accepted
      & (rejections >= options.filter_reset_trigger)
      & ((options.max_filter_resets < 0) | (resets < options.max_filter_resets))
  )
  state = jax.lax.cond(reset, ip_filter.reset, lambda f: f, state)
  return state, jp.where(reset, 0, rejections), resets + reset.astype(jp.int32)
