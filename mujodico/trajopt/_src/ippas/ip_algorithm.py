# Policies adapted from FATROP's Ipopt-derived interior-point algorithm.
# Copyright (c) Lander Vanroye, KU Leuven.
# Ipopt: Copyright (C) 2004, 2010 International Business Machines and others.
# SPDX-License-Identifier: EPL-2.0
"""Public solver and compiled interior-point iteration controller."""

from typing import NamedTuple

import jax
import jax.numpy as jp
import numpy as np

from mujodico.trajopt._src.ippas import ip_convergence_check
from mujodico.trajopt._src.ippas import ip_filter
from mujodico.trajopt._src.ippas import ip_initializer
from mujodico.trajopt._src.ippas import ip_iterate
from mujodico.trajopt._src.ippas import ip_linesearch
from mujodico.trajopt._src.ippas import ip_mu_update
from mujodico.trajopt._src.ippas import ip_search_dir
from mujodico.trajopt._src.ippas.ip_iterate import Derivatives
from mujodico.trajopt._src.ippas.ip_iterate import Evaluation
from mujodico.trajopt._src.ippas.ip_iterate import InitialGuess
from mujodico.trajopt._src.ippas.ip_iterate import Iterate
from mujodico.trajopt._src.ippas.ip_iterate import Metrics
from mujodico.trajopt._src.ippas.ip_iterate import SolveResult
from mujodico.trajopt._src.ippas.ip_iterate import Status
from mujodico.trajopt._src.ippas.ip_nlp_orig import IpNlp
from mujodico.trajopt._src.ippas.options import Options
from mujodico.trajopt._src.stagewise_ocp_model import StagewiseOcpModel


class State(NamedTuple):
  iterate: Iterate
  evaluation: Evaluation
  derivatives: Derivatives
  metrics: Metrics
  iteration: jax.Array
  status: jax.Array
  acceptable_count: jax.Array
  mu: jax.Array
  filter: ip_filter.Filter
  last_delta: jax.Array
  filter_rejections: jax.Array
  filter_resets: jax.Array


class IppasSolver:
  """Primal-dual interior-point solver taking a StagewiseOcpModel directly.

  ``jax.jit(solver.solve)(params, InitialGuess(xs, us))`` compiles all iteration
  and globalization logic. The three-argument ``solve(params, xs, us)`` form
  is also supported. Options and model structure are static; parameters and
  initial guesses are dynamic. Enable JAX float64 before constructing models.

  The unchanged linear backend requires N>=1, nu>=1, no terminal equalities,
  and full-row-rank stage control equality Jacobians. Rank loss during a solve
  returns UNSUPPORTED_EQUALITIES. Quaternion retractions and either dynamics
  Jacobian convention are supplied by the model. Bounds on quaternion storage
  components are currently unsupported.
  """

  def __init__(self, model: StagewiseOcpModel, options: Options | None = None):
    if not isinstance(model, StagewiseOcpModel):
      raise TypeError('model must be a StagewiseOcpModel.')
    if options is not None and not isinstance(options, Options):
      raise TypeError('options must be an Options instance.')
    if model.N < 1 or model.nu < 1:
      raise ValueError('IPPAS currently requires N >= 1 and nu >= 1.')
    for i in model.quaternion_positions:
      if any(np.any(np.isfinite(v[i : i + 4])) for v in model.x_bounds):
        raise ValueError(
            'IPPAS does not support bounds on quaternion storage components.'
        )
    self.model = model
    self.options = options or Options()

    self._nlp = IpNlp(model, self.options)

  def solve(self, params, initial_guess, us_init=None) -> SolveResult:
    """Return a compact result, entirely on device, including full warm state."""
    if not jax.config.x64_enabled:
      raise ValueError(
          'IPPAS requires float64; enable jax_enable_x64 before constructing'
          ' the model.'
      )
    m, o = self.model, self.options
    nlp = self._nlp
    # Model initialization creates closures over stage indices. Keep those
    # structural constants out of the surrounding solve/vmap trace so a cold
    # compiled solve cannot leak tracers into the reusable model instance.
    with jax.ensure_compile_time_eval():
      m.initialize(params)
    if m.evaluation_dtype != jp.float64:
      raise ValueError(
          'StagewiseOcpModel must be initialized with float64 enabled.'
      )
    ne, _, nef, _ = m.constraint_sizes
    if nef:
      raise ValueError('IPPAS currently does not support terminal equalities.')
    if ne > m.nu:
      raise ValueError(
          'IPPAS requires no more stage equalities than controls and'
          ' full-row-rank Eu.'
      )
    if us_init is not None:
      initial_guess = InitialGuess(initial_guess, us_init)
    if isinstance(initial_guess, Iterate):
      initial_guess = InitialGuess(
          initial_guess.xs, initial_guess.us, initial_guess
      )
    if not isinstance(initial_guess, InitialGuess):
      raise TypeError(
          'Pass InitialGuess, Iterate, or separate xs_init and us_init arrays.'
      )
    bounds = nlp.bounds()
    original_bounds = nlp.bounds(original=True)
    it = ip_initializer.initialize(m, params, initial_guess, bounds, o)
    ev, d = nlp.evaluate(it, params), nlp.derivatives(it, params)
    if o.initialize_duals and initial_guess.primal_dual is None:
      it = ip_initializer.initialize_multipliers(
          nlp, it, ev, d, params, bounds, o
      )
      d = nlp.derivatives(it, params)
    mu = jp.asarray(o.mu_init, jp.float64)
    metrics = ip_iterate.metrics(it, ev, d, mu, bounds, original_bounds, o)
    zero = jp.asarray(0, jp.int32)
    status = ip_convergence_check.check(
        m, it, ev, d, metrics, zero, zero, bounds, o
    )
    status = jp.where(
        ip_iterate.valid(m, it, bounds), status, Status.INVALID_INITIAL_GUESS
    ).astype(jp.int32)
    theta = sum(jp.sum(jp.abs(v)) for v in ev[1:])
    state = State(
        it,
        ev,
        d,
        metrics,
        zero,
        status,
        zero,
        mu,
        ip_filter.empty_filter(o.max_iter + 1, theta, o),
        jp.asarray(0.0, jp.float64),
        zero,
        zero,
    )

    def body(st):
      new_mu = ip_mu_update.update(
          st.mu,
          lambda mu: ip_iterate.metrics(
              st.iterate,
              st.evaluation,
              st.derivatives,
              mu,
              bounds,
              original_bounds,
              o,
          ).barrier_error,
          o,
      )
      fs = jax.lax.cond(
          new_mu != st.mu,
          lambda _: ip_filter.empty_filter(
              o.max_iter + 1,
              sum(jp.sum(jp.abs(v)) for v in st.evaluation[1:]),
              o,
          ),
          lambda _: st.filter,
          None,
      )
      system = nlp.system(
          st.iterate,
          st.evaluation,
          st.derivatives,
          new_mu,
          params,
          bounds,
      )
      direction, good, delta = ip_search_dir.compute(
          system, new_mu, st.last_delta, o
      )
      found, fs = ip_linesearch.find_acceptable_trial_point(
          nlp,
          st.iterate,
          st.evaluation,
          st.derivatives,
          direction,
          new_mu,
          params,
          bounds,
          fs,
          o,
      )
      accepted = good & found.accepted
      fs, rejections, resets = ip_linesearch.update_filter_rejections(
          fs,
          found._replace(accepted=accepted),
          st.filter_rejections,
          st.filter_resets,
          o,
      )
      it = jax.tree.map(
          lambda a, b: jp.where(accepted, a, b), found.iterate, st.iterate
      )
      ev = nlp.evaluate(it, params)
      d = nlp.derivatives(it, params)
      mm = ip_iterate.metrics(it, ev, d, new_mu, bounds, original_bounds, o)
      count = jp.where(mm.error <= o.tol_acceptable, st.acceptable_count + 1, 0)
      iteration = st.iteration + 1
      status = ip_convergence_check.check(
          m, it, ev, d, mm, count, iteration, bounds, o
      )
      status = jp.where(~accepted, Status.LINE_SEARCH_FAILED, status)
      status = jp.where(~good, Status.NUMERICAL_ERROR, status).astype(jp.int32)
      if o.print_level:
        jax.debug.print(
            'IPPAS {i}: f={f:.8e} primal={p:.2e} dual={d:.2e} compl={c:.2e}'
            ' mu={mu:.2e} alpha={a:.2e} reg={r:.2e}',
            i=iteration,
            f=ev.objective,
            p=mm.primal,
            d=mm.dual,
            c=mm.complementarity,
            mu=new_mu,
            a=found.alpha,
            r=delta,
        )
      return State(
          it,
          ev,
          d,
          mm,
          iteration,
          status,
          count,
          new_mu,
          fs,
          jp.where(delta > 0, delta, st.last_delta),
          rejections,
          resets,
      )

    final = jax.lax.while_loop(
        lambda st: st.status == Status.RUNNING, body, state
    )
    mm = final.metrics
    return SolveResult(
        final.iterate,
        final.status,
        final.iteration,
        final.evaluation.objective,
        mm.primal,
        mm.dual,
        mm.complementarity,
        mm.error,
        final.mu,
    )
