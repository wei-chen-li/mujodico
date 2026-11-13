# Policies adapted from FATROP's Ipopt-derived interior-point algorithm.
# Copyright (c) Lander Vanroye, KU Leuven.
# Ipopt: Copyright (C) 2004, 2010 International Business Machines and others.
# SPDX-License-Identifier: EPL-2.0
"""Stagewise model evaluation and assembly of the reduced Newton system.

The adapter owns no iterate or globalization state. This is the IPPAS analogue
of FATROP's ip_nlp_orig component, backed directly by StagewiseOcpModel.
"""

import jax.numpy as jp

from mujodico.trajopt._src.ippas import ip_primal_dual_system as pd
from mujodico.trajopt._src.ippas.ip_iterate import _mv
from mujodico.trajopt._src.ippas.ip_iterate import complementarity
from mujodico.trajopt._src.ippas.ip_iterate import damping
from mujodico.trajopt._src.ippas.ip_iterate import Derivatives
from mujodico.trajopt._src.ippas.ip_iterate import Evaluation


class IpNlp:
  """Adapter for a fixed model/options pair; all numerical state is supplied."""

  def __init__(self, model, options):
    self.model = model
    self.options = options

  def bounds(self, *, original=False):
    """Return stage/terminal lower and upper bounds, optionally unrelaxed."""
    m = self.model
    lower, upper = (
        jp.broadcast_to(jp.asarray(v), (m.N, v.size))
        for v in m.stage_inequality_bounds
    )
    lower_f, upper_f = map(jp.asarray, m.terminal_inequality_bounds)
    bounds = lower, upper, lower_f, upper_f
    if original:
      return bounds
    result = []
    for i, bound in enumerate(bounds):
      finite = jp.isfinite(bound)
      change = jp.minimum(
          self.options.constr_viol_tol,
          self.options.bound_relax_factor
          * jp.maximum(1.0, jp.abs(jp.where(finite, bound, 0.0))),
      )
      result.append(
          jp.where(finite, bound + (-1 if i % 2 == 0 else 1) * change, bound)
      )
    return tuple(result)

  def evaluate(self, it, params):
    """Evaluate the objective and initial/dynamics/path/slack residuals."""
    m = self.model
    return Evaluation(
        jp.sum(m.objective(it.xs, it.us, params)),
        m.initial_residual(it.xs[0], params),
        m.dynamics_residual(it.xs[:-1], it.us, it.xs[1:], params),
        m.stage_equality_residual(it.xs[:-1], it.us, params),
        m.stage_inequality_residual(it.xs[:-1], it.us, params) - it.s,
        m.terminal_inequality_residual(it.xs[-1], params) - it.sf,
    )

  def derivatives(self, it, params):
    """Evaluate model Jacobians and assemble full-space stationarity residuals."""
    m = self.model
    gx, gu = m.objective_gradient(it.xs, it.us, params)
    A, B = m.dynamics_jacobian(it.xs[:-1], it.us, it.xs[1:], params)
    Ex, Eu = m.stage_equality_jacobian(it.xs[:-1], it.us, params)
    Gx, Gu = m.stage_inequality_jacobian(it.xs[:-1], it.us, params)
    Gf = m.terminal_inequality_jacobian(it.xs[-1], params)
    E0 = m.initial_jacobian(it.xs[0], params)
    li, lif = it.y, it.yf
    rx = gx.at[:-1].add(_mv(A.swapaxes(-1, -2), it.lmbda[1:]))
    rx = rx.at[1:].add(-it.lmbda[1:])
    rx = rx.at[0].add(E0.T @ it.lmbda[0])
    rx = rx.at[:-1].add(
        _mv(Ex.swapaxes(-1, -2), it.beta) + _mv(Gx.swapaxes(-1, -2), li)
    )
    rx = rx.at[-1].add(Gf.T @ lif)
    ru = gu + _mv(B.swapaxes(-1, -2), it.lmbda[1:])
    ru += _mv(Eu.swapaxes(-1, -2), it.beta) + _mv(Gu.swapaxes(-1, -2), li)
    return Derivatives(gx, gu, A, B, Ex, Eu, Gx, Gu, Gf, E0, rx, ru)

  def system(self, it, ev, d, mu, params, bounds):
    """Eliminate slack-equality duals into the existing primal-dual Newton API."""
    m = self.model
    lower, upper, lf, uf = bounds
    damp, dampf = damping(lower, upper, mu, self.options), damping(
        lf, uf, mu, self.options
    )
    Hxx, Hxu, Huu = m.lagrangian_hessian(
        it.xs,
        it.us,
        it.lmbda[0],
        it.lmbda[1:],
        it.beta,
        it.y,
        jp.zeros(0),
        it.yf,
        1.0,
        params,
    )
    # Eliminate the independent g-s multipliers without changing the existing
    # Newton solver: dy = dzu - dzl + r_s, r_s = -y-zl+zu+damping.
    rs, rsf = (
        -it.y - it.z_l + it.z_u + damp,
        -it.yf - it.zf_l + it.zf_u + dampf,
    )
    rx = d.rx.at[:-1].add(_mv(d.Gx.swapaxes(-1, -2), rs))
    rx = rx.at[-1].add(d.Gf.T @ rsf)
    ru = d.ru + _mv(d.Gu.swapaxes(-1, -2), rs)
    cl, cu = complementarity(it.s, it.z_l, it.z_u, lower, upper, mu)
    cfl, cfu = complementarity(it.sf, it.zf_l, it.zf_u, lf, uf, mu)
    return pd.IpPrimalDualSystem.create(
        Hxx,
        Hxu,
        Huu,
        d.A,
        d.B,
        d.Ex,
        d.Eu,
        d.E0,
        d.Gx,
        d.Gu,
        lower,
        upper,
        d.Gf,
        lf,
        uf,
        it.s,
        it.sf,
        it.z_l,
        it.z_u,
        it.zf_l,
        it.zf_u,
        rx,
        ru,
        ev.dynamics,
        ev.equality,
        ev.inequality,
        cl,
        cu,
        ev.terminal,
        cfl,
        cfu,
        ev.initial,
    )
