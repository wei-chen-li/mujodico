# Tests/fixtures adapted from FATROP (BSD-2-Clause).
# Copyright (c) Lander Vanroye, KU Leuven. See NOTICE.md.
"""Ports of FATROP unittest/ocp/ip_search_dir_test.cpp and curvature/refinement regressions.

The upstream point-mass fixture's terminal equalities are omitted because the
unchanged IPPAS backend does not support them. The Newton/KKT assertions and
linearized complementarity checks are preserved.
"""

from dataclasses import replace
from unittest import mock

from absl.testing import absltest
import jax
import jax.numpy as jp
import numpy as np

from mujodico.trajopt._src.ippas import InitialGuess
from mujodico.trajopt._src.ippas import ip_initializer
from mujodico.trajopt._src.ippas import ip_iterate
from mujodico.trajopt._src.ippas import ip_primal_dual_system as pd
from mujodico.trajopt._src.ippas import ip_search_dir
from mujodico.trajopt._src.ippas import IppasSolver
from mujodico.trajopt._src.ippas import Options
from mujodico.trajopt._src.ippas import Status
from mujodico.trajopt._src.ippas.ip_nlp_orig import IpNlp
from mujodico.trajopt._src.ippas.ip_primal_dual_system_test import _random_problem_data
from mujodico.trajopt._src.ippas.test_util import fatrop_point_mass_model
from mujodico.trajopt._src.ippas.test_util import model as _model


class IpSearchDirTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.enterContext(jax.enable_x64())

  def test_negative_curvature_is_regularized(self):
    m = _model(
        N=1,
        stage_cost=lambda x, u, k, p: jp.sum((u * u - 1) ** 2),
        terminal_cost=lambda x, p: jp.array(0.0),
        gauss_newton=False,
    )
    s = IppasSolver(m)
    p = jp.array([0.0, 0.0])
    m.initialize(p)
    it = ip_initializer.initialize(
        m,
        p,
        InitialGuess(jp.zeros((2, 1)), jp.array([[0.2]])),
        s._nlp.bounds(),
        s.options,
    )
    ev, d = s._nlp.evaluate(it, p), s._nlp.derivatives(it, p)
    sys = s._nlp.system(it, ev, d, jp.array(0.1), p, s._nlp.bounds())
    _, good, delta = jax.jit(
        lambda sys, mu, delta: ip_search_dir.compute(sys, mu, delta, s.options)
    )(sys, jp.array(0.1), jp.array(0.0))
    self.assertTrue(bool(good))
    self.assertGreater(float(delta), 3.5)
    result = jax.jit(s.solve)(p, it.xs, it.us)
    self.assertEqual(int(result.status), Status.SUCCESS)
    np.testing.assert_allclose(result.us, 1.0, atol=1e-7)

  def _native_fixture(self):
    model, guess = fatrop_point_mass_model()
    # Remove optional one-sided barrier damping when checking original-NLP
    # stationarity, as in the native test's zero dual-infeasibility assertions.
    options = Options(kappa_d=0.0, initialize_duals=False)
    nlp = IpNlp(model, options)
    bounds = nlp.bounds()
    it = ip_initializer.initialize(model, None, guess, bounds, options)
    ev, d = nlp.evaluate(it, None), nlp.derivatives(it, None)
    system = nlp.system(it, ev, d, jp.array(1.0), None, bounds)
    return model, nlp, bounds, it, system, options

  def test_solve_linear_system(self):
    _, _, _, _, system, options = self._native_fixture()
    direction, good, _ = jax.jit(
        lambda p: ip_search_dir.compute(
            p, jp.array(1.0), jp.array(0.0), options
        )
    )(system)
    self.assertTrue(bool(good))
    self.assertTrue(
        bool(
            ip_iterate.all_finite(
                tuple(getattr(direction, n) for n in ip_search_dir._STEPS)
            )
        )
    )

  def test_update_iterate_and_check_infeasibility(self):
    model, nlp, bounds, it, system, options = self._native_fixture()
    direction, good, _ = jax.jit(
        lambda p: ip_search_dir.compute(
            p, jp.array(1.0), jp.array(0.0), options
        )
    )(system)
    self.assertTrue(bool(good))
    # Native test takes full primal and dual steps without line-search clipping.
    trial = replace(
        it,
        xs=model.retract_state(it.xs, direction.dx, 1.0, None),
        us=it.us + direction.du,
        s=it.s + direction.ds,
        sf=it.sf + direction.dsf,
        lmbda=it.lmbda + direction.dlmbda,
        beta=it.beta + direction.dbeta,
        y=direction.dz_u - direction.dz_l - it.z_l + it.z_u,
        yf=direction.dzf_u - direction.dzf_l - it.zf_l + it.zf_u,
        z_l=it.z_l + direction.dz_l,
        z_u=it.z_u + direction.dz_u,
        zf_l=it.zf_l + direction.dzf_l,
        zf_u=it.zf_u + direction.dzf_u,
    )
    ev, d = nlp.evaluate(trial, None), nlp.derivatives(trial, None)
    self.assertLess(float(ip_iterate.max_abs(d.rx, d.ru)), 1e-6)
    self.assertLess(
        float(
            ip_iterate.max_abs(
                -trial.y - trial.z_l + trial.z_u,
                -trial.yf - trial.zf_l + trial.zf_u,
            )
        ),
        1e-6,
    )
    self.assertLess(float(ip_iterate.max_abs(*ev[1:])), 1e-6)
    # Complementarity is nonlinear: check its linearization at the saved iterate.
    residuals = ip_search_dir.residuals(direction)
    for name in ('r_c_l', 'r_c_u', 'r_cf_l', 'r_cf_u'):
      self.assertLess(
          float(ip_iterate.max_abs(residuals[name])), 1e-6, msg=name
      )

  def test_refinement_recovers_perturbed_step_without_refactorization(self):
    data = _random_problem_data(23, 5, 3, 2, 1, 3, 2, 'mixed', 'pd')
    exact = jax.jit(pd.solve)(pd.IpPrimalDualSystem.create(**data))
    perturbed = replace(
        exact,
        dx=exact.dx + 0.03,
        dz_l=exact.dz_l + jp.where(jp.isfinite(exact.g_l), 0.1, 0.0),
    )
    with mock.patch.object(pd, 'solve', wraps=pd.solve) as solve:
      improved, good = jax.jit(lambda p: ip_search_dir.refine(p, Options()))(
          perturbed
      )
    self.assertTrue(bool(good))
    self.assertTrue(solve.called)
    self.assertIsNotNone(solve.call_args.args[0]._cache)
    for name in ip_search_dir._STEPS:
      np.testing.assert_allclose(
          getattr(improved, name), getattr(exact, name), atol=1e-8
      )
    for name, residual in ip_search_dir.residuals(improved).items():
      np.testing.assert_allclose(residual, 0, atol=1e-8, err_msg=name)


if __name__ == '__main__':
  absltest.main()
