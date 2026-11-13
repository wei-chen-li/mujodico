# Tests/fixtures adapted from FATROP (BSD-2-Clause).
# Copyright (c) Lander Vanroye, KU Leuven. See NOTICE.md.
"""Ports of FATROP unittest/ip_algorithm/ip_initializer_test.cpp.

IPPAS takes explicit primal guesses, so the low/high cases pass those guesses
rather than ask an NLP callback to overwrite a stored primal state.
"""

from dataclasses import replace

from absl.testing import absltest
import jax
import jax.numpy as jp
import numpy as np

from mujodico.trajopt._src.ippas import ip_initializer
from mujodico.trajopt._src.ippas import ip_iterate
from mujodico.trajopt._src.ippas import Options
from mujodico.trajopt._src.ippas.ip_nlp_orig import IpNlp
from mujodico.trajopt._src.ippas.test_util import fatrop_point_mass_model


class IpInitializerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.enterContext(jax.enable_x64())
    self.model, self.guess = fatrop_point_mass_model()
    self.options = Options()
    self.nlp = IpNlp(self.model, self.options)
    self.bounds = self.nlp.bounds()

  def test_eq_mult_initializer(self):
    def initialize(xs, us):
      guess = replace(self.guess, xs=xs, us=us)
      it = ip_initializer.initialize(
          self.model, None, guess, self.bounds, self.options
      )
      ev, d = self.nlp.evaluate(it, None), self.nlp.derivatives(it, None)
      return ip_initializer.initialize_multipliers(
          self.nlp, it, ev, d, None, self.bounds, self.options
      )

    result = jax.jit(initialize)(self.guess.xs, self.guess.us)
    self.assertTrue(bool(ip_iterate.all_finite(result)))
    self.assertLessEqual(
        float(ip_iterate.max_abs(result.lmbda, result.beta)),
        self.options.lam_max,
    )

  def _check_bound_push(self, value):
    guess = replace(
        self.guess,
        xs=jp.full_like(self.guess.xs, value),
        us=jp.full_like(self.guess.us, value),
    )
    raw = self.model.stage_inequality_residual(guess.xs[:-1], guess.us, None)
    before = float(jp.sum(jp.abs(raw)))
    result = jax.jit(
        lambda xs, us: ip_initializer.initialize(
            self.model,
            None,
            replace(guess, xs=xs, us=us),
            self.bounds,
            self.options,
        )
    )(guess.xs, guess.us)
    lower, upper = self.bounds[:2]
    self.assertTrue(np.all(np.asarray(result.s) > np.asarray(lower)))
    self.assertTrue(np.all(np.asarray(result.s) < np.asarray(upper)))
    self.assertLess(
        float(jp.sum(jp.abs(self.nlp.evaluate(result, None).inequality))),
        before,
    )
    np.testing.assert_array_equal(result.us, guess.us)
    np.testing.assert_array_equal(result.z_l, jp.isfinite(lower))
    np.testing.assert_array_equal(result.z_u, jp.isfinite(upper))

  def test_bound_push_low(self):
    self._check_bound_push(-100.0)

  def test_bound_push_high(self):
    self._check_bound_push(100.0)


if __name__ == '__main__':
  absltest.main()
