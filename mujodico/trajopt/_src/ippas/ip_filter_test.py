# Tests/fixtures adapted from FATROP (BSD-2-Clause).
# Copyright (c) Lander Vanroye, KU Leuven. See NOTICE.md.
"""Ports of FATROP unittest/ip_algorithm/ip_filter_test.cpp plus margin tests."""

from absl.testing import absltest
import jax
import jax.numpy as jp

from mujodico.trajopt._src.ippas import ip_filter as g
from mujodico.trajopt._src.ippas import ip_linesearch
from mujodico.trajopt._src.ippas import Options


class IpFilterTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.enterContext(jax.enable_x64())
    self.options = Options()

  def test_default_constructor(self):
    self.assertEqual(
        int(g.empty_filter(10, jp.array(0.0), self.options).count), 0
    )

  def test_reset(self):
    state = g.empty_filter(10, jp.array(0.0), self.options)
    self.assertEqual(int(g.reset(state).count), 0)
    state = g.augment(state, jp.array(0.0), jp.array(1.0))
    state = jax.jit(g.reset)(state)
    self.assertEqual(int(state.count), 0)
    self.assertTrue(
        bool(g.filter_acceptable(state, jp.array(2.0), jp.array(2.0)))
    )

  def test_reserve(self):
    state = g.empty_filter(10, jp.array(0.0), self.options)
    self.assertEqual(state.theta.shape, (10,))
    self.assertEqual(state.phi.shape, (10,))
    state = jax.jit(
        lambda f: jax.lax.fori_loop(
            0,
            10,
            lambda i, f: g.augment(
                f, i.astype(jp.float64), i.astype(jp.float64)
            ),
            f,
        )
    )(state)
    self.assertEqual(int(state.count), 10)

  def test_is_acceptable_empty_filter(self):
    state = g.empty_filter(10, jp.array(0.0), self.options)
    self.assertTrue(
        bool(g.filter_acceptable(state, jp.array(0.0), jp.array(1.0)))
    )

  def test_augment_and_is_acceptable(self):
    state = g.empty_filter(10, jp.array(0.0), self.options)
    # Upstream data is ordered (objective, violation); our arrays take theta,phi.
    state = g.augment(state, jp.array(0.0), jp.array(1.0))
    self.assertEqual(int(state.count), 1)
    self.assertTrue(
        bool(g.filter_acceptable(state, jp.array(0.0), jp.array(1.0)))
    )
    self.assertTrue(
        bool(g.filter_acceptable(state, jp.array(0.5), jp.array(0.5)))
    )
    state = g.augment(state, jp.array(0.5), jp.array(0.5))
    self.assertEqual(int(state.count), 2)
    self.assertFalse(
        bool(g.filter_acceptable(state, jp.array(2.0), jp.array(2.0)))
    )

  def test_size(self):
    state = g.empty_filter(10, jp.array(0.0), self.options)
    self.assertEqual(int(state.count), 0)
    state = g.augment(state, jp.array(0.0), jp.array(1.0))
    self.assertEqual(int(state.count), 1)
    state = g.augment(state, jp.array(0.5), jp.array(0.5))
    self.assertEqual(int(state.count), 2)

  def test_filter_shift_and_strict_dominance(self):
    o = self.options
    f = g.empty_filter(4, jp.array(1.0), o)
    f = g.augment_filter(f, jp.array(1.0), jp.array(2.0), o)
    self.assertAlmostEqual(float(f.theta[0]), 1 - o.gamma_theta)
    self.assertAlmostEqual(float(f.phi[0]), 2 - o.gamma_phi)
    # Equal corners are accepted by the filter (FATROP uses strict dominance).
    accept = lambda theta, phi: ip_linesearch.acceptable(
        jp.array(phi),
        jp.array(theta),
        jp.array(True),
        jp.array(100.0),
        jp.array(10.0),
        jp.array(0.0),
        jp.array(1.0),
        f,
        o,
    )
    self.assertTrue(bool(accept(f.theta[0], 3.0)))
    self.assertTrue(bool(accept(3.0, f.phi[0])))
    self.assertFalse(bool(accept(2.0, 3.0)))


if __name__ == '__main__':
  absltest.main()
