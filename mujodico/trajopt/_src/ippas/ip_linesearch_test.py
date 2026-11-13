"""Focused IPPAS component regression tests."""

from dataclasses import replace

from absl.testing import absltest
import jax
import jax.numpy as jp
import numpy as np

from mujodico.trajopt._src.ippas import InitialGuess
from mujodico.trajopt._src.ippas import ip_filter as glob
from mujodico.trajopt._src.ippas import ip_initializer
from mujodico.trajopt._src.ippas import ip_iterate
from mujodico.trajopt._src.ippas import ip_linesearch
from mujodico.trajopt._src.ippas import ip_linesearch as g
from mujodico.trajopt._src.ippas import ip_search_dir
from mujodico.trajopt._src.ippas import Options
from mujodico.trajopt._src.ippas.ip_nlp_orig import IpNlp
from mujodico.trajopt._src.ippas.test_util import model as _model


class IpLinesearchTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.enterContext(jax.enable_x64())
    self.options = Options()

  def test_soc_overcomes_maratos_step(self):
    m = _model(
        N=1,
        nu=2,
        dynamics=lambda x, u, k, p: x,
        stage_cost=lambda x, u, k, p: 0.5 * jp.sum((u - 0.1) ** 2),
        terminal_cost=lambda x, p: jp.array(0.0),
        stage_equality_constraint=lambda x, u, k, p: jp.array(
            [jp.sum(u * u) - 1]
        ),
        gauss_newton=False,
    )
    p = jp.array([0.0, 0.0])
    m.initialize(p)
    options = Options()
    nlp = IpNlp(m, options)
    bounds = nlp.bounds()
    it = ip_initializer.initialize(
        m,
        p,
        InitialGuess(jp.zeros((2, 1)), jp.array([[0.8, 0.6]])),
        bounds,
        options,
    )
    it = replace(it, beta=jp.array([[(np.sqrt(0.02) - 1) / 2]]))
    ev, d = nlp.evaluate(it, p), nlp.derivatives(it, p)
    sys = nlp.system(it, ev, d, jp.array(0.1), p, bounds)
    direction, _, _ = ip_search_dir.compute(
        sys, jp.array(0.1), jp.array(0.0), options
    )
    first = ip_iterate.candidate(
        m,
        it,
        direction,
        jp.array(1.0),
        jp.array(1.0),
        jp.array(0.1),
        p,
        bounds,
        options,
    )
    self.assertGreater(
        float(nlp.evaluate(first, p).objective), float(ev.objective)
    )
    fs = glob.empty_filter(20, jp.array(0.0), options)
    step, _ = jax.jit(
        lambda: ip_linesearch.find_acceptable_trial_point(
            nlp, it, ev, d, direction, jp.array(0.1), p, bounds, fs, options
        )
    )()
    self.assertTrue(bool(step.accepted))
    self.assertEqual(
        int(step.count), 0
    )  # No backtracking: the SOC was accepted.
    self.assertLess(float(step.evaluation.objective), float(ev.objective))

  def test_switching_and_armijo_near_feasibility(self):
    o = self.options
    f = glob.empty_filter(4, jp.array(0.0), o)
    common = (
        jp.array(1.0),
        jp.array(0.0),
        jp.array(-1.0),
        jp.array(1.0),
        f,
        o,
    )
    self.assertTrue(
        bool(
            g.acceptable(jp.array(0.9), jp.array(0.0), jp.array(True), *common)
        )
    )
    self.assertFalse(
        bool(
            g.acceptable(jp.array(1.1), jp.array(0.0), jp.array(True), *common)
        )
    )
    self.assertGreater(
        float(g.minimum_step(jp.array(0.0), jp.array(-1.0), f, o)), 0
    )
    self.assertFalse(
        bool(
            g.acceptable(
                jp.array(jp.nan), jp.array(0.0), jp.array(True), *common
            )
        )
    )

  def test_filter_reset_after_repeated_rejections(self):
    state = glob.empty_filter(10, jp.array(1.0), self.options)
    state = glob.augment(state, jp.array(1.0), jp.array(1.0))
    trial = ip_linesearch.Search(
        None, None, jp.array(True), jp.array(0.5), jp.array(1), jp.array(True)
    )
    update = jax.jit(
        lambda f, n, r: ip_linesearch.update_filter_rejections(
            f, trial, n, r, self.options
        )
    )
    count, resets = jp.array(0, jp.int32), jp.array(0, jp.int32)
    for _ in range(self.options.filter_reset_trigger - 1):
      state, count, resets = update(state, count, resets)
      self.assertEqual(int(state.count), 1)
    state, count, resets = update(state, count, resets)
    self.assertEqual(int(state.count), 0)
    self.assertEqual(int(count), 0)
    self.assertEqual(int(resets), 1)
    disabled = replace(self.options, max_filter_resets=0)
    state = glob.augment(state, jp.array(1.0), jp.array(1.0))
    state, _, resets = ip_linesearch.update_filter_rejections(
        state, trial, jp.array(100, jp.int32), jp.array(0, jp.int32), disabled
    )
    self.assertEqual(int(state.count), 1)
    self.assertEqual(int(resets), 0)


if __name__ == '__main__':
  absltest.main()
