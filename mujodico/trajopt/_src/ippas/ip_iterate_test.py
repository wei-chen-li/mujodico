"""Focused IPPAS component regression tests."""

from dataclasses import replace

from absl.testing import absltest
import jax
import jax.numpy as jp
import numpy as np

from mujodico.trajopt._src.ippas import InitialGuess
from mujodico.trajopt._src.ippas import ip_initializer
from mujodico.trajopt._src.ippas import ip_iterate
from mujodico.trajopt._src.ippas import ip_search_dir
from mujodico.trajopt._src.ippas import Options
from mujodico.trajopt._src.ippas.ip_iterate import maximum_step
from mujodico.trajopt._src.ippas.ip_iterate import push_interior
from mujodico.trajopt._src.ippas.ip_nlp_orig import IpNlp
from mujodico.trajopt._src.ippas.test_util import model as _model


class IpIterateTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.enterContext(jax.enable_x64())
    self.options = Options()

  def test_independent_slack_equality_multipliers(self):
    m = _model(
        N=1,
        stage_inequality_constraint=lambda x, u, k, p: 1e3 * u,
        stage_inequality_bounds=(np.array([-1.0]), np.array([np.inf])),
    )
    p = jp.array([0.0, 2.0])
    m.initialize(p)
    options = Options()
    nlp = IpNlp(m, options)
    bounds = nlp.bounds()
    it = ip_initializer.initialize(
        m, p, InitialGuess(jp.zeros((2, 1)), jp.zeros((1, 1))), bounds, options
    )
    # A shortened primal step and a larger bound-dual step must leave y and
    # zu-zl distinct. The reduced Newton RHS still accounts for that residual.
    it = replace(it, y=it.y + 0.2)
    ev, d = nlp.evaluate(it, p), nlp.derivatives(it, p)
    sys = nlp.system(it, ev, d, jp.array(0.1), p, bounds)
    direction, good, _ = ip_search_dir.compute(
        sys, jp.array(0.1), jp.array(0.0), options
    )
    self.assertTrue(bool(good))
    rs = (
        -it.y
        - it.z_l
        + it.z_u
        + ip_iterate.damping(*bounds[:2], jp.array(0.1), options)
    )
    dy = direction.dz_u - direction.dz_l + rs
    control = (
        d.ru
        + jp.einsum('tij,tj->ti', sys.Huu, direction.du)
        + jp.einsum('tji,tj->ti', sys.B, direction.dlmbda[1:])
        + jp.einsum('tji,tj->ti', sys.Gu, dy)
    )
    np.testing.assert_allclose(control, 0.0, atol=1e-9)
    candidate = ip_iterate.candidate(
        m,
        it,
        direction,
        jp.array(0.1),
        jp.array(0.8),
        jp.array(0.1),
        p,
        bounds,
        options,
    )
    np.testing.assert_allclose(candidate.y, it.y + 0.1 * dy, atol=1e-12)
    self.assertGreater(
        float(jp.max(jp.abs(candidate.y - candidate.z_u + candidate.z_l))),
        1e-3,
    )

  def test_original_bound_interior_and_fraction_to_boundary(self):
    lower = jp.array([0.0, -jp.inf, -2.0, -jp.inf])
    upper = jp.array([1.0, 0.0, 2.0, jp.inf])
    s = push_interior(
        jp.array([-3.0, 2.0, 4.0, 5.0]), lower, upper, self.options
    )
    self.assertTrue(np.all(np.asarray(s) > np.asarray(lower)))
    self.assertTrue(np.all(np.asarray(s) < np.asarray(upper)))
    delta = jp.array([-2.0, 2.0, 0.0, 100.0])
    alpha = jax.jit(maximum_step)(s, delta, lower, upper, jp.array(0.99))
    trial = s + alpha * delta
    self.assertTrue(np.all(np.asarray(trial) > np.asarray(lower)))
    self.assertTrue(np.all(np.asarray(trial) < np.asarray(upper)))


if __name__ == '__main__':
  absltest.main()
