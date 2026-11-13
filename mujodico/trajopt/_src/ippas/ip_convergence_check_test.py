"""Termination boundaries; no dedicated counterpart exists in this FATROP checkout."""

from absl.testing import absltest
import jax
import jax.numpy as jp

from mujodico.trajopt._src.ippas import InitialGuess
from mujodico.trajopt._src.ippas import ip_convergence_check
from mujodico.trajopt._src.ippas import ip_initializer
from mujodico.trajopt._src.ippas import ip_iterate
from mujodico.trajopt._src.ippas import Options
from mujodico.trajopt._src.ippas import Status
from mujodico.trajopt._src.ippas.ip_nlp_orig import IpNlp
from mujodico.trajopt._src.ippas.test_util import model


class IpConvergenceCheckTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.enterContext(jax.enable_x64())
    self.options = Options()
    self.model = model(N=1)
    self.model.initialize(jp.array([0.0, 0.0]))
    self.nlp = IpNlp(self.model, self.options)
    self.bounds = self.nlp.bounds()
    self.it = ip_initializer.initialize(
        self.model,
        jp.array([0.0, 0.0]),
        InitialGuess(jp.zeros((2, 1)), jp.zeros((1, 1))),
        self.bounds,
        self.options,
    )
    self.ev = self.nlp.evaluate(self.it, jp.array([0.0, 0.0]))
    self.d = self.nlp.derivatives(self.it, jp.array([0.0, 0.0]))

  def _check(self, error, primal=0.0, count=0, iteration=0):
    metrics = ip_iterate.Metrics(
        *(jp.array(v) for v in (primal, error, 0.0, error, error))
    )
    check = jax.jit(
        lambda mm, c, i: ip_convergence_check.check(
            self.model,
            self.it,
            self.ev,
            self.d,
            mm,
            c,
            i,
            self.bounds,
            self.options,
        )
    )
    return int(
        check(metrics, jp.array(count, jp.int32), jp.array(iteration, jp.int32))
    )

  def test_success_has_priority_at_iteration_limit(self):
    self.assertEqual(
        self._check(
            self.options.tol,
            primal=self.options.constr_viol_tol,
            iteration=self.options.max_iter,
        ),
        Status.SUCCESS,
    )

  def test_acceptable_requires_consecutive_count(self):
    error = self.options.tol_acceptable
    self.assertEqual(
        self._check(error, count=self.options.acceptable_iter - 1),
        Status.RUNNING,
    )
    self.assertEqual(
        self._check(error, count=self.options.acceptable_iter),
        Status.ACCEPTABLE,
    )

  def test_original_bounds_must_meet_constraint_tolerance(self):
    self.assertEqual(
        self._check(0.0, primal=1.1 * self.options.constr_viol_tol),
        Status.RUNNING,
    )

  def test_iteration_limit(self):
    self.assertEqual(
        self._check(1.0, iteration=self.options.max_iter), Status.MAX_ITER
    )

  def test_nonfinite_error(self):
    self.assertEqual(self._check(jp.nan), Status.NUMERICAL_ERROR)


if __name__ == '__main__':
  absltest.main()
