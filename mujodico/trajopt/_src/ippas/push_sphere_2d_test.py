"""End-to-end regression for the migrated lifted-shooting example.

This deliberately checks the requested goal-error target separately from KKT
convergence: the cost-only IPPAS and FATROP runs can both exhaust 500 iterations.
Run on CPU with JAX float64. The optional native comparison requires cyfatrop.
"""

import importlib.util

from absl.testing import absltest
import jax
import numpy as np


class PushSphere2dTest(absltest.TestCase):

  def test_goal_error_against_fatrop(self):
    with jax.enable_x64():
      try:
        from mujodico.examples.methods import util
        from mujodico.examples.methods.ippas_lifted_multiple_shooting import IppasLiftedMultipleShooting
        from mujodico.examples.tasks import PushSphere2d
      except ImportError as exc:
        self.skipTest(f'Optional example dependencies unavailable: {exc}')
      task = PushSphere2d(seed=0)
      smoothing = task.contact_smoothing
      method = IppasLiftedMultipleShooting(task)
      result = method.solve(
          {'xs': task.xs_init, 'us': task.us_init},
          iterations=500,
          verbose=False,
      )
      self.assertEqual(task.contact_smoothing, smoothing)
      self.assertTrue(np.all(np.isfinite(result['xs'])))
      self.assertTrue(np.all(np.isfinite(result['cfs'])))
      self.assertGreater(float(np.linalg.norm(result['cfs'])), 0.1)
      goal = float(
          util.state_cost(
              task.Q, task.x_goal, result['xs'][-1], task.quaternion_w_idxs
          )
      )
      self.assertLess(goal, 1e-4)
      print(
          f'IPPAS PushSphere2d: goal_error={goal:.10g},'
          f' status={int(method.info["status"])},'
          f' primal_error={float(method.info["primal_error"]):.6g}',
          flush=True,
      )
      if importlib.util.find_spec('cyfatrop') is not None:
        from mujodico.examples.methods.fatrop_lifted_multiple_shooting import FatropLiftedMultipleShooting

        reference = FatropLiftedMultipleShooting(PushSphere2d(seed=0)).solve(
            {'xs': task.xs_init, 'us': task.us_init},
            iterations=500,
            verbose=False,
        )
        native_goal = float(
            util.state_cost(
                task.Q, task.x_goal, reference['xs'][-1], task.quaternion_w_idxs
            )
        )
        self.assertLess(goal, max(1e-4, 2 * native_goal))
        print(f'FATROP PushSphere2d: goal_error={native_goal:.10g}', flush=True)


if __name__ == '__main__':
  absltest.main()
