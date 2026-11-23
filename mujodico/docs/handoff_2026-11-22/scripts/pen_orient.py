"""Pen plans under the true law: final position error, orientation error (deg) to the goal quaternion, at start and end.
usage: pen_orient.py <plan.npz> [...]"""
import sys, numpy as np
H_ = '/home/gchou/PycharmProjects/mujodico-main-dev/mujodico/docs'
sys.path.insert(0, H_ + '/handoff_2026-11-22/scripts'); sys.path.insert(0, H_ + '/handoff_2026-11-14/followup/scripts')
from mujodico.examples.benchmark_ippas import make_task
from mujodico.examples.methods import IppasLiftedMultipleShooting
import hard_tasks, extra_tasks  # noqa
task = make_task('AllegroPen', 0); tm = IppasLiftedMultipleShooting(task)
g = np.asarray(task.x_goal)
def ang(q, p):
    q = q / np.linalg.norm(q); p = p / np.linalg.norm(p)
    return np.degrees(2 * np.arccos(min(1.0, abs(float(q @ p)))))
for f in sys.argv[1:]:
    X = np.asarray(tm.rollout(task.x_0, np.load(f)['us'], 0.0)[0])
    print(f.split('/')[-1], 'pos err %.1f mm' % (1e3 * np.linalg.norm(X[-1, 16:19] - g[16:19])),
          'orient err start %.1f deg end %.1f deg' % (ang(X[0, 19:23], g[19:23]), ang(X[-1, 19:23], g[19:23])))
