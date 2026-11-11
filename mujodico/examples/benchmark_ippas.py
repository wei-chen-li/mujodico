"""Benchmark IP-PAS on the Table II tasks.

Example::

  python -m mujodico.examples.benchmark_ippas --tasks PushBox2d \\
      --seeds 0:200 --iterations 150 --out /tmp/ippas.jsonl

Success is a final goal error (terminal state cost) below 0.01, as in the
paper.  The first solve of each task includes compilation ("cold"); all later
seeds reuse the compiled program ("warm").  ``--options`` takes a JSON dict of
:class:`IpPasOptions` / method keyword overrides.
"""
import json
import time

from absl import app
from absl import flags
import numpy as np

from mujodico.examples.methods import IppasLiftedMultipleShooting
from mujodico.examples.methods import util
from pathlib import Path
import tempfile

import mujoco

from mujodico.examples.tasks import AllegroBall
from mujodico.examples.tasks import PushSphere2d
from mujodico.examples.tasks import PushSquareKnob
from mujodico.examples.tasks import allegro_ball as _allegro_ball
from mujodico.examples.tasks import parse_task_name
from mujodico.examples.tasks import Task

_TASKS = flags.DEFINE_list('tasks', ['PushBox2d'], 'Task names.')
_SEEDS = flags.DEFINE_string('seeds', '0:20', 'Seed range start:stop.')
_ITERS = flags.DEFINE_integer('iterations', 150, 'Iteration cap.')
_OPTIONS = flags.DEFINE_string('options', '{}', 'JSON solver overrides.')
_OUT = flags.DEFINE_string('out', '', 'Append per-seed JSON lines here.')


def _paper_hand_model():
  """Allegro scene with the hand collision model used for the paper."""
  models = Path(_allegro_ball.__file__).resolve().parent / 'models' / 'wonik_allegro'
  xml = _allegro_ball._XML.replace('right_hand.xml', 'right_hand_paper.xml')
  with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', dir=models,
                                   delete=False) as f:
    f.write(xml)
  try:
    return mujoco.MjModel.from_xml_path(f.name)
  finally:
    Path(f.name).unlink()


class AllegroBallTable(AllegroBall):
  """Sphere in-hand task of Table II: 20 steps, goal rotation in +-30 deg,
  and the paper's hand collision geometry."""

  def __init__(self, seed=0):
    super().__init__(seed)
    rng = np.random.default_rng(seed)
    rng.uniform([-0.03, -0.03, 0.0], [0.03, 0.03, 0.0])
    angle = np.deg2rad(rng.uniform(-30, 30))
    x_goal = self.x_goal.copy()
    x_goal[19:23] = [np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)]
    object.__setattr__(self, 'horizon', 20)
    object.__setattr__(self, 'x_goal', x_goal)
    object.__setattr__(self, 'model', _paper_hand_model())


class AllegroBallTablePlainCost(AllegroBallTable):
  """AllegroBallTable with the paper benchmark's plain quadratic cost on the
  quaternion coordinates (no quaternion-specific cost term)."""

  @property
  def quaternion_w_idxs(self):
    return []


class AllegroBallTableDevHand(AllegroBall):
  """As AllegroBallTable but with the current (dev) hand model."""

  def __init__(self, seed=0):
    super().__init__(seed)
    rng = np.random.default_rng(seed)
    rng.uniform([-0.03, -0.03, 0.0], [0.03, 0.03, 0.0])
    angle = np.deg2rad(rng.uniform(-30, 30))
    x_goal = self.x_goal.copy()
    x_goal[19:23] = [np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)]
    object.__setattr__(self, 'horizon', 20)
    object.__setattr__(self, 'x_goal', x_goal)


class PushSquareKnobTable(PushSquareKnob):
  """Square knob of Table II: initial angle uniform in +-1 revolution."""

  def __init__(self, seed=0):
    super().__init__(seed)
    rng = np.random.default_rng(seed)
    x_0 = self.x_0.copy()
    x_0[2] = rng.uniform(-2 * np.pi, 2 * np.pi)
    object.__setattr__(self, 'x_0', x_0)


class PushSphere2dTable(PushSphere2d):
  """Push circle of Table II: contact smoothing 0.01 (dev uses 0.02)."""

  def __init__(self, seed=0):
    super().__init__(seed)
    object.__setattr__(self, 'contact_smoothing', 0.01)


_PEN_SEP15_XML = """
<mujoco model="right_allegro_hand scene">
  <option timestep="0.05"  />
  <option integrator="implicitfast" />
  <option iterations="20" tolerance="1e-5" />
  <include file="right_hand2_sep15.xml"/>
  <worldbody>
    <body name="object" pos="-0.025 0 0.026">
      <freejoint/>
      <geom type="capsule" size="0.015" fromto="0 -0.08 0 0 0.08 0" rgba="0.5 0.7 0.5 1" friction="0.7" density="300"/>
      <geom name="angle_indicator" type="sphere" size="0.005" pos="0 0 0.011" rgba="0 0 0 1"
            contype="0" conaffinity="0" />
    </body>
  </worldbody>
</mujoco>
"""


class AllegroPenSep15(Task):
  """Original pen task (repo commit 4d95342, 2026-09-15): 15 steps of 0.05 s,
  goal = random x translation in +-1 cm, no rotation."""

  HORIZON = 15

  def __init__(self, seed=0):
    models = Path(_allegro_ball.__file__).resolve().parent / 'models' / 'wonik_allegro'
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', dir=models,
                                     delete=False) as f:
      f.write(_PEN_SEP15_XML)
    try:
      model = mujoco.MjModel.from_xml_path(f.name)
    finally:
      Path(f.name).unlink()
    data = mujoco.MjData(model)
    x_0 = np.concatenate((data.qpos, data.qvel))
    x_0[0:4] = np.deg2rad([0, 70, 80, 60])
    x_0[4:8] = np.deg2rad([0, 40, 60, 30])
    x_0[8:12] = np.deg2rad([0, 70, 80, 60])
    x_0[12:16] = np.deg2rad([50, 110, 80, 60])
    rng = np.random.default_rng(seed)
    x_goal = x_0.copy()
    x_goal[16:19] += rng.uniform([-0.01, 0.0, 0.0], [0.01, 0.0, 0.0])
    x_goal[19:23] = [1.0, 0.0, 0.0, 0.0]
    Q = np.diag([0.0] * 16 + [100] * 3 + [0.01] * 4 + [0.0] * 22)
    super().__init__(model=model, horizon=self.HORIZON, x_0=x_0, x_goal=x_goal,
                     Q=Q, Qf=Q * 40.0, u_ref=x_0[:16], R=np.eye(16) * 0.01,
                     contact_smoothing=0.002)


class AllegroPenSep15H20(AllegroPenSep15):
  """The original pen task with the 20-step horizon stated for Table II."""
  HORIZON = 20


_EXTRA_TASKS = {c.__name__: c for c in (
    AllegroPenSep15, AllegroPenSep15H20,
    AllegroBallTable, AllegroBallTablePlainCost, AllegroBallTableDevHand,
    PushSquareKnobTable,
    PushSphere2dTable)}


def make_task(name, seed):
  if name in _EXTRA_TASKS:
    return _EXTRA_TASKS[name](seed)
  return parse_task_name(name)(seed)


def run(task_name, seeds, iterations, options, out=None):
  method = None
  rows = []
  for seed in seeds:
    task = make_task(task_name, seed)
    if method is None:
      method = IppasLiftedMultipleShooting(task, max_iter=iterations,
                                           **options)
    else:
      method.update(task)
    tic = time.perf_counter()
    traj = method.solve({'xs': task.xs_init, 'us': task.us_init},
                        iterations=iterations, verbose=False)
    dt = time.perf_counter() - tic
    ge = float(util.state_cost(task.Q, task.x_goal, traj['xs'][-1],
                               task.quaternion_w_idxs))
    row = dict(task=task_name, seed=seed, goal_error=ge, time=dt,
               status=traj['status'], iterations=traj['iterations'],
               restorations=traj['restorations'], options=options,
               max_iter=iterations)
    rows.append(row)
    print(json.dumps(row), flush=True)
    if out:
      with open(out, 'a') as f:
        f.write(json.dumps(row) + '\n')
  return rows


def summarize(rows):
  ge = np.array([r['goal_error'] for r in rows])
  t = np.array([r['time'] for r in rows])
  warm = t[1:] if t.size > 1 else t
  return (f"{rows[0]['task']}: n={len(rows)} success={np.mean(ge < 0.01)*100:.1f}% "
          f"goal={ge.mean():.4f}+-{ge.std():.4f} "
          f"warm={warm.mean():.2f}+-{warm.std():.2f}s cold={t[0]:.1f}s")


def main(argv):
  del argv
  a, b = (int(v) for v in _SEEDS.value.split(':'))
  options = json.loads(_OPTIONS.value)
  summaries = []
  for name in _TASKS.value:
    rows = run(name, range(a, b), _ITERS.value, options, _OUT.value or None)
    summaries.append(summarize(rows))
  print('\n'.join(summaries))


if __name__ == '__main__':
  app.run(main)
