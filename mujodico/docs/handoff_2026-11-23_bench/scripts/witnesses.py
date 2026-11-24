"""Scripted open-loop witnesses: controls that solve each benchmark task under the TRUE law.

A witness certifies physical solvability (reachability, grasp strength, friction, actuator
limits, horizon).  It is never given to a solver.

usage: python witnesses.py <Task> [a:b] [--law c|true|both] [--save]
"""
import sys, os, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_tasks as T
from bench_tasks import _qpos_adr
from simtools import keyframes, report, sim_true, sim_c, metric

WITNESS = {}


def witness(fn):
  WITNESS[fn.__name__] = fn
  return fn


def _free(t, joint):
  a = _qpos_adr(t.model, joint)
  return t.x_0[a:a + 3].copy(), t.x_0[a + 3:a + 7].copy()


def _yaw(q):
  return 2 * np.arctan2(q[3], q[0])


def _wrap_near(a, ref):
  """a + k*pi/2 closest to ref (cube symmetry)."""
  k = np.round((ref - a) / (np.pi / 2))
  return a + k * np.pi / 2


@witness
def GripperCubeStack(t):
  pa, qa = _free(t, 'cubeA'); pb, qb = _free(t, 'cubeB')
  ya_cube = _yaw(qa)
  yw0 = _wrap_near(ya_cube, 0.0)              # wrist yaw at grasp: the cube face pair nearest 0
  delta = ya_cube - yw0                       # cube yaw relative to the wrist (kept by the grasp)
  yb_goal = _yaw(t.x_goal[_qpos_adr(t.model, 'cubeA') + 3:][:4])
  yw_place = (yb_goal - delta + np.pi) % (2 * np.pi) - np.pi
  ya = yw0
  h = T.CUBE_H
  zg = h + 0.035          # pad middle sphere at cube mid-height
  zp = 3 * h + 0.035 + 0.003
  kf = [
      (0.8, dict(a_wx=pa[0], a_wy=pa[1], a_wz=0.12, a_wrz=ya)),
      (1.3, dict(a_wz=zg)),
      (1.6, dict(a_grip=0.0)),
      (2.1, dict(a_wz=0.16)),
      (3.0, dict(a_wx=pb[0], a_wy=pb[1], a_wrz=yw_place)),
      (3.5, dict(a_wz=zp)),
      (3.8, dict(a_grip=0.04)),
      (4.3, dict(a_wz=zp + 0.05)),
  ]
  return keyframes(t, kf)


@witness
def GripperCubeLift(t):
  pa, qa = _free(t, 'cubeA')
  ya = _wrap_near(_yaw(qa), 0.0)
  zg = T.CUBE_H + 0.035
  kf = [(0.8, dict(a_wx=pa[0], a_wy=pa[1], a_wz=0.12, a_wrz=ya)), (1.3, dict(a_wz=zg)),
        (1.6, dict(a_grip=0.0)), (2.2, dict(a_wz=zg + 0.08))]
  return keyframes(t, kf)


@witness
def GripperCubeLiftPregrasp(t):
  zg = T.CUBE_H + 0.035
  return keyframes(t, [(0.3, dict(a_grip=0.0)), (0.5, {}), (1.2, dict(a_wz=zg + 0.08))])


@witness
def GripperCubeLiftGrasped(t):
  zg = T.CUBE_H + 0.035
  return keyframes(t, [(0.2, {}), (1.0, dict(a_wz=zg + 0.08))])


for _n in ('GripperCubeLift', 'GripperCubeLiftPregrasp', 'GripperCubeLiftGrasped', 'GripperCubeStack'):
  WITNESS[_n + 'G'] = WITNESS[_n]
  for _tag in T.PAD_VARIANTS:
    WITNESS[_n + _tag] = WITNESS[_n]
WITNESS['GripperCubeLiftLong'] = WITNESS['GripperCubeLift']
WITNESS['GripperCubeStackLong'] = WITNESS['GripperCubeStack']
for _n in ('GripperCubeLift', 'GripperCubeLiftPregrasp', 'GripperCubeLiftGrasped'):
  WITNESS[_n + 'Still'] = WITNESS[_n + 'RunStill'] = WITNESS[_n]


def main(argv):
  name = argv[1]
  a, b = (int(v) for v in (argv[2] if len(argv) > 2 and ':' in argv[2] else '0:1').split(':'))
  law = argv[argv.index('--law') + 1] if '--law' in argv else 'both'
  which = {'c': ('c',), 'true': ('true',), 'both': ('c', 'true')}[law]
  rows = []
  for seed in range(a, b):
    t = T.TASKS[name](seed)
    us = WITNESS[name](t) if name in WITNESS else t.witness()
    print(f'{name} seed {seed}')
    tic = time.time()
    out = report(t, us, which)
    r = dict(task=name, seed=seed, **{f'metric_{w}': out[w][0] for w in out}, time=time.time() - tic)
    rows.append(r)
    if '--save' in argv:
      d = os.path.join(HERE, '..', 'witness'); os.makedirs(d, exist_ok=True)
      np.savez(os.path.join(d, f'{name}_s{seed}.npz'), us=us, **{f'xs_{w}': out[w][1] for w in out})
  ok = {w: sum(r[f'metric_{w}'] < 0.01 for r in rows) for w in which}
  print('solved', ok, '/', len(rows))
  return rows


if __name__ == '__main__':
  main(sys.argv)
