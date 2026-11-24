"""Flip test for table_contacts: each free object is turned upside down (180 deg about world x), raised 5 cm,
and released (zero controls, true law).  Reports how deep its lowest geom point (all non-helper geoms, group 0)
ends below the table top.  usage: BENCH_TABLE=legacy|plane check_flip.py"""
import os, sys
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, mujoco
import bench_tasks as T, simtools as S

def lowest(m, d, b):
  z = np.inf
  for g in range(m.ngeom):
    if m.geom_bodyid[g] != b or m.geom_group[g] != 0: continue
    R = d.geom_xmat[g].reshape(3, 3); s = m.geom_size[g]; t = m.geom_type[g]
    if t == 6: ext = np.abs(R[2]) @ s
    elif t == 2: ext = s[0]
    elif t == 3: ext = s[0] + abs(R[2, 2]) * s[1]
    elif t == 5: ext = abs(R[2, 2]) * s[1] + s[0] * np.sqrt(max(0.0, 1 - R[2, 2] ** 2))
    else: continue
    z = min(z, d.geom_xpos[g][2] - ext)
  return z

for name in ['GripperCubeLift', 'GripperPegInsert', 'GripperLegTwist', 'GripperDrawer', 'AllegroHammer', 'AllegroScrewdriver',
             'AllegroSpatulaFlip', 'AllegroPenBoard', 'AllegroBrushSweep', 'AllegroEraserWipe']:
  t = T.TASKS[name](0); m = t.model
  x0 = np.array(t.x_0, float)
  fj = [j for j in range(m.njnt) if m.jnt_type[j] == 0]
  for j in fj:
    a = m.jnt_qposadr[j]
    q = x0[a + 3:a + 7]; qf = np.zeros(4); mujoco.mju_mulQuat(qf, np.array([0.0, 1.0, 0, 0]), q)
    x0[a + 2] += 0.05; x0[a + 3:a + 7] = qf
  X = S.sim_true(t, np.zeros((t.horizon, m.nu)), x0=x0)
  d = mujoco.MjData(m); d.qpos[:] = X[-1, :m.nq]; mujoco.mj_kinematics(m, d)
  low = {m.body(m.jnt_bodyid[j]).name: lowest(m, d, m.jnt_bodyid[j]) for j in fj}
  print(f'{os.environ.get("BENCH_TABLE", "plane"):6s} {name:20s} flipped, settled lowest point (mm): ' +
        ', '.join(f'{k} {1000 * v:.1f}' for k, v in low.items()), flush=True)
