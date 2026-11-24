"""Verify table_contacts: per task, legacy vs plane models.
  pairs: non-table pairs must be identical; lists the table pairs and the MJX contact count
  rest : zero-control true-law rollout from x_0 (start-relative servos hold the pose): max object drift
  witness: scripted witness metric under the true law (seeds 0-2), legacy vs plane
usage: JAX_PLATFORMS=cpu check_table.py [Task,Task,...] [--no-witness]"""
import os, sys, importlib
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, mujoco
from mujodico.mjx._src import collision_driver as CD
from mujodico.mjx._src.types import GeomType

TASKS = ['GripperCubeStack', 'GripperCubeLift', 'GripperPegInsert', 'GripperLegTwist', 'GripperDrawer', 'AllegroHammer',
         'AllegroScrewdriver', 'AllegroSpatulaFlip', 'AllegroPenBoard', 'AllegroBrushSweep', 'AllegroEraserWipe']
ONLY = sys.argv[1].split(',') if len(sys.argv) > 1 and not sys.argv[1].startswith('--') else TASKS
WIT = '--no-witness' not in sys.argv


def pairs(m):
  out = {}
  for g1 in range(m.ngeom):
    for g2 in range(g1 + 1, m.ngeom):
      b1, b2 = m.geom_bodyid[g1], m.geom_bodyid[g2]
      if m.body_weldid[b1] == m.body_weldid[b2]: continue
      if not ((m.geom_contype[g1] & m.geom_conaffinity[g2]) or (m.geom_contype[g2] & m.geom_conaffinity[g1])): continue
      if any(m.exclude_signature[i] in ((b1 << 16) + b2, (b2 << 16) + b1) for i in range(m.nexclude)): continue
      w1, w2 = m.body_weldid[b1], m.body_weldid[b2]
      if w1 and w2 and (m.body_weldid[m.body_parentid[w1]] == w2 or m.body_weldid[m.body_parentid[w2]] == w1): continue
      t1, t2 = GeomType(m.geom_type[g1]), GeomType(m.geom_type[g2])
      if t1 > t2: t1, t2 = t2, t1
      fn = CD._COLLISION_FUNC.get((t1, t2))
      key = tuple(sorted((m.geom(g1).name or f'#{g1}', m.geom(g2).name or f'#{g2}')))
      out[key] = fn.ncon if fn else 0
  return out




def free_pos(m, x):
  return np.concatenate([x[m.jnt_qposadr[j]:m.jnt_qposadr[j] + 3] for j in range(m.njnt) if m.jnt_type[j] == 0])


import json, subprocess
if '--mode' in sys.argv:
  mode = sys.argv[sys.argv.index('--mode') + 1]
  os.environ['BENCH_TABLE'] = mode
  import bench_tasks as T, witnesses as Wn, simtools as S
  out = {}
  for name in ONLY:
    r = {}
    t = T.TASKS[name](0); m = t.model
    r['pairs'] = [[list(k), v] for k, v in pairs(m).items()]
    X = S.sim_true(t, np.zeros((t.horizon, m.nu)))
    r['rest'] = float(np.max(np.abs(free_pos(m, X[-1]) - free_pos(m, X[0])))) if any(m.jnt_type == 0) else 0.0
    r['H'] = t.horizon
    if WIT:
      mets = []
      for seed in range(3):
        ts = T.TASKS[name](seed)
        us = Wn.WITNESS[name](ts) if name in Wn.WITNESS else ts.witness()
        mets.append(S.metric(ts, S.sim_true(ts, us)[-1]))
      r['wit'] = mets
    out[name] = r
    print('done', mode, name, file=sys.stderr, flush=True)
  print('JSON' + json.dumps(out))
  sys.exit(0)

res = {}
for mode in ('legacy', 'plane'):
  o = subprocess.run([sys.executable, __file__] + sys.argv[1:] + ['--mode', mode], capture_output=True, text=True,
                     env=dict(os.environ, BENCH_TABLE=mode))
  line = [l for l in o.stdout.splitlines() if l.startswith('JSON')]
  if not line:
    print(o.stderr[-3000:]); sys.exit(1)
  for name, r in json.loads(line[0][4:]).items():
    d = res.setdefault(name, {})
    d[mode + '_pairs'] = {tuple(k): v for k, v in r['pairs']}; d[mode + '_rest'] = r['rest']; d['H'] = r['H']
    if WIT: d[mode + '_wit'] = r['wit']

for name in ONLY:
  r = res[name]; L, P = r['legacy_pairs'], r['plane_pairs']
  nt = lambda d: {k: v for k, v in d.items() if 'table' not in k}
  same = nt(L) == nt(P)
  tab = sorted(f'{a if b == "table" else b}:{v}' for (a, b), v in P.items() if 'table' in (a, b))
  print(f'\n{name}: contacts {sum(L.values())} -> {sum(P.values())}; non-table pairs unchanged: {same}')
  if not same:
    print('   only legacy:', set(nt(L)) - set(nt(P)), ' only plane:', set(nt(P)) - set(nt(L)))
  print('   table pairs:', ' '.join(tab))
  print(f'   rest drift (zero controls, {r["H"]} steps): legacy {r["legacy_rest"]*1000:.2f} mm, plane {r["plane_rest"]*1000:.2f} mm')
  if WIT:
    print('   witness metric legacy', np.round(r['legacy_wit'], 4), ' plane', np.round(r['plane_wit'], 4))
