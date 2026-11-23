"""Side-by-side GIFs of representative final plans: left, the planning model's rollout of the saved controls (smoothed
contact law at the finest level sigma_min, as the run's diag.plan_goal_sigmin); right, the TRUE-law rollout of the same
controls (what the success test judges).  Translucent ghost: the goal object pose.  Representative = the solved seed with the median wall time among
the solved seeds of that task and label (not hand-picked); extra rows can be named explicitly.
usage: MUJOCO_GL=egl render_plans.py <out_dir>          (writes <out_dir>/<key>.gif and <out_dir>/manifest.json)"""
import os, sys, json
import numpy as np
H_ = '/home/gchou/PycharmProjects/mujodico-main-dev/mujodico/docs'
sys.path.insert(0, H_ + '/handoff_2026-11-22/scripts'); sys.path.insert(0, H_ + '/handoff_2026-11-14/followup/scripts')
import mujoco
from PIL import Image, ImageDraw, ImageFont
from mujodico.examples.benchmark_ippas import make_task
from mujodico.examples.methods import IppasLiftedMultipleShooting, util
import hard_tasks, extra_tasks  # noqa
import jax, jax.numpy as jp
jax.config.update('jax_enable_x64', True)
from reloc_lms import LMS_KW
from lfddp_v import LFDDP

R = H_ + '/handoff_2026-11-22/results/vcycle'
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)

# (task, label, rows file) -> median solved seed;  extras: (task, label, seed, rows file, note)
SETS = [('PushT2d', 'V14', 'final.jsonl'), ('PushT2dHard', 'V14', 'final.jsonl'), ('PushT2dHardLong', 'V14', 'final.jsonl'),
        ('AllegroPen', 'V14p*', 'final.jsonl'),
        ('AllegroJar', 'PV11', 'plfv_hard.jsonl'), ('AllegroJarC', 'PV11r', 'plfv_hard.jsonl'),
        ('FingersSphere2dRandomized', 'GC1', 'gate_gc.jsonl'), ('FingersSphere2dRandomizedHardLong', 'GC1', 'gate_gc.jsonl'),
        ('PushSphere2d', 'GC1', 'gate_gc.jsonl'), ('PushSphere2dHard', 'GC1', 'gate_gc.jsonl'),
        ('PushMultiSphere2d', 'GC1', 'gate_gc.jsonl'), ('PushBox2d', 'GC1', 'gate_gc.jsonl'),
        ('PushSquareKnobTable', 'GC1', 'gate_gc.jsonl'), ('AllegroBallTableHard', 'GC1', 'gate_gc.jsonl')]
EXTRAS = [('PushT2dHard', 'V14', 0, 'final.jsonl', 'lpec'), ('AllegroPen', 'V14p2', 0, 'final.jsonl', 'lpec'),
          ('PushT2dHard', 'V14', 4, 'final.jsonl', 'miss'), ('PushT2dHardLong', 'V14', 0, 'final.jsonl', 'miss')]
ONLY = sys.argv[2].split(',') if len(sys.argv) > 2 else None   # optional: render only these keys (manifest is merged)


def rows_of(f, task, label):
  out = []
  for l in open(f'{R}/{f}'):
    r = json.loads(l)
    if r['task'] == task and (r['label'] == label or (label.endswith('*') and r['label'].startswith(label[:-1])
                                                         and r['label'][len(label) - 1:].isdigit())):
      out.append(r)
  return out


def pick(task, label, f):
  rs = [r for r in rows_of(f, task, label) if r['true_ok']]
  rs.sort(key=lambda r: r['time'])
  return rs[(len(rs) - 1) // 2], len(rs), len(rows_of(f, task, label))


def subtree(m, roots):
  keep = set(roots)
  for b in range(m.nbody):
    a = b
    while a > 0:
      if a in keep:
        keep.add(b); break
      a = m.body_parentid[a]
  return keep


def render(task, panels, path, planar):
  xs = panels[-1][0]   # the camera fits the true rollout and the goal
  m = task.model; nq = m.nq
  W = 360
  m.vis.headlight.ambient[:] = 0.35; m.vis.headlight.diffuse[:] = 0.75   # brighter, flatter light for small GIFs
  r = mujoco.Renderer(m, W, W)
  d = mujoco.MjData(m); dg = mujoco.MjData(m)
  qw = np.diag(task.Q)[:nq]
  objq = [i for i in range(nq) if qw[i] > 0]
  objb = subtree(m, {int(m.jnt_bodyid[j]) for j in range(m.njnt) if any(
      m.jnt_qposadr[j] <= i < m.jnt_qposadr[j] + {0: 7, 1: 4, 2: 1, 3: 1}[int(m.jnt_type[j])] for i in objq)})
  # camera: fit every body over the trajectory and the goal pose
  cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
  qg = xs[0, :nq].copy(); qg[objq] = np.asarray(task.x_goal)[objq]
  pts = []
  for q in list(xs[::max(1, len(xs) // 40), :nq]) + [qg]:
    d.qpos[:] = q; mujoco.mj_kinematics(m, d)
    sel = np.array([m.geom_bodyid[g] > 0 for g in range(m.ngeom)])
    pts.append(d.geom_xpos[sel])
  pts = np.concatenate(pts); lo, hi = pts.min(0), pts.max(0)
  cam.lookat[:] = (lo + hi) / 2
  span = float(np.max(hi - lo)) / 2 + (0.12 if planar else 0.06)
  if planar:
    cam.distance = span / np.tan(np.deg2rad(m.vis.global_.fovy / 2)); cam.elevation = -90.0; cam.azimuth = 90.0
  else:
    cam.distance = 1.25 * span / np.tan(np.deg2rad(m.vis.global_.fovy / 2)); cam.elevation = -30.0; cam.azimuth = 135.0
  # ghost of the goal object pose (scene geoms of the object's subtree at the goal, drawn translucent)
  dg.qpos[:] = qg; mujoco.mj_forward(m, dg)
  r.update_scene(dg, cam)
  ghost = []
  for i in range(r.scene.ngeom):
    g = r.scene.geoms[i]
    if g.objtype == mujoco.mjtObj.mjOBJ_GEOM and m.geom_bodyid[g.objid] in objb:
      ghost.append((int(g.type), g.size.copy(), g.pos.copy(), g.mat.copy(), int(g.dataid)))
  dt = m.opt.timestep
  N = len(xs) - 1
  stride = max(1, int(np.ceil(N / 110)))
  try:
    font = ImageFont.load_default(size=15)
  except TypeError:
    font = ImageFont.load_default()
  frames = []
  ks = list(range(0, N + 1, stride))
  if ks[-1] != N:
    ks.append(N)
  for k in ks:
    row = []
    for xp, ttl in panels:
      q = xp[min(k, len(xp) - 1), :nq]
      d.qpos[:] = q if np.all(np.isfinite(q)) else xp[0, :nq]; mujoco.mj_forward(m, d)
      r.update_scene(d, cam)
      for (ty, sz, pos, mat, did) in ghost:
        if r.scene.ngeom >= r.scene.maxgeom:
          break
        g = r.scene.geoms[r.scene.ngeom]
        mujoco.mjv_initGeom(g, ty, sz, pos, mat.flatten(), np.array([0.15, 0.75, 0.45, 0.35], np.float32))
        g.dataid = did
        r.scene.ngeom += 1
      img = Image.fromarray(r.render())
      dr = ImageDraw.Draw(img)
      dr.rectangle([0, 0, W, 24], fill=(255, 255, 255))
      dr.text((8, 4), ttl, fill=(20, 20, 20), font=font)
      dr.text((8, W - 24), f't = {k * dt:4.2f} s', fill=(20, 20, 20), font=font, stroke_width=2, stroke_fill=(255, 255, 255))
      row.append(np.asarray(img))
    sep = np.full((W, 4, 3), 255, np.uint8)
    frames.append(Image.fromarray(np.concatenate([row[0], sep, row[1]], 1)))
  dur = int(round(1000 * dt * stride))
  pal = [f.convert('P', palette=Image.ADAPTIVE, colors=96) for f in frames]
  durs = [dur] * (len(pal) - 1) + [1500]
  pal[0].save(path, save_all=True, append_images=pal[1:], duration=durs, loop=0, optimize=True)
  frames[-1].save(path.replace('.gif', '_final.png'))
  r.close()
  return len(pal)


man = []
todo = [(t, *pick(t, l, f), f, None) for t, l, f in SETS]
for t, l, s, f, note in EXTRAS:
  rr = [r for r in rows_of(f, t, l) if r['seed'] == s][0]
  todo.append((t, rr, None, None, f, note))
for task_name, row, nsol, ntot, f, note in todo:
  seed, label = row['seed'], row['label']
  if ONLY and f'{task_name}_{label}_{seed}' not in ONLY:
    continue
  task = make_task(task_name, seed)
  us = np.load(f"{R}/plans/{label}_{task_name}_{seed}.npz")['us']
  truem = IppasLiftedMultipleShooting(task)
  xt, _ = truem.rollout(task.x_0, np.asarray(us), 0.0); xt = np.asarray(xt)
  Qm = getattr(task, 'metric_Q', task.Q)
  goal = float(util.state_cost(Qm, task.x_goal, xt[-1], task.quaternion_w_idxs))
  key = f'{task_name}_{label}_{seed}'
  planar = '2d' in task_name.lower()
  lms = IppasLiftedMultipleShooting(task, smoothing_schedule=[dict(smoothing=1.0, iters=10, hess=[1, 0, 0])], max_iter=10, **LMS_KW)
  S = LFDDP(task, lms)
  sig_min = row.get('cfg', {}).get('sig_min', 1e-3)
  xp = np.asarray(S._roll(jp.asarray(task.x_0), jp.asarray(us), jp.asarray(S.base * sig_min))[0])
  pgoal = float(util.state_cost(Qm, task.x_goal, xp[-1], task.quaternion_w_idxs)) if np.all(np.isfinite(xp[-1])) else float('inf')
  n = render(task, [(xp, f'plan model (smoothing {sig_min:g})'), (xt, 'true-law replay')], f'{OUT}/{key}.gif', planar)
  ent = dict(key=key, task=task_name, label=label, seed=seed, goal=goal, row_goal=row['true_goal'], plan_goal=pgoal,
             row_plan_goal=(row.get('diag') or {}).get('plan_goal_sigmin'), time=row['time'],
             cycles=row.get('cycles'), lpec=len(row.get('lpec') or []), solved_of=(nsol, ntot), note=note,
             horizon_s=task.horizon * task.model.opt.timestep, frames=n, bytes=os.path.getsize(f'{OUT}/{key}.gif'))
  man.append(ent)
  print(json.dumps(ent), flush=True)
if ONLY and os.path.exists(f'{OUT}/manifest.json'):
  old = {e['key']: e for e in json.load(open(f'{OUT}/manifest.json'))}
  old.update({e['key']: e for e in man}); man = list(old.values())
json.dump(man, open(f'{OUT}/manifest.json', 'w'), indent=1)
