"""GIFs of the scripted witnesses under the TRUE law (seed 0 by default), for checking physical plausibility.
Left panel: visual geometry. Right panel: collision geometry only (what the contact law sees; hidden helper spheres
made visible) with contact points (red). Green ghost: goal pose of the scored object(s).
usage: JAX_PLATFORMS=cpu MUJOCO_GL=egl render_witness.py <out_dir> [Task,Task,...] [--seed s]"""
import os, sys, json, copy
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mujoco
from PIL import Image, ImageDraw, ImageFont
import bench_tasks as T
import witnesses as Wn
from simtools import sim_true, metric

OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
ONLY = sys.argv[2].split(',') if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else None
SEED = int(sys.argv[sys.argv.index('--seed') + 1]) if '--seed' in sys.argv else 0
W = 340


def subtree(m, roots):
  keep = set(roots)
  for b in range(m.nbody):
    a = b
    while a > 0:
      if a in keep:
        keep.add(b); break
      a = m.body_parentid[a]
  return keep


def collision_model(m):
  m2 = copy.deepcopy(m)
  for g in range(m2.ngeom):
    coll = m2.geom_contype[g] or m2.geom_conaffinity[g]
    if coll:
      m2.geom_group[g] = 0
      c = m2.geom_rgba[g].copy()
      if m2.geom_matid[g] >= 0:
        c[:3] = m2.mat_rgba[m2.geom_matid[g], :3]; m2.geom_matid[g] = -1
      if c[3] < 0.5 or np.allclose(c, [1, 0, 0, 1]):   # hidden helpers / Allegro's red collision class
        c = np.array([0.95, 0.75, 0.2, 0.75]) if c[3] < 0.5 else np.array([0.6, 0.6, 0.65, 0.8])
      m2.geom_rgba[g] = c
    else:
      m2.geom_group[g] = 5                             # hidden in this panel
  return m2


def render(task, xs, path, title):
  m = task.model; nq = m.nq
  for mm in (m,):
    mm.vis.headlight.ambient[:] = 0.35; mm.vis.headlight.diffuse[:] = 0.75
  m2 = collision_model(m)
  m2.vis.map.znear = m.vis.map.znear
  r1 = mujoco.Renderer(m, W, W); r2 = mujoco.Renderer(m2, W, W)
  d1 = mujoco.MjData(m); d2 = mujoco.MjData(m2); dg = mujoco.MjData(m)
  qw = np.diag(task.Q)[:nq]
  objq = [i for i in range(nq) if qw[i] > 0]
  nadr = {0: 7, 1: 4, 2: 1, 3: 1}
  objb = subtree(m, {int(m.jnt_bodyid[j]) for j in range(m.njnt) if any(
      m.jnt_qposadr[j] <= i < m.jnt_qposadr[j] + nadr[int(m.jnt_type[j])] for i in objq)})
  # camera: fit the non-hand bodies (objects, tools, fixtures) over the trajectory + goal, plus the tool path
  hand = subtree(m, {m.body('wrist').id})
  cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
  qg = xs[0, :nq].copy(); qg[objq] = np.asarray(task.x_goal)[objq]
  pts = []
  for q in list(xs[::max(1, len(xs) // 40), :nq]) + [qg]:
    d1.qpos[:] = q; mujoco.mj_kinematics(m, d1)
    sel = np.array([(m.geom_bodyid[g] > 0 and m.geom_bodyid[g] not in hand) or
                    (m.body(m.geom_bodyid[g]).name in ('ff_tip', 'th_tip', 'jawl', 'jawr')) for g in range(m.ngeom)])
    pts.append(d1.geom_xpos[sel])
  pts = np.concatenate(pts); lo, hi = np.percentile(pts, 2, 0), np.percentile(pts, 98, 0)
  cam.lookat[:] = (lo + hi) / 2
  span = float(np.max(hi - lo)) / 2 + 0.05
  cam.distance = 1.3 * span / np.tan(np.deg2rad(m.vis.global_.fovy / 2)); cam.elevation = -28.0; cam.azimuth = 140.0
  opt = mujoco.MjvOption(); opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
  for gi in range(6):
    opt.geomgroup[gi] = 1 if gi in (0, 1, 2) else 0
  m2.vis.scale.contactwidth = 0.35; m2.vis.scale.contactheight = 0.08; m2.vis.rgba.contactpoint[:] = [1, 0.1, 0.1, 1]
  cam2 = mujoco.MjvCamera(); cam2.type = mujoco.mjtCamera.mjCAMERA_FREE
  cam2.lookat[:] = cam.lookat; cam2.distance = cam.distance; cam2.elevation = -35.0; cam2.azimuth = 30.0
  dg.qpos[:] = qg; mujoco.mj_forward(m, dg)
  r1.update_scene(dg, cam)
  ghost = []
  for i in range(r1.scene.ngeom):
    g = r1.scene.geoms[i]
    if g.objtype == mujoco.mjtObj.mjOBJ_GEOM and m.geom_bodyid[g.objid] in objb:
      ghost.append((int(g.type), g.size.copy(), g.pos.copy(), g.mat.copy(), int(g.dataid)))
  dt = m.opt.timestep; N = len(xs) - 1
  stride = max(1, int(np.ceil(N / 100)))
  try:
    font = ImageFont.load_default(size=14)
  except TypeError:
    font = ImageFont.load_default()
  ks = list(range(0, N + 1, stride))
  if ks[-1] != N:
    ks.append(N)
  frames = []
  for k in ks:
    row = []
    for (r, mm, d, lab, o, cc) in ((r1, m, d1, 'visual', None, cam), (r2, m2, d2, 'collision geometry + contacts (2nd view)', opt, cam2)):
      d.qpos[:] = xs[k, :nq]; mujoco.mj_forward(mm, d)
      if o is None:
        r.update_scene(d, cc)
      else:
        r.update_scene(d, cc, scene_option=o)
      for (ty, sz, pos, mat, did) in ghost:
        if r.scene.ngeom >= r.scene.maxgeom:
          break
        g = r.scene.geoms[r.scene.ngeom]
        mujoco.mjv_initGeom(g, ty, sz, pos, mat.flatten(), np.array([0.15, 0.8, 0.45, 0.3], np.float32))
        g.dataid = did if o is None else -1
        r.scene.ngeom += 1
      img = Image.fromarray(r.render()); dr = ImageDraw.Draw(img)
      dr.rectangle([0, 0, W, 20], fill=(255, 255, 255)); dr.text((6, 3), lab, fill=(20, 20, 20), font=font)
      dr.text((6, W - 20), f't = {k * dt:4.2f} s', fill=(20, 20, 20), font=font, stroke_width=2, stroke_fill=(255, 255, 255))
      row.append(np.asarray(img))
    frames.append(Image.fromarray(np.concatenate([row[0], np.full((W, 4, 3), 255, np.uint8), row[1]], 1)))
  dur = int(round(1000 * dt * stride))
  pal = [f.convert('P', palette=Image.ADAPTIVE, colors=96) for f in frames]
  pal[0].save(path, save_all=True, append_images=pal[1:], duration=[dur] * (len(pal) - 1) + [1500], loop=0, optimize=True)
  frames[-1].save(path.replace('.gif', '_final.png'))
  r1.close(); r2.close()
  return len(pal)


man = []
for name, cls in T.TASKS.items():
  if ONLY and name not in ONLY:
    continue
  t = cls(SEED)
  us = Wn.WITNESS[name](t) if name in Wn.WITNESS else t.witness()
  xs = sim_true(t, us)
  met = metric(t, xs[-1]); m0 = metric(t, t.x_0)
  key = f'{name}_s{SEED}'
  n = render(t, xs, f'{OUT}/{key}.gif', name)
  ent = dict(key=key, task=name, seed=SEED, metric=met, start_metric=m0, horizon_s=t.horizon * t.model.opt.timestep,
             frames=n, doc=(cls.__doc__ or '').strip(), bytes=os.path.getsize(f'{OUT}/{key}.gif'))
  man.append(ent); print(json.dumps({k: v for k, v in ent.items() if k != 'doc'}), flush=True)
old = {}
if os.path.exists(f'{OUT}/manifest.json'):
  old = {e['key']: e for e in json.load(open(f'{OUT}/manifest.json'))}
old.update({e['key']: e for e in man})
json.dump(list(old.values()), open(f'{OUT}/manifest.json', 'w'), indent=1)
