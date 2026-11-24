"""Fingertip / wrist inverse kinematics for scripting Allegro witnesses (kinematics only).

ik(m, qpos, targets, joints)  damped least squares: move `joints` so each geom in `targets`
                              ({geom_name: world point}) reaches its point; returns qpos.
"""
import numpy as np
import mujoco

TIPS = {'ff': 'ff_tip', 'mf': 'mf_tip', 'rf': 'rf_tip', 'th': 'th_tip'}
FINGER_JOINTS = {f: [f'{f}j{i}' for i in range(4)] for f in ('ff', 'mf', 'rf', 'th')}


def tip_geom(m, finger):
  b = m.body(TIPS[finger]).id
  gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b and m.geom_conaffinity[g]]
  return gs[0]


def ik(m, qpos, targets, joints, iters=200, damp=1e-3, step=0.5, offsets=None):
  """targets: {geom_id: world point}; joints: joint names allowed to move.
  offsets: {geom_id: local offset} -- point on the geom (default its centre)."""
  d = mujoco.MjData(m)
  q = np.array(qpos, float)
  dofs = [int(m.jnt_dofadr[m.joint(j).id]) for j in joints]
  qadr = [int(m.jnt_qposadr[m.joint(j).id]) for j in joints]
  lo = np.array([m.jnt_range[m.joint(j).id, 0] if m.jnt_limited[m.joint(j).id] else -np.inf for j in joints])
  hi = np.array([m.jnt_range[m.joint(j).id, 1] if m.jnt_limited[m.joint(j).id] else np.inf for j in joints])
  offsets = offsets or {}
  for _ in range(iters):
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
    errs, Js = [], []
    for g, p in targets.items():
      off = offsets.get(g, np.zeros(3))
      x = d.geom_xpos[g] + d.geom_xmat[g].reshape(3, 3) @ off
      jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
      mujoco.mj_jac(m, d, jp, jr, x, m.geom_bodyid[g])
      errs.append(np.asarray(p) - x); Js.append(jp[:, dofs])
    e = np.concatenate(errs); J = np.concatenate(Js)
    if np.linalg.norm(e) < 1e-5:
      break
    dq = J.T @ np.linalg.solve(J @ J.T + damp * np.eye(len(e)), e)
    q[qadr] = np.clip(q[qadr] + step * dq, lo, hi)
  return q, float(np.linalg.norm(e))


def tip_positions(m, qpos):
  d = mujoco.MjData(m); d.qpos[:] = qpos; mujoco.mj_kinematics(m, d)
  return {f: d.geom_xpos[tip_geom(m, f)].copy() for f in TIPS}
