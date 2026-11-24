"""SimToolReal-style Allegro tool-use tasks, group A: AllegroHammer, AllegroScrewdriver,
AllegroSpatulaFlip (2026-11-23 bench).

Shared design
-------------
* Embodiment: `embodiments.allegro_scene` (Allegro on a 6-DoF position-servoed wrist), palm-down
  start (palm quat `quat_mul(axis_quat(x, pi), [0,1,0,1]/sqrt2)`: fingers point +x and flex toward
  -z, thumb opposes from the -x side).  Each tool starts lying on the table with its handle along
  world y (plus a small random yaw); the hand must pinch it, lift / reorient it and use it on a
  target that the hand itself cannot touch (contype/conaffinity exclusion).
* Contact filter (contype / conaffinity bits, this module):
    1   hand fingertip capsules (+ palm capsules, hammer only) (ca 1) <-> tool handle (ct 1)
    2   tool support spheres (ct 2)                                   <-> table (ca 2)
    4   toast support spheres (ct 4)                                  <-> table (ca 4)
    8   fingertip spheres added at the 4 tip ends (ct 8)              <-> table (ca 8)
    16  target (nail head / screw posts / toast crust sphere, ct 16)  <-> tool working geom (ca 16)
  Palm vs table is replaced by a wrist-height floor (lower range of `wz`); the finger links
  (proximal/medial) do not collide (embodiments default).
* Finger servos use kp 3 (embodiment default 1; omega*dt ~ 1.0 at dt 0.04) for a firmer pinch.
* Witnesses: scripted keyframes plus a few dozen true-law rollouts each (grasp measured after
  the pick, alignment / press depth / twist overshoot refined on rollouts) -- the returned
  controls are open loop.  Robust pinch = tips 8 mm below the handle axis (a cage; screwdriver,
  spatula) on handles raised 2 cm by end spheres; wrist rotations kept slow (fast wrist rotation
  makes the tool creep out of the pinch under the true law).
* The solver supports a fixed terminal goal only, so the outcome is a state of the target
  (nail depth, screw angle, toast pose); the tool and hand have weight 0 in the goal cost.
"""
import numpy as np
import mujoco
import embodiments as E
from embodiments import fmt
from bench_tasks import register, BenchTask, _qpos_adr, _diag, pose_Q, yaw_quat

Q_PALM_DOWN = E.quat_mul(E.axis_quat([1, 0, 0], np.pi), np.array([0, 1, 0, 1]) / np.sqrt(2))
TIP_BODIES = ('ff_tip', 'mf_tip', 'rf_tip', 'th_tip')
TIP_END = {'ff_tip': (0, 0, 0.029), 'mf_tip': (0, 0, 0.029), 'rf_tip': (0, 0, 0.029), 'th_tip': (0, 0, 0.043)}
WRIST_FLOOR = 0.03          # wz lower bound: palm capsules (lowest point 11 mm below the wrist) stay above the table
FINGER_ACTS = tuple(f'{f}a{i}' for f in ('ff', 'mf', 'rf', 'th') for i in range(4))


def hand_scene(env_world, *, ranges, tip_table=True, palm_exclude=(), extra_contact='', **kw):
  """`allegro_scene` (palm down) + a massless sphere (r 12 mm) at each fingertip end colliding with
  the table only (ct 8), and optional palm exclusions."""
  extra = extra_contact + ''.join(f'\n    <exclude body1="palm" body2="{b}"/>' for b in palm_exclude)
  xml = E.allegro_scene(env_world, wrist_pos=(0, 0, 0), palm_quat=Q_PALM_DOWN, ranges=ranges,
                        extra_contact=extra, **kw)
  if tip_table:
    for b in TIP_BODIES:
      tag = f'<body name="{b}" gravcomp="1">'
      assert tag in xml, b
      xml = xml.replace(tag, tag + f'\n<geom name="{b}_end" type="sphere" size="0.012" pos="{fmt(TIP_END[b])}" '
                                   f'mass="0" contype="8" conaffinity="0" group="3" rgba="0 1 0 0.3"/>')
  return xml


def hand_model(env_world, **kw):
  return E.from_xml_in(E.ALLEGRO_DIR, hand_scene(env_world, **kw))


def rot_z(a):
  c, s = np.cos(a), np.sin(a)
  return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def quat_to_mat(q):
  m = np.zeros(9); mujoco.mju_quat2Mat(m, np.asarray(q, float)); return m.reshape(3, 3)


def mat_to_quat(R):
  q = np.zeros(4); mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(9)); return q


# ---------------------------------------------------------------------------------------------
# Palm-down pinch of a horizontal handle along the wrist's y axis (cached finger IK)
# ---------------------------------------------------------------------------------------------
# With zero wrist angles the handle axis is world y at (xh, yh, zh); the wrist sits at
# (xh, yh, zh) + PINCH_OFF.  Fingers ff/mf/rf at y = +0.04/0/-0.04 on the handle's +x side, thumb
# at y=0 on the -x side, all at handle height + 4 mm (scripts/example_allegro_lift.py).
PINCH_OFF = np.array([-0.01, -0.013, 0.104])
_PINCH_CACHE = {}


def pinch_fingers(m, r_handle, inward, dz=0.004, spread=0.04):
  """Finger joint targets (16,) for a pinch of a handle of radius r_handle; inward > 0 squeezes
  (targets that far past the surface), < 0 opens."""
  import handik as H
  key = (m.nq, round(r_handle, 5), round(inward, 5), round(dz, 5), round(spread, 5))
  if key in _PINCH_CACHE:
    return _PINCH_CACHE[key]
  wa = _qpos_adr(m, 'wx'); fa = _qpos_adr(m, 'ffj0')
  q = np.zeros(m.nq)
  q[wa:wa + 3] = PINCH_OFF; q[wa + 3:wa + 6] = 0      # handle axis at the origin
  q[_qpos_adr(m, 'thj0')] = 1.4
  for j in range(m.njnt):                             # free joints: identity quats (kinematics only)
    if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
      q[m.jnt_qposadr[j] + 3] = 1.0
  s = r_handle + 0.012 - 0.002 - inward
  tg = {H.tip_geom(m, 'ff'): [s, spread, dz], H.tip_geom(m, 'mf'): [s, 0, dz],
        H.tip_geom(m, 'rf'): [s, -spread, dz], H.tip_geom(m, 'th'): [-s, 0, dz]}
  qs, e = H.ik(m, q, tg, sum(H.FINGER_JOINTS.values(), []), iters=3000)
  out = qs[fa:fa + 16].copy()
  _PINCH_CACHE[key] = out
  return out


def fingers_dict(qf):
  return {n: float(v) for n, v in zip(FINGER_ACTS, qf)}


# ---------------------------------------------------------------------------------------------
# Wrist kinematics for witnesses: wrist hinges are nested x -> y -> z, R_w = Rx(a) Ry(b) Rz(c)
# ---------------------------------------------------------------------------------------------
def _rx(a):
  c, s = np.cos(a), np.sin(a); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
  c, s = np.cos(a), np.sin(a); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def wrist_R(ang):
  return _rx(ang[0]) @ _ry(ang[1]) @ rot_z(ang[2])


def euler_xyz(R):
  b = np.arcsin(np.clip(R[0, 2], -1, 1))
  return np.array([np.arctan2(-R[1, 2], R[2, 2]), b, np.arctan2(-R[0, 1], R[0, 0])])


def euler_near(R, ref):
  """XYZ angles of R on the branch (and 2 pi wrap) closest to `ref`."""
  a, b, c = euler_xyz(R)
  best = None
  for cand in (np.array([a, b, c]), np.array([a + np.pi, np.pi - b, c + np.pi])):
    cand = ref + (cand - ref + np.pi) % (2 * np.pi) - np.pi
    d = np.abs(cand - ref).sum()
    if best is None or d < best[0]:
      best = (d, cand)
  return best[1]


WRIST_LEAD = (0.1, 0.1, 0.1, 0.075, 0.075, 0.075)   # kv / kp of the wrist servos (s)


def lead_wrist(task, us, lead=WRIST_LEAD):
  """Velocity feed-forward for the wrist position servos (u = q - kv qdot / kp tracks with no
  velocity lag): command each wrist target `lead` seconds early."""
  m = task.model; dt = m.opt.timestep
  out = np.array(us, float).copy()
  tk = np.arange(len(us)) * dt
  for i, j in enumerate(E.WRIST_JOINTS):
    c = m.actuator(f'a_{j}').id
    out[:, c] = np.interp(tk + lead[i], tk, us[:, c])
  return out


def tool_rel(m, x, joint):
  """Pose of free joint `joint` in the wrist frame: (p_rel, R_rel)."""
  w = _qpos_adr(m, 'wx'); a = _qpos_adr(m, joint)
  Rw = wrist_R(x[w + 3:w + 6])
  return Rw.T @ (x[a:a + 3] - x[w:w + 3]), Rw.T @ quat_to_mat(x[a + 3:a + 7])


def wrist_for(p_des, R_des, rel, point=np.zeros(3), ref=None):
  """Wrist (position, angles) that puts the tool's local `point` at p_des with orientation R_des
  for a rigid grasp `rel` = (p_rel, R_rel); angles on the branch nearest `ref` if given."""
  p_rel, R_rel = rel
  Rw = R_des @ R_rel.T
  ang = euler_xyz(Rw) if ref is None else euler_near(Rw, np.asarray(ref, float))
  return p_des - Rw @ (p_rel + R_rel @ point), ang


def table_geom(ca):
  return (f'    <geom name="table" type="box" size="0.4 0.4 0.1" pos="0 0 -0.1" rgba="0.55 0.45 0.35 1" '
          f'contype="0" conaffinity="{ca}" friction="0.6"/>')


def _wrist_kf(p, yaw=None, rx=None, ry=None):
  d = dict(a_wx=float(p[0]), a_wy=float(p[1]), a_wz=float(p[2]))
  if yaw is not None: d['a_wrz'] = float(yaw)
  if rx is not None: d['a_wrx'] = float(rx)
  if ry is not None: d['a_wry'] = float(ry)
  return d


# =============================================================================================
# AllegroHammer
# =============================================================================================
HAM = dict(r_head=0.02, hh_head=0.02, r_handle=0.011, L=0.16, r_knob=0.02,
           m_head=0.06, m_handle=0.05, m_knob=0.02)
NAIL_XY = np.array([-0.10, 0.12])
BLOCK_TOP = 0.04
NAIL_R = 0.006
NAIL_Z0 = 0.025             # nail-head sphere centre above the block top (nail sticks out 3.1 cm)


@register
class AllegroHammer(BenchTask):
  """SimToolReal 'hammer': pick up a hammer lying on the table and drive a nail.

  Model: Allegro on a 6-DoF wrist (palm down).  Hammer = head (cylinder r 2 cm, 4 cm tall, axis
  vertical, 60 g) + handle (capsule r 11 mm, 14 cm, 50 g) + end knob (sphere r 2 cm, 20 g); it
  starts lying with the head standing on its face and the handle horizontal along world y (the
  knob props the handle end: simplification, "handle raised on its head"; head face down =
  striking face down, so the palm-down hand needs no in-hand reorientation, only a pick,
  transport, a wrist correction for the grasp roll, and a strike / press).  The nail is a
  vertical slide joint (range -3.5..+0.1 cm, nail head 3.1 cm above the block) in a fixed block
  (visual only) with dry friction NAIL_FRICTION = 4 N and damping 5 N s/m: the hammer's weight
  (1.3 N) does not move it; it must be pressed or struck through the grip (or the hammer dropped
  head-first from >~6 cm).  Goal: nail driven down GOAL_DEPTH = 2 cm (the range allows 3.5 cm,
  so overdriving is penalised).  Note: at dt 0.04 the true law shows a slow creep of the nail
  under its own weight (~0.3 mm over the horizon).
  Contacts (14): 4 fingertip capsules + 2 palm capsules vs handle (capsule-capsule); 4 fingertip
  end spheres vs table; head support spheres (2) + knob vs table; hammer head (cylinder) vs nail-head
  sphere.  Not modelled (simplifications): hand vs nail / block (excluded: only the hammer head
  can act on the nail), hammer vs block (the head cannot reach the block top before the nail hits
  its range limit), finger links, palm vs table (wrist-height floor wz >= WRIST_FLOOR).
  Cost: nail joint (weight 400); Qf = 40 Q.  Success: metric < 0.01 (|depth error| < 5 mm).
  Seeds randomize the hammer's (x, y, yaw).  Witness (~4.4 s of 6 s): pinch the handle near its
  balance point, lift, roll / move the wrist so the head face is down over the nail, press
  (depth by bisection on true-law rollouts), retract."""
  HORIZON = 150
  DT = 0.04
  GOAL_DEPTH = 0.02
  NAIL_FRICTION = 4.0
  NAIL_DAMPING = 5.0
  FINGER_KP = 3.0

  @classmethod
  def env(cls):
    h = HAM
    tbl = table_geom(2 | 8)
    block = (f'    <geom name="block" type="box" size="0.03 0.03 {BLOCK_TOP / 2}" pos="{NAIL_XY[0]} {NAIL_XY[1]} {BLOCK_TOP / 2}" '
             f'contype="0" conaffinity="0" rgba="0.6 0.4 0.2 1"/>')
    nail = f"""    <body name="nail" pos="{NAIL_XY[0]} {NAIL_XY[1]} {BLOCK_TOP}">
      <joint name="nail" type="slide" axis="0 0 1" range="-0.035 0.001" frictionloss="{cls.NAIL_FRICTION}" damping="{cls.NAIL_DAMPING}" armature="0.01"/>
      <geom name="nail_shaft" type="capsule" size="0.002" fromto="0 0 -0.03 0 0 {NAIL_Z0}" mass="0.01" contype="0" conaffinity="0" rgba="0.7 0.7 0.75 1"/>
      <geom name="nail_head" type="sphere" size="{NAIL_R}" pos="0 0 {NAIL_Z0}" mass="0.005" contype="16" conaffinity="0" rgba="0.7 0.7 0.75 1"/>
    </body>"""
    rs = 0.006
    ham = f"""    <body name="hammer" pos="0 0 {h['hh_head']}">
      <freejoint name="hammer"/>
      <geom name="ham_head" type="cylinder" size="{h['r_head']} {h['hh_head']}" mass="{h['m_head']}" contype="0" conaffinity="16" rgba="0.4 0.4 0.45 1" friction="0.6"/>
      <geom name="ham_handle" type="capsule" size="{h['r_handle']}" fromto="0 {-h['r_head']} 0 0 {-h['r_head'] - h['L'] + 0.02} 0" mass="{h['m_handle']}" contype="1" conaffinity="0" rgba="0.6 0.4 0.2 1" friction="1"/>
      <geom name="ham_knob" type="sphere" size="{h['r_knob']}" pos="0 {-h['r_head'] - h['L'] + 0.02} 0" mass="{h['m_knob']}" contype="2" conaffinity="0" rgba="0.5 0.3 0.15 1"/>
      <geom name="ham_s1" type="sphere" size="{rs}" pos="0.013 0 {-h['hh_head'] + rs}" mass="0" contype="2" conaffinity="0" group="3"/>
      <geom name="ham_s2" type="sphere" size="{rs}" pos="-0.013 0 {-h['hh_head'] + rs}" mass="0" contype="2" conaffinity="0" group="3"/>
    </body>"""
    return '\n'.join([tbl, block, nail, ham])

  @classmethod
  def build(cls):
    ranges = ((-0.4, 0.4), (-0.4, 0.4), (WRIST_FLOOR, 0.4), (-1.6, 1.6), (-1.6, 1.6), (-3.2, 3.2))
    return hand_model(cls.env(), ranges=ranges, timestep=cls.DT, finger_kp=cls.FINGER_KP)

  def __init__(self, seed=0):
    rng = np.random.default_rng(7100 + seed)
    m = self.build()
    self.ham_xy = np.array([0.06, 0.02]) + rng.uniform(-0.02, 0.02, 2)
    self.ham_yaw = rng.uniform(-0.3, 0.3)
    nx = m.nq + m.nv
    x0 = np.zeros(nx)
    a = _qpos_adr(m, 'hammer')
    x0[a:a + 3] = [*self.ham_xy, HAM['hh_head'] - 1e-4]
    x0[a + 3:a + 7] = yaw_quat(self.ham_yaw)
    w = _qpos_adr(m, 'wx'); f = _qpos_adr(m, 'ffj0')
    x0[w:w + 6] = [0.0, -0.05, 0.22, 0, 0, 0]
    self.q_open = pinch_fingers(m, HAM['r_handle'], -0.03)
    x0[f:f + 16] = self.q_open
    g = x0.copy()
    n = _qpos_adr(m, 'nail')
    g[n] = -self.GOAL_DEPTH
    Q = _diag(nx, {n: 400.0})
    u_ref = np.concatenate([x0[w:w + 6], self.q_open])
    R = np.diag([1e-2] * 3 + [1e-3] * 3 + [1e-2] * 16)
    super().__init__(model=m, horizon=self.HORIZON, x_0=x0, x_goal=g, Q=Q, Qf=40 * Q,
                     u_ref=u_ref, R=R, contact_smoothing=0.002)

  # -- witness ------------------------------------------------------------------------------
  W_GRASP_Y = -0.06         # grasp centre along the handle (hammer frame, head at 0)
  W_LIFT = 0.10

  W_T = dict(top=0.6, down=1.0, squeeze=1.5, lift=2.1, above=3.0, settle=3.3, press=3.8, hold=4.1, retract=4.4)

  def witness(self, verbose=False):
    """Pinch the handle near its balance point, lift, measure the grasp (true-law rollout of the
    pick), turn the wrist so the head face is down above the nail, press down (press depth found
    by bisection on true-law rollouts), and retract."""
    from simtools import keyframes, sim_true
    m = self.model; T = self.W_T
    psi = self.ham_yaw
    Rz = rot_z(psi)
    hp = np.array([*self.ham_xy, HAM['hh_head']])
    wgrasp = hp + Rz @ np.array([0, self.W_GRASP_Y, 0]) + Rz @ PINCH_OFF
    qs = pinch_fingers(m, HAM['r_handle'], 0.015)
    pick = [(T['top'], _wrist_kf(wgrasp + [0, 0, 0.04], psi)), (T['down'], _wrist_kf(wgrasp)),
            (T['squeeze'], fingers_dict(qs)), (T['lift'], _wrist_kf(wgrasp + [0, 0, self.W_LIFT]))]
    k_lift = int(round(T['lift'] / self.DT)) + 3
    X = sim_true(self, keyframes(self, pick))
    rel = tool_rel(m, X[k_lift], 'hammer')
    nail_top = np.array([*NAIL_XY, BLOCK_TOP + NAIL_Z0 + NAIL_R])
    a = _qpos_adr(m, 'hammer'); n = _qpos_adr(m, 'nail')

    def plan(off, wp):
      pa, aa = wrist_for(nail_top + off + [0, 0, HAM['hh_head'] + 0.03], rot_z(0.0), rel)
      pp, _ = wrist_for(nail_top + off + [0, 0, HAM['hh_head'] - self.GOAL_DEPTH - wp], rot_z(0.0), rel)
      kf = pick + [(T['above'], _wrist_kf(pa, aa[2], aa[0], aa[1])), (T['settle'], {}),
                   (T['press'], _wrist_kf(pp)), (T['hold'], {}), (T['retract'], _wrist_kf(pa))]
      us = keyframes(self, kf)
      return us, sim_true(self, us)

    off = np.zeros(3)
    for it in range(2):           # align the head over the nail
      us, X = plan(off, 0.0)
      off[:2] -= X[int(round(T['settle'] / self.DT)), a:a + 2] - nail_top[:2]
    lo, hi = -0.01, 0.10           # bisection on the press depth (deeper press -> deeper nail)
    for it in range(12):
      wp = 0.5 * (lo + hi)
      us, X = plan(off, wp)
      depth = -X[-1, n]
      if verbose:
        print(f'  press {wp:.4f} -> depth {depth:.4f}')
      if abs(depth - self.GOAL_DEPTH) < 0.001:
        break
      lo, hi = (wp, hi) if depth < self.GOAL_DEPTH else (lo, wp)
    return us


# =============================================================================================
# AllegroScrewdriver
# =============================================================================================
SD = dict(r_handle=0.013, hl_handle=0.045, r_shaft=0.003, z_tip=-0.11, r_blade=0.0015, hw_blade=0.009,
          m_handle=0.04, m_shaft=0.008, m_blade=0.002, r_sup=0.02)
SCREW_XY = np.array([-0.06, 0.10])
POST = dict(a=0.006, d=0.0045, r=0.0015, z0=0.002, z1=0.010)     # slot posts on the screw head
SCREW_YAW = 0.30           # slot direction at angle 0 (chosen so the pi/2 twist fits the wry range)


@register
class AllegroScrewdriver(BenchTask):
  """SimToolReal 'screwdriver': pick up a screwdriver lying on the table, turn it upright, put
  the blade in the screw's slot and turn the screw by GOAL_ANGLE (pi/2).

  Model: Allegro on a 6-DoF wrist (palm down).  Screwdriver = handle (capsule r 13 mm, 9 cm,
  40 g) with end bolsters (spheres r 2 cm, its table supports) + shaft (capsule r 3 mm, 6.5 cm,
  8 g) + blade (horizontal capsule r 1.5 mm, 2.1 cm wide, 2 g, with a tip sphere vs the table);
  it starts lying along world y (blade toward -y, blade edge roughly vertical) with the handle
  axis 2 cm up (simplification: bolstered handle so the pinch can cage it from below).  The screw
  is a hinge about z in a fixed block (visual only), viscous only (SCREW_DAMPING = 0.02 N m s/rad; dry friction
  made the light screw rattle in the slot after the twist)
  and damping 0.002 N m s; its slot is two vertical post capsules (r 1.5 mm, 8 mm tall) at
  (+a, +d) and (-a, -d) = (+-6, +-4.5) mm on the head, slot direction SCREW_YAW: the blade, once
  between them (0.25 rad play), turns the screw counter-clockwise (the goal direction; the slot is
  one-sided for clockwise turns: simplification).  The hand must reorient the tool by 90 deg
  (wrist roll, blade down), insert the blade and twist the tool about its own axis -- which for
  this wrist is a wrist rotation about vertical plus a 13 cm circular wrist translation (the wrist
  hinges are 13 cm from the pinched tool axis).  SCREW_YAW and the blade-roll randomization are
  chosen so the pi/2 twist (plus slot play and overshoot) fits the wry range without a regrasp.
  Contacts (13): 4 fingertip capsules vs handle; 4 fingertip end spheres vs table; 2 bolsters +
  blade-tip sphere vs table; blade vs 2 posts (capsule-capsule).  Not modelled: hand vs screw /
  block (excluded: only the blade can turn the screw), palm vs tool (excluded: a pinch never
  uses it), shaft vs table, tool vs block and blade vs the screw-head top (the blade can sink below
  the posts and lose the slot, but gains nothing), finger links, palm vs table (wrist floor).
  Cost: screw angle (weight 1); Qf = 40 Q.  Success: metric < 0.01 (|angle error| < 0.1 rad).
  Seeds randomize the screwdriver's (x, y, yaw) and its roll about its axis (blade angle, +-0.3).
  Witness (~5.3 s of 6 s): pinch (cage) the handle, lift, roll the wrist 90 deg while moving over
  the screw, align the blade over the slot at a hover point (refined on rollouts; grasp re-measured
  there), insert, twist 12 keyframes about the screw axis with per-keyframe drift corrections learnt
  on rollouts (iterative learning control, first in the air, then in the slot) and wrist velocity
  feed-forward, retract; the twist overshoot is grid-searched for the final angle."""
  HORIZON = 150
  DT = 0.04
  GOAL_ANGLE = np.pi / 2
  SCREW_FRICTION = 0.0      # 2026-11-23: was 0.02 N m dry friction (+ damping 0.002); the light screw then rattled
  SCREW_DAMPING = 0.02      # in the slot after the twist (+-1.7 rad/s at t = T), failing the at-rest goal. Viscous only.
  FINGER_KP = 3.0

  @classmethod
  def env(cls):
    s, p = SD, POST
    tbl = table_geom(2 | 8)
    block = (f'    <geom name="block" type="box" size="0.03 0.03 {BLOCK_TOP / 2}" pos="{SCREW_XY[0]} {SCREW_XY[1]} {BLOCK_TOP / 2}" '
             f'contype="0" conaffinity="0" rgba="0.6 0.4 0.2 1"/>')
    screw = f"""    <body name="screw" pos="{SCREW_XY[0]} {SCREW_XY[1]} {BLOCK_TOP}" quat="{fmt(yaw_quat(SCREW_YAW))}">
      <joint name="screw" type="hinge" axis="0 0 1" frictionloss="{cls.SCREW_FRICTION}" damping="{cls.SCREW_DAMPING}" armature="0.0005"/>
      <geom name="screw_head" type="cylinder" size="0.012 {p['z0'] / 2}" pos="0 0 {p['z0'] / 2}" mass="0.01" contype="0" conaffinity="0" rgba="0.75 0.75 0.8 1"/>
      <geom name="screw_mark" type="box" size="0.011 0.0008 0.0003" pos="0 0 {p['z0']}" mass="0" contype="0" conaffinity="0" rgba="0.1 0.1 0.1 1"/>
      <geom name="post1" type="capsule" size="{p['r']}" fromto="{p['a']} {p['d']} {p['z0']} {p['a']} {p['d']} {p['z1']}" mass="0.001" contype="16" conaffinity="0" rgba="0.8 0.2 0.2 1"/>
      <geom name="post2" type="capsule" size="{p['r']}" fromto="{-p['a']} {-p['d']} {p['z0']} {-p['a']} {-p['d']} {p['z1']}" mass="0.001" contype="16" conaffinity="0" rgba="0.8 0.2 0.2 1"/>
    </body>"""
    tool = f"""    <body name="sdriver" pos="0 0 0.2">
      <freejoint name="sdriver"/>
      <geom name="sd_handle" type="capsule" size="{s['r_handle']}" fromto="0 0 {-s['hl_handle']} 0 0 {s['hl_handle']}" mass="{s['m_handle']}" contype="1" conaffinity="0" rgba="0.85 0.2 0.15 1" friction="1"/>
      <geom name="sd_shaft" type="capsule" size="{s['r_shaft']}" fromto="0 0 {-s['hl_handle']} 0 0 {s['z_tip']}" mass="{s['m_shaft']}" contype="0" conaffinity="0" rgba="0.7 0.7 0.75 1"/>
      <geom name="sd_blade" type="capsule" size="{s['r_blade']}" fromto="{-s['hw_blade']} 0 {s['z_tip'] - s['r_blade']} {s['hw_blade']} 0 {s['z_tip'] - s['r_blade']}" mass="{s['m_blade']}" contype="0" conaffinity="16" rgba="0.6 0.6 0.65 1" friction="0.8"/>
      <geom name="sd_s1" type="sphere" size="{s['r_sup']}" pos="0 0 {s['hl_handle']}" mass="0" contype="2" conaffinity="0" rgba="0.2 0.2 0.2 1"/>
      <geom name="sd_s2" type="sphere" size="{s['r_sup']}" pos="0 0 {-s['hl_handle']}" mass="0" contype="2" conaffinity="0" rgba="0.2 0.2 0.2 1"/>
      <geom name="sd_s3" type="sphere" size="{s['r_blade']}" pos="0 0 {s['z_tip'] - s['r_blade']}" mass="0" contype="2" conaffinity="0" group="3"/>
    </body>"""
    return '\n'.join([tbl, block, screw, tool])

  @classmethod
  def build(cls):
    ranges = ((-0.4, 0.4), (-0.4, 0.4), (WRIST_FLOOR, 0.4), (-1.6, 1.6), (-1.6, 1.6), (-3.2, 3.2))
    return hand_model(cls.env(), ranges=ranges, timestep=cls.DT, finger_kp=cls.FINGER_KP,
                      palm_exclude=('sdriver',))

  def __init__(self, seed=0):
    rng = np.random.default_rng(7200 + seed)
    m = self.build()
    self.sd_xy = np.array([0.06, 0.0]) + rng.uniform(-0.02, 0.02, 2)
    self.sd_yaw = rng.uniform(-0.3, 0.3)
    self.sd_roll = np.pi / 2 + rng.uniform(-0.3, 0.3)   # blade roughly vertical while lying
    nx = m.nq + m.nv
    x0 = np.zeros(nx)
    a = _qpos_adr(m, 'sdriver')
    x0[a:a + 3] = [*self.sd_xy, SD['r_sup'] - 1e-4]
    R0 = rot_z(self.sd_yaw) @ _rx(-np.pi / 2) @ rot_z(self.sd_roll)
    x0[a + 3:a + 7] = mat_to_quat(R0)
    w = _qpos_adr(m, 'wx'); f = _qpos_adr(m, 'ffj0')
    x0[w:w + 6] = [0.0, -0.05, 0.22, 0, 0, 0]
    self.q_open = pinch_fingers(m, SD['r_handle'], -0.03, dz=self.W_DZ)
    x0[f:f + 16] = self.q_open
    g = x0.copy()
    n = _qpos_adr(m, 'screw')
    g[n] = self.GOAL_ANGLE
    Q = _diag(nx, {n: 1.0})
    u_ref = np.concatenate([x0[w:w + 6], self.q_open])
    R = np.diag([1e-2] * 3 + [1e-3] * 3 + [1e-2] * 16)
    super().__init__(model=m, horizon=self.HORIZON, x_0=x0, x_goal=g, Q=Q, Qf=40 * Q,
                     u_ref=u_ref, R=R, contact_smoothing=0.002)

  # -- witness ------------------------------------------------------------------------------
  W_T = dict(top=0.5, down=0.9, squeeze=1.3, lift=1.7, above=2.9, hover=3.2, insert=3.55,
             dwell=3.7, twist=5.0, retract=5.3)
  W_HOVER = 0.012           # blade height above the insertion point while aligning
  W_DZ = -0.008             # pinch height relative to the handle axis (< 0: tips below the axis, a cage)
  W_OVER = 0.25             # twist overshoot (rad) for the blade-post clearance and compliance

  def witness(self, verbose=False):
    """Pinch the handle, lift and roll the wrist 90 deg (blade down), measure the grasp, align
    the blade with the slot above the screw, insert, twist about the tool axis (wrist moves on a
    circle around it), retract.  Alignment and the twist overshoot are refined on true-law
    rollouts."""
    from simtools import keyframes, sim_true
    m = self.model; T = self.W_T; dt = self.DT
    psi = self.sd_yaw; Rz = rot_z(psi)
    hp = np.array([*self.sd_xy, SD['r_sup']])
    wgrasp = hp + Rz @ PINCH_OFF
    qs = pinch_fingers(m, SD['r_handle'], 0.015, dz=self.W_DZ)
    lift_p = wgrasp + [0, 0, 0.10]
    pick = [(T['top'], _wrist_kf(wgrasp + [0, 0, 0.04], psi)), (T['down'], _wrist_kf(wgrasp)),
            (T['squeeze'], fingers_dict(qs)), (T['lift'], _wrist_kf(lift_p))]
    k_lift = int(round(T['lift'] / dt)) + 3
    k_hover = int(round(T['hover'] / dt)) + 3
    X = sim_true(self, keyframes(self, pick))
    rel_a = tool_rel(m, X[k_lift], 'sdriver')      # grasp model for the approach (refined below)
    rel_t = rel_a                                   # grasp model for insertion / twist (re-measured at the hover)
    ref0 = np.array([np.pi / 2, 0.0, psi])
    blade = np.array([0, 0, SD['z_tip'] - SD['r_blade']])
    a = _qpos_adr(m, 'sdriver'); n = _qpos_adr(m, 'screw')
    p_ins = np.array([*SCREW_XY, BLOCK_TOP + 0.5 * (POST['z0'] + POST['z1'])])
    p_hov = p_ins + [0, 0, self.W_HOVER]
    lo, hi = m.jnt_range[m.joint('wry').id]

    nst = 12
    n_tw = nst + 3                 # keyframes from insertion to retraction

    def plan(phi0, over, off, dphi, offs=None, h=0.0):
      """Approach (grasp model rel_a, corrected by off / dphi) to the hover point with the blade
      along the slot at yaw phi0; insert and twist about the screw axis using the grasp measured
      at the hover (rel_t) plus per-keyframe corrections `offs` (iterative learning on rollouts:
      they cancel the slow drift of the tool in the pinch); retract."""
      offs = np.zeros((n_tw, 3)) if offs is None else offs
      kf = list(pick); ref = ref0; times = []
      def go(t, p, phi, rel, o=np.zeros(3), dp=0.0, Rh=None):
        nonlocal ref
        Rd = rot_z(phi + dp) if Rh is None else rot_z(phi - phi0) @ Rh   # Rh: keep the measured tilt
        pw, aw = wrist_for(p + o, Rd, rel, blade, ref=ref)
        ref = aw
        kf.append((t, _wrist_kf(pw, aw[2], aw[0], aw[1])))
      go(T['above'], p_ins + [0, 0, 0.03], phi0, rel_a, off, dphi)
      go(T['hover'], p_hov, phi0, rel_a, off, dphi)
      go(T['hover'] + 0.12, p_hov, phi0, rel_a, off, dphi)
      pi_ = p_ins + [0, 0, h]
      tw = [(T['insert'], pi_, phi0), (T['dwell'], pi_, phi0)]
      tw += [(T['dwell'] + (T['twist'] - T['dwell']) * i / nst, pi_, phi0 + (self.GOAL_ANGLE + over) * i / nst)
             for i in range(1, nst + 1)]
      tw += [(T['retract'], p_ins + [0, 0, 0.04], phi0 + self.GOAL_ANGLE + over)]
      for jj, (t, p, phi) in enumerate(tw):
        go(t, p, phi, rel_t, offs[jj], Rh=R_h)
        times.append(t)
      us = lead_wrist(self, keyframes(self, kf))
      return us, sim_true(self, us), times

    def blade_at(X, k):
      Rt = quat_to_mat(X[k, a + 3:a + 7])
      return X[k, a:a + 3] + Rt @ blade, Rt

    # blade direction: SCREW_YAW or SCREW_YAW + pi, whichever keeps wry in range for the whole twist
    best = None
    for phi0 in (SCREW_YAW, SCREW_YAW + np.pi, SCREW_YAW - np.pi):
      _, a0 = wrist_for(p_ins, rot_z(phi0), rel_a, blade, ref=ref0)
      _, a1 = wrist_for(p_ins, rot_z(phi0 + self.GOAL_ANGLE + self.W_OVER), rel_a, blade, ref=a0)
      viol = max(lo - min(a0[1], a1[1]), max(a0[1], a1[1]) - hi, -0.3)
      if best is None or viol < best[0]:
        best = (viol, phi0)
    phi0 = best[1]
    off = np.zeros(3); dphi = 0.0; R_h = rot_z(phi0)
    for it in range(5):            # align the blade over the slot at the hover; re-measure the grasp there
      us, X, times = plan(phi0, self.W_OVER, off, dphi)
      pb, Rt = blade_at(X, k_hover)
      err = pb - p_hov
      eyaw = (np.arctan2(Rt[1, 0], Rt[0, 0]) - phi0 + np.pi / 2) % np.pi - np.pi / 2
      if verbose:
        print(f'  align it {it}: blade err {np.round(err * 1000, 2)} mm, yaw err {eyaw:+.3f}, screw {X[-1, n]:.3f}')
      off -= err; dphi -= eyaw
      rel_t = tool_rel(m, X[k_hover], 'sdriver'); R_h = Rt

    def ilc(over, sched=(0.03,) * 4 + (0.0,) * 4):
      """Learn per-keyframe corrections in the air first (blade 3 cm above the slot), then at the
      insertion height."""
      offs = np.zeros((n_tw, 3))
      for it, h in enumerate(sched):
        us, X, times = plan(phi0, over, off, dphi, offs, h)
        errs = []
        for jj, t in enumerate(times[:-1]):
          k = min(int(round(t / dt)) + 2, len(X) - 1)
          pb, _ = blade_at(X, k)
          e = pb - (p_ins + [0, 0, h])
          errs.append(np.linalg.norm(e[:2]))
          if np.isfinite(e).all() and np.linalg.norm(e) < 0.01:
            offs[jj] -= 0.7 * e
        offs[-1] = offs[-2]
        if verbose:
          print(f'    ilc over {over:+.2f} it {it} h {h}: max blade xy err {1000 * max(errs):.2f} mm, screw {X[-1, n]:.3f}')
      us, X, times = plan(phi0, over, off, dphi, offs)
      return us, X

    # twist overshoot: grid search on true-law rollouts (the final angle is not monotone in it)
    best = None
    for grid in (np.linspace(-0.3, 0.6, 7), None):
      if grid is None:
        grid = best[1] + np.linspace(-0.1, 0.1, 5)
      for over in grid:
        us, X = ilc(float(over))
        e = abs(X[-1, n] - self.GOAL_ANGLE) if np.isfinite(X).all() else np.inf
        if verbose:
          print(f'  over {over:+.3f} -> screw {X[-1, n]:.4f}')
        if best is None or e < best[0]:
          best = (e, float(over), us)
      if best[0] < 0.02:
        break
    return best[2]


# =============================================================================================
# AllegroSpatulaFlip
# =============================================================================================
SP = dict(r_handle=0.012, y0=-0.15, y1=-0.03, r_end=0.02, r_neck=0.004, blade_c=(0.0, 0.04, -0.0194),
          blade_h=(0.03, 0.04, 0.0006), r_bs=0.0006, m_handle=0.03, m_neck=0.005, m_blade=0.02)
TOAST_H = (0.04, 0.04, 0.008)          # half sizes (8 x 8 x 1.6 cm)
TOAST_R = TOAST_H[2]                   # support spheres: radius = half thickness, at mid-thickness
TOAST_PIV = TOAST_H[0] - TOAST_R       # pivot-sphere x offset (local +x edge)
TOAST_XM = -0.012                      # inboard support sphere x offset (keeps the -x crust raised)
TOAST_XL, TOAST_RL = 0.034, 0.004      # crust (lift) sphere at local x = -TOAST_XL, radius TOAST_RL
TOAST_ROLL = 2 * TOAST_PIV + np.pi * TOAST_R   # ideal roll-over displacement (pivot spheres roll pi * r)
SPAT_FLOOR = 0.03


@register
class AllegroSpatulaFlip(BenchTask):
  """SimToolReal 'spatula': pick up a spatula lying on the table, slide its blade under a toast
  slice and flip the toast upside down (180 deg about its local y axis) onto the spot next to it.

  Model: Allegro on a 6-DoF wrist (palm down).  Spatula = handle (capsule r 12 mm, 12 cm, 30 g)
  with an end knob (sphere r 2 cm, its table support), a cranked neck and a thin blade (box
  6 x 8 x 0.12 cm, 20 g) that lies flat on the table while the handle is raised 2 cm (a turner's
  offset handle).  Toast = light box (8 x 8 x 1.6 cm, 30 g) lying ~20 cm in front of the blade.
  Its collision geometry (simplification: no corner / face collisions, which would cost 8+
  contacts) is four spheres at mid-thickness: two pivot spheres (r = half thickness) on its local
  +x edge and one inboard support sphere (same r, at x = -1.2 cm) -- it rests on these three both
  right side up and upside down, and rolls over the rounded +x edge -- plus a small crust sphere
  (r 4 mm) at the middle of its -x edge, 4 mm clear of the table (a slightly raised crust edge),
  which is the only part the blade can touch.  Flip = slide the blade under the crust, lift it,
  rotating the toast about its +x edge past vertical, let it fall.  Goal = the ideal roll-over:
  upside down (Rz(yaw) Ry(pi)), centre moved by TOAST_ROLL = 2 * 3.2 cm + pi * 0.8 cm (the
  pivot spheres roll) = 8.9 cm along the toast's local +x (~world +y).
  Contacts (15): 4 fingertip capsules vs handle; 4 fingertip end spheres vs table; spatula knob +
  2 blade front-corner spheres vs table; 3 toast support spheres vs table; blade (box) vs the crust
  sphere.  Not modelled: hand vs toast (excluded: only the blade can move the toast), blade vs the
  toast's support spheres / faces (the blade passes through the toast's underside), the blade's
  rear edge vs table, finger links, palm vs tool (excluded), palm vs table (wrist floor
  wz >= SPAT_FLOOR).  Frictions: toast supports 1.0, blade / crust 0.3.
  Cost: toast pose (pos weight W_POS = 100, quaternion weight 1); Qf = 40 Q.  Success: metric
  < 0.01 (~1 cm and ~6 deg).  Seeds randomize the spatula's (x, y, yaw) and the toast's (x, y, yaw).
  Witness (~5.2 s of 6 s): pinch (cage) the handle, lift, measure the grasp, put the blade flat on
  the table behind the crust (refined on rollouts), slide under, rotate the toast with the blade
  tangent to its underside up to th_max, push on along the blade normal, retract (th_max / push
  grid-searched on rollouts)."""
  HORIZON = 150
  DT = 0.04
  W_POS = 100.0
  FINGER_KP = 3.0

  @classmethod
  def env(cls):
    s = SP; hx, hy, hz = TOAST_H; r = TOAST_R
    tbl = table_geom(2 | 4 | 8)
    bc, bh = np.array(s['blade_c']), np.array(s['blade_h'])
    yb = bc[1] + bh[1] - s['r_bs']; zb = bc[2] - bh[2] + s['r_bs']; xb = bh[0] - 0.004
    tool = f"""    <body name="spatula" pos="0 0 0.3">
      <freejoint name="spatula"/>
      <geom name="sp_handle" type="capsule" size="{s['r_handle']}" fromto="0 {s['y0']} 0 0 {s['y1']} 0" mass="{s['m_handle']}" contype="1" conaffinity="0" rgba="0.2 0.2 0.25 1" friction="1"/>
      <geom name="sp_knob" type="sphere" size="{s['r_end']}" pos="0 {s['y0']} 0" mass="0.005" contype="2" conaffinity="0" rgba="0.2 0.2 0.25 1"/>
      <geom name="sp_neck" type="capsule" size="{s['r_neck']}" fromto="0 {s['y1']} 0 0 {bc[1] - bh[1]} {bc[2]}" mass="{s['m_neck']}" contype="0" conaffinity="0" rgba="0.7 0.7 0.75 1"/>
      <geom name="sp_blade" type="box" size="{fmt(bh)}" pos="{fmt(bc)}" mass="{s['m_blade']}" contype="0" conaffinity="16" rgba="0.75 0.75 0.8 1" friction="0.3"/>
      <geom name="sp_b1" type="sphere" size="{s['r_bs']}" pos="{xb} {yb} {zb}" mass="0" contype="2" conaffinity="0" group="3"/>
      <geom name="sp_b2" type="sphere" size="{s['r_bs']}" pos="{-xb} {yb} {zb}" mass="0" contype="2" conaffinity="0" group="3"/>
    </body>"""
    toast = f"""    <body name="toast" pos="0 0 0.3">
      <freejoint name="toast"/>
      <geom name="toast" type="box" size="{hx} {hy} {hz}" mass="0.03" contype="0" conaffinity="0" rgba="0.85 0.65 0.35 1"/>
      <geom name="toast_top" type="box" size="{hx * 0.8} {hy * 0.8} 0.0005" pos="0 0 {hz}" mass="0" contype="0" conaffinity="0" rgba="0.55 0.35 0.15 1"/>
      <geom name="toast_p1" type="sphere" size="{r}" pos="{TOAST_PIV} {hy - r} 0" mass="0" contype="4" conaffinity="0" group="3" friction="1"/>
      <geom name="toast_p2" type="sphere" size="{r}" pos="{TOAST_PIV} {-(hy - r)} 0" mass="0" contype="4" conaffinity="0" group="3" friction="1"/>
      <geom name="toast_m" type="sphere" size="{r}" pos="{TOAST_XM} 0 0" mass="0" contype="4" conaffinity="0" group="3" friction="1"/>
      <geom name="toast_l" type="sphere" size="{TOAST_RL}" pos="{-TOAST_XL} 0 0" mass="0" contype="16" conaffinity="0" group="3" friction="0.3"/>
    </body>"""
    return '\n'.join([tbl, tool, toast])

  @classmethod
  def build(cls):
    ranges = ((-0.4, 0.4), (-0.4, 0.4), (SPAT_FLOOR, 0.4), (-1.6, 1.6), (-1.6, 1.6), (-3.2, 3.2))
    return hand_model(cls.env(), ranges=ranges, timestep=cls.DT, finger_kp=cls.FINGER_KP,
                      palm_exclude=('spatula',))

  def __init__(self, seed=0):
    rng = np.random.default_rng(7300 + seed)
    m = self.build()
    self.sp_xy = np.array([0.0, -0.10]) + rng.uniform(-0.02, 0.02, 2)
    self.sp_yaw = rng.uniform(-0.2, 0.2)
    self.toast_xy = np.array([0.0, 0.08]) + rng.uniform(-0.015, 0.015, 2)
    self.toast_yaw = np.pi / 2 + rng.uniform(-0.2, 0.2)     # its flip direction (local +x) ~ world +y
    nx = m.nq + m.nv
    x0 = np.zeros(nx)
    a = _qpos_adr(m, 'spatula'); b = _qpos_adr(m, 'toast')
    x0[a:a + 3] = [*self.sp_xy, SP['r_end'] - 1e-4]
    x0[a + 3:a + 7] = yaw_quat(self.sp_yaw)
    x0[b:b + 3] = [*self.toast_xy, TOAST_R - 1e-4]
    x0[b + 3:b + 7] = yaw_quat(self.toast_yaw)
    w = _qpos_adr(m, 'wx'); f = _qpos_adr(m, 'ffj0')
    x0[w:w + 6] = [0.0, -0.20, 0.22, 0, 0, 0]
    self.q_open = pinch_fingers(m, SP['r_handle'], -0.03, dz=self.W_DZ)
    x0[f:f + 16] = self.q_open
    g = x0.copy()
    Rz = rot_z(self.toast_yaw)
    g[b:b + 3] = x0[b:b + 3] + Rz @ np.array([TOAST_ROLL, 0, 0])
    g[b + 3:b + 7] = mat_to_quat(Rz @ _ry(np.pi))
    Q = pose_Q(m, 'toast', w_pos=self.W_POS, w_rot=1.0)
    u_ref = np.concatenate([x0[w:w + 6], self.q_open])
    R = np.diag([1e-2] * 3 + [1e-3] * 3 + [1e-2] * 16)
    super().__init__(model=m, horizon=self.HORIZON, x_0=x0, x_goal=g, Q=Q, Qf=40 * Q,
                     u_ref=u_ref, R=R, contact_smoothing=0.002)

  # -- witness ------------------------------------------------------------------------------
  W_DZ = -0.008
  W_GRASP_Y = -0.085        # pinch centre along the handle (tool frame)
  W_T = dict(top=0.5, down=0.9, squeeze=1.3, lift=1.7, above=2.4, touch=2.8, slide=3.3, flip=4.5,
             push=4.9, retract=5.2)
  W_ALPHA = 0.0             # blade nose-down tilt while sliding under the crust sphere
  W_PRESS = 0.004           # blade front pressed this far into the table while sliding
  W_SLIDE = 0.02            # blade-front start distance behind the lift sphere
  W_UNDER = 0.015           # lift-sphere contact this far behind the blade front

  def witness(self, verbose=False):
    """Pinch the handle, lift, measure the grasp; put the blade's front edge on the table behind
    the toast's lift sphere (blade along the toast's +x, nose down), slide it under the sphere,
    rotate the toast about its pivot edge with the blade tangent to its bottom face, push it past
    vertical, retract.  Alignment and the flip schedule are refined on true-law rollouts."""
    from simtools import keyframes, sim_true
    m = self.model; T = self.W_T; dt = self.DT
    psi = self.sp_yaw; Rz = rot_z(psi)
    hp = np.array([*self.sp_xy, SP['r_end']])
    wgrasp = hp + Rz @ np.array([0, self.W_GRASP_Y, 0]) + Rz @ PINCH_OFF
    qs = pinch_fingers(m, SP['r_handle'], 0.015, dz=self.W_DZ)
    pick = [(T['top'], _wrist_kf(wgrasp + [0, 0, 0.04], psi)), (T['down'], _wrist_kf(wgrasp)),
            (T['squeeze'], fingers_dict(qs)), (T['lift'], _wrist_kf(wgrasp + [0, 0, 0.08]))]
    k_lift = int(round(T['lift'] / dt)) + 3
    X = sim_true(self, keyframes(self, pick))
    rel = tool_rel(m, X[k_lift], 'spatula')
    a = _qpos_adr(m, 'spatula'); b = _qpos_adr(m, 'toast')
    bc, bh = np.array(SP['blade_c']), np.array(SP['blade_h'])
    front_t = bc + [0, bh[1], bh[2]]                   # blade front edge, top surface (tool frame)
    front_b = bc + [0, bh[1], -bh[2]]                  # blade front edge, bottom surface
    tz = self.toast_yaw; d = rot_z(tz) @ [1, 0, 0]
    p_t = np.array([*self.toast_xy, TOAST_R])
    piv = p_t + d * TOAST_PIV                          # pivot axis point (toast local +x edge, mid-height)
    ref0 = np.array([0.0, 0.0, psi])
    R_flat = rot_z(tz - np.pi / 2)                     # blade pointing along d, flat

    def lift_pt(th):                                   # crust (lift) sphere centre at toast angle th
      return piv + _ry_about(d, th) @ (-d * (TOAST_PIV + TOAST_XL))

    def plan(off, th_max, push, alpha=None):
      alpha = self.W_ALPHA if alpha is None else alpha
      kf = list(pick); ref = ref0
      def go(t, p, R, pt, o=np.zeros(3)):
        nonlocal ref
        pw, aw = wrist_for(p + o, R, rel, pt, ref=ref)
        ref = aw
        kf.append((t, _wrist_kf(pw, aw[2], aw[0], aw[1])))
      Ra = _ry_about(d, alpha) @ R_flat
      l0 = lift_pt(0.0)
      start = l0 - d * (TOAST_RL + self.W_SLIDE); start[2] = -self.W_PRESS
      go(T['above'], start + [0, 0, 0.03] - d * 0.02, Ra, front_b, off)
      go(T['touch'], start, Ra, front_b, off)
      under = l0 + d * self.W_UNDER; under[2] = -self.W_PRESS
      go(T['slide'], under, Ra, front_b, off)
      nst = 8
      for i in range(1, nst + 1):
        th = th_max * i / nst
        Rb = _ry_about(d, th) @ R_flat                 # blade tangent to the toast's bottom face
        n = _ry_about(d, th) @ [0, 0, 1]
        c = lift_pt(th) - n * TOAST_RL                 # crust-sphere contact point on the blade top
        go(T['slide'] + (T['flip'] - T['slide']) * i / nst, c, Rb, front_t - [0, self.W_UNDER, 0], off)
      # push on past vertical along the blade normal, then retract
      Rb = _ry_about(d, th_max) @ R_flat
      n = _ry_about(d, th_max) @ [0, 0, 1]
      c = lift_pt(th_max) - n * TOAST_RL
      go(T['push'], c + n * push, Rb, front_t - [0, self.W_UNDER, 0], off)
      go(T['retract'], c + n * push - d * 0.04 + [0, 0, 0.04], Rb, front_t - [0, self.W_UNDER, 0], off)
      us = keyframes(self, kf)
      return us, sim_true(self, us)

    off = np.zeros(3)
    k_touch = int(round(T['touch'] / dt)) + 3
    for it in range(4):
      us, X = plan(off, 1.35, 0.03)
      Rt = quat_to_mat(X[k_touch, a + 3:a + 7])
      pf = X[k_touch, a:a + 3] + Rt @ front_b
      l0 = lift_pt(0.0); target = l0 - d * (TOAST_RL + self.W_SLIDE); target[2] = 0.0
      err = pf - target
      if verbose:
        print(f'  align it {it}: blade front err {np.round(err * 1000, 2)} mm, metric {metric_of(self, X[-1]):.4f}')
      off[:2] -= err[:2]
      off[2] -= max(err[2], 0.0)       # lower the blade until its front edge touches the table
    best = None
    for th_max in (1.2, 1.35, 1.5):
      for push in (0.0, 0.02, 0.04):
        us, X = plan(off, th_max, push)
        e = metric_of(self, X[-1]) if np.isfinite(X).all() else np.inf
        if verbose:
          print(f'  th_max {th_max} push {push}: metric {e:.4f} toast {np.round(X[-1, b:b + 7], 3)}')
        if best is None or e < best[0]:
          best = (e, us)
    return best[1]


def _ry_about(axis_x, th):
  """Rotation by th about the horizontal axis perpendicular to `axis_x` (toast flip axis):
  rotates `axis_x` downward (the toast's local y axis when axis_x is its local x)."""
  ax = np.cross([0, 0, 1], axis_x); ax = ax / np.linalg.norm(ax)
  K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
  return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def metric_of(task, x):
  from simtools import metric
  return metric(task, x)
